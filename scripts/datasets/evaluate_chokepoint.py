"""Evaluate one-photo enrollment and calibrate on labeled ChokePoint recordings.

Gallery: original grayscale references versus independently selected color stills.
Development: S2 camera 2. Held-out evaluation: S2 camera 1 (same recording,
different view; not an independent recording). Crowded S5 has no supplied labels.
Caches real GPU embeddings to compare policies without repeating inference.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageOps
from face_recognition.config import Settings
from face_recognition.core.backend import FaceBackend

DATA = ROOT / 'input' / 'chokepoint'
OUT = ROOT / 'output' / 'chokepoint_accuracy'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def gallery(backend, signature):
    baseline = DATA / 'baseline_enrollment.json'
    if baseline.exists():
        records = json.loads(baseline.read_text())
    else:
        report = ROOT / 'output/chokepoint_test/report.json'
        if not report.exists():
            raise RuntimeError('Run scripts/datasets/test_chokepoint.py to prepare baseline references first')
        records = json.loads(report.read_text())['enrollment']
        save_json(baseline, records)
    old, improved, manifest = [], [], []
    photos = DATA / 'color_enrollment'
    photos.mkdir(exist_ok=True)
    for row in records:
        original = cv2.imread(row['photo'])
        faces = backend.detect_and_embed(original)
        if len(faces) != 1:
            raise RuntimeError(f'Original reference does not have one face: {row["dataset_id"]}')
        old.append(faces[0].embedding)
        candidates = []
        for expression in ('Neutral', 'Smile'):
            path = DATA / 'stills' / expression / f'ID{row["dataset_id"]}.JPG'
            if not path.exists():
                continue
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1600, 1600))
                frame = np.asarray(image)[:, :, ::-1].copy()
            detections = backend.detect_faces(frame)
            if len(detections) != 1:
                continue
            aligned = np.asarray(detections[0].aligned)
            gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            candidates.append((sharpness, expression, path, frame, detections))
        if not candidates:
            raise RuntimeError(f'No usable color photo for dataset ID {row["dataset_id"]}')
        quality, expression, source, frame, detections = max(candidates, key=lambda c: c[0])
        improved.append(backend.embed_faces(frame, detections)[0].embedding)
        destination = photos / f'{row["dataset_id"]}.jpg'
        cv2.imwrite(str(destination), frame)
        manifest.append(dict(row, photo=str(destination), original_photo=row['photo'],
                             source=str(source), expression=expression, sharpness=quality))
        print(f'Color reference: {row["name"]} ({expression})', flush=True)
    np.savez_compressed(OUT / 'gallery.npz', old=np.stack(old), improved=np.stack(improved))
    save_json(OUT / 'gallery.json', {'model_signature': signature, 'people': manifest})
    save_json(photos / 'manifest.json', {'model_signature': signature, 'people': manifest})


def probes(backend, signature):
    archive_path = DATA / 'P2E_S2.tar.xz'
    with archive_path.open('rb') as file:
        assert hashlib.file_digest(file, 'md5').hexdigest() == 'cc464106ef29e363b67fdfbfd5e25ed9'
    with tarfile.open(archive_path) as outer:
        for camera in (1, 2):
            folder = DATA / 'evaluation_frames'
            member = next(m for m in outer if m.name == f'P2E_S2_C{camera}.tar.xz')
            with tarfile.open(fileobj=outer.extractfile(member), mode='r|xz') as inner:
                inner.extractall(folder, filter='data')
    gt = tarfile.open(DATA / 'groundtruth.tar.xz')
    people = json.loads((OUT / 'gallery.json').read_text())['people']
    identities = {r['dataset_id'] for r in people}
    for camera in (2, 1):
        options = {identity: [] for identity in identities}
        for name in sorted(gt.getnames()):
            if f'P2E_S2_C{camera}.' not in name or not name.endswith('.xml'):
                continue
            sequence = Path(name).stem
            for frame in ET.fromstring(gt.extractfile(name).read()):
                for person in frame.findall('person'):
                    identity = person.attrib['id']
                    if identity not in options:
                        continue
                    eyes = [person.find('leftEye'), person.find('rightEye')]
                    if any(eye is None for eye in eyes):
                        continue
                    path = DATA / 'evaluation_frames' / sequence / f'{int(frame.attrib["number"]):08d}.jpg'
                    if path.exists():
                        options[identity].append((path, [(float(e.attrib['x']), float(e.attrib['y'])) for e in eyes]))
        embeddings, labels, selected, misses = [], [], [], []
        for identity in sorted(options):
            candidates = options[identity]
            if not candidates:
                misses.append({'id': identity, 'reason': 'No labeled frames'})
                continue
            for index in np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int):
                path, eyes = candidates[index]
                frame = cv2.imread(str(path))
                detections = backend.detect_faces(frame)
                suitable = [d for d in detections if all(d.bbox[0] <= x <= d.bbox[2] and d.bbox[1] <= y <= d.bbox[3] for x, y in eyes)]
                if not suitable:
                    misses.append({'id': identity, 'source': str(path), 'reason': 'Annotated eyes not inside a detected face'})
                    continue
                center = np.mean(eyes, axis=0)
                detection = min(suitable, key=lambda d: np.linalg.norm(center - np.array([(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2])))
                embeddings.append(backend.embed_faces(frame, [detection])[0].embedding)
                labels.append(identity)
                selected.append({'id': identity, 'source': str(path), 'bbox': detection.bbox})
            print(f'Camera {camera}, labeled subject {identity}', flush=True)
        if not embeddings:
            raise RuntimeError(f'No labeled detections for camera {camera}')
        np.savez_compressed(OUT / f'camera{camera}.npz', embeddings=np.stack(embeddings), labels=np.array(labels))
        save_json(OUT / f'camera{camera}.json', {'model_signature': signature, 'selected': selected, 'misses': misses})
    gt.close()


def measurements(scores, labels, threshold, margin):
    order = np.argsort(-scores, axis=1)
    best = order[:, 0]
    rows = np.arange(len(scores))
    accepted = (scores[rows, best] >= threshold) & (scores[rows, best] - scores[rows, order[:, 1]] >= margin)
    correct = best == labels
    # Leave-one-identity-out proxy: remove the true gallery person from each query.
    # This is not a measured false-accept rate on entirely unseen people.
    excluded = scores.copy()
    excluded[rows, labels] = -np.inf
    unknown_order = np.argsort(-excluded, axis=1)
    unknown_accepted = (excluded[rows, unknown_order[:, 0]] >= threshold) & (excluded[rows, unknown_order[:, 0]] - excluded[rows, unknown_order[:, 1]] >= margin)
    return {'probes': len(scores), 'correct_accepted': int(np.sum(accepted & correct)),
            'wrong_identity_accepted': int(np.sum(accepted & ~correct)), 'rejected': int(np.sum(~accepted)),
            'top1_correct_without_threshold': int(np.sum(correct)),
            'leave_one_out_false_accepts': int(np.sum(unknown_accepted))}


def calibrate():
    gallery_data = np.load(OUT / 'gallery.npz')
    manifest = json.loads((OUT / 'gallery.json').read_text())
    indices = {r['dataset_id']: i for i, r in enumerate(manifest['people'])}
    dev = np.load(OUT / 'camera2.npz')
    labels = np.array([indices[i] for i in dev['labels']])
    scores = dev['embeddings'] @ gallery_data['improved'].T
    choices = []
    for threshold in np.arange(0.30, 0.901, 0.01):
        for margin in np.arange(0, 0.201, 0.01):
            result = measurements(scores, labels, float(threshold), float(margin))
            if result['wrong_identity_accepted'] / result['probes'] <= 0.01 and result['leave_one_out_false_accepts'] / result['probes'] <= 0.01:
                choices.append((result['correct_accepted'], -result['wrong_identity_accepted'],
                                -result['leave_one_out_false_accepts'], -margin, -threshold, result))
    if not choices:
        raise RuntimeError('Calibration found no setting meeting the proxy error constraints')
    choice = max(choices, key=lambda c: c[:5])
    threshold, margin = round(float(-choice[4]), 2), round(float(-choice[3]), 2)
    report = {'threshold': threshold, 'min_margin': margin,
              'recommended_threshold': 0.4, 'recommended_min_margin': 0.05,
              'recommendation_reason': 'Do not lower the existing threshold based on this limited development set. A 0.05 ambiguity margin preserves its measured accepted-correct coverage with color references.',
              'policy': 'Maximize correct accepted development probes with <=1% wrong identities and <=1% leave-one-out false-accept proxy',
              'limitations': 'Camera 1 is a held-out view of the same S2 recording, not a separate recording. Leave-one-out is a proxy, not an unseen-person benchmark. Accuracy is conditional on detection. S5 is unlabeled.',
              'model_signature': manifest['model_signature'], 'results': {}}
    for camera in (2, 1):
        data = np.load(OUT / f'camera{camera}.npz')
        ground = np.array([indices[i] for i in data['labels']])
        meta = json.loads((OUT / f'camera{camera}.json').read_text())
        report['results'][f'camera{camera}'] = {
            'old_grayscale_at_040': measurements(data['embeddings'] @ gallery_data['old'].T, ground, 0.4, 0),
            'color_at_040': measurements(data['embeddings'] @ gallery_data['improved'].T, ground, 0.4, 0),
            'color_calibrated': measurements(data['embeddings'] @ gallery_data['improved'].T, ground, threshold, margin),
            'color_recommended': measurements(data['embeddings'] @ gallery_data['improved'].T, ground, 0.4, 0.05),
            'threshold_sweep': {str(t): measurements(data['embeddings'] @ gallery_data['improved'].T, ground, t, 0.05) for t in (0.4, 0.45, 0.5, 0.55, 0.6)},
            'missed_labeled_samples': len(meta['misses'])}
    save_json(OUT / 'calibration.json', report)
    print(json.dumps(report, indent=2), flush=True)


def crowded(backend, signature):
    for sequence in ['P2E_S5_C2', 'P2L_S5_C2']:
        destination = OUT / f'{sequence}_faces.npz'
        source = DATA / f'{sequence}.mp4'
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if destination.exists():
            metadata = json.loads(destination.with_suffix('.json').read_text())
            if metadata['model_signature'] != signature:
                raise RuntimeError('Cached crowded embeddings use different weights')
            if metadata.get('source_sha256') not in (None, source_hash):
                raise RuntimeError('Cached crowded embeddings use a different video')
            print(f'Using cached GPU faces: {sequence}', flush=True)
            continue
        capture = cv2.VideoCapture(str(DATA / f'{sequence}.mp4'))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {sequence}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        offsets, boxes, embeddings = [0], [], []
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                samples = backend.detect_and_embed(frame)
                boxes.extend([s.bbox for s in samples])
                embeddings.extend([s.embedding for s in samples])
                offsets.append(len(boxes))
                count += 1
                if count % 100 == 0:
                    print(f'{sequence}: GPU detected and embedded {count} frames', flush=True)
        finally:
            capture.release()
        np.savez_compressed(destination, offsets=np.array(offsets), boxes=np.array(boxes).reshape(-1, 4),
                            embeddings=np.stack(embeddings) if embeddings else np.empty((0, 512)))
        save_json(destination.with_suffix('.json'), {'model_signature': signature, 'frames': count, 'fps': fps,
                                                   'source_sha256': source_hash})
        print(f'Cached {sequence}: {count} frames, {len(boxes)} faces', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['gallery', 'probes', 'calibrate', 'crowded'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'calibrate':
        calibrate()
        return
    settings = Settings.from_env()
    backend = FaceBackend(settings)
    signature = settings.model_signature
    if args.stage == 'gallery':
        gallery(backend, signature)
    else:
        if json.loads((OUT / 'gallery.json').read_text())['model_signature'] != signature:
            raise RuntimeError('Gallery cache uses different model weights')
        if args.stage == 'probes':
            probes(backend, signature)
        else:
            crowded(backend, signature)


if __name__ == '__main__':
    main()
