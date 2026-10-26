"""Evaluate actual MTCNN/AdaFace + Qdrant matching against frozen eye labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from qdrant_client import QdrantClient
from face_recognition.config import Settings
from face_recognition.core.backend import FaceBackend
from face_recognition.storage.people import PeopleStore


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class ResourceSampler:
    """Sample this process RSS/CPU and whole-device GPU usage, excluding cloud."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        before_cpu, before_wall = time.process_time(), time.perf_counter()
        while not self.stop.wait(0.5):
            now_cpu, now_wall = time.process_time(), time.perf_counter()
            cpu = 100 * (now_cpu - before_cpu) / (now_wall - before_wall)
            before_cpu, before_wall = now_cpu, now_wall
            status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
            sample = {'rss_mib': int(status['VmRSS'].split()[0]) / 1024, 'cpu_percent': cpu}
            try:
                run = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                if run.returncode == 0:
                    memory, utilization = map(float, run.stdout.splitlines()[0].split(','))
                    sample.update(whole_device_vram_mib=memory, whole_device_gpu_percent=utilization)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.samples.append(sample)

    def summary(self):
        result = {'sample_count': len(self.samples), 'interval_seconds': 0.5,
                  'cpu_definition': '100 percent is one logical core; sampled process RSS, whole-device GPU'}
        for key in ('rss_mib', 'cpu_percent', 'whole_device_vram_mib', 'whole_device_gpu_percent'):
            values = [s[key] for s in self.samples if key in s]
            if values:
                result[key] = {'mean': statistics.mean(values), 'peak': max(values)}
        return result


def summarize(rows):
    known = [r for r in rows if r['known']]
    unknown = [r for r in rows if not r['known']]
    detected_known = [r for r in known if r['detected']]
    detected_unknown = [r for r in unknown if r['detected']]
    correct = sum(r['prediction'] == r['identity'] for r in known)
    wrong = sum(r['prediction'] is not None and r['prediction'] != r['identity'] for r in known)
    false_accepts = sum(r['prediction'] is not None for r in unknown)
    durations = sorted(r['total_seconds'] for r in rows)
    return {'probes': len(rows), 'known_probes': len(known), 'unknown_probes': len(unknown),
            'detected_known': len(detected_known), 'detected_unknown': len(detected_unknown),
            'correct_known_accepted': correct, 'wrong_known_accepted': wrong,
            'known_rejected_after_detection': sum(r['prediction'] is None for r in detected_known),
            'unknown_false_accepts': false_accepts,
            'known_end_to_end_correct_rate': correct / len(known) if known else None,
            'known_correct_rate_given_detection': correct / len(detected_known) if detected_known else None,
            'unknown_false_accept_rate_given_detection': false_accepts / len(detected_unknown) if detected_unknown else None,
            'latency_seconds': {'median': statistics.median(durations),
                                'p95': float(np.percentile(durations, 95)), 'total': sum(durations)},
            'component_median_seconds': {key: statistics.median(r[key] for r in rows)
                                         for key in ('detection_seconds', 'embedding_seconds', 'search_seconds')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'input/recognition-evaluation')
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'held-out-view'])
    parser.add_argument('--mtcnn-final-threshold', type=float, default=Settings.mtcnn_final_threshold)
    parser.add_argument('--mtcnn-scale-factor', type=float, default=Settings.mtcnn_scale_factor)
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args()
    manifest_path = args.cases / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['gallery'] + manifest['cases']:
        path = args.cases / row.get('path', row.get('source'))
        if sha256(path) != row['sha256']:
            raise RuntimeError(f'Fixture checksum mismatch: {path}')
    if args.split:
        manifest['cases'] = [row for row in manifest['cases'] if row['split'] == args.split]
    torch.set_num_threads(2)
    sampler = ResourceSampler()
    sampler.thread.start()
    try:
        start = time.perf_counter()
        backend = FaceBackend(Settings(device=args.device, model_path=args.model_path,
                              mtcnn_final_threshold=args.mtcnn_final_threshold,
                              mtcnn_scale_factor=args.mtcnn_scale_factor))
        startup = time.perf_counter() - start
        def synchronize():
            if args.device == 'cuda':
                torch.cuda.synchronize()
        client = QdrantClient(':memory:')
        people = PeopleStore(backend.settings, client)
        gallery_rows = []
        for row in manifest['gallery']:
            with Image.open(args.cases / row['path']) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1600, 1600))
                frame = np.asarray(image)[:, :, ::-1].copy()
            faces = backend.detect_and_embed(frame)
            if len(faces) != 1:
                gallery_rows.append({'identity': row['identity'], 'enrolled': False, 'detected_faces': len(faces)})
                continue
            identity = row['identity']
            # Stable UUIDs ensure gallery records are deterministic across devices/runs.
            from qdrant_client.models import PointStruct
            person_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f'chokepoint:{identity}'))
            client.upsert(people.collection, points=[PointStruct(id=person_id, vector=faces[0].embedding.tolist(),
                          payload={'person_id': person_id, 'name': f'Subject {identity}', 'model_signature': people.signature})])
            gallery_rows.append({'identity': identity, 'enrolled': True})
        # Warm the probe shape once; warmup is excluded from per-probe latency.
        backend.detect_and_embed(cv2.imread(str(args.cases / manifest['cases'][0]['source'])))
        synchronize()
        results = []
        for row in manifest['cases']:
            frame = cv2.imread(str(args.cases / row['source']))
            if frame is None:
                raise RuntimeError(f'Invalid probe: {row["case_id"]}')
            synchronize(); start = time.perf_counter()
            detections = backend.detect_faces(frame)
            synchronize(); detection_time = time.perf_counter() - start
            suitable = [d for d in detections if all(d.bbox[0] <= x <= d.bbox[2] and d.bbox[1] <= y <= d.bbox[3]
                                                    for x, y in row['eyes'])]
            prediction, score, embedding_time, search_time = None, None, 0., 0.
            if suitable:
                center = np.mean(row['eyes'], axis=0)
                selected = min(suitable, key=lambda d: np.linalg.norm(center - np.array(
                    [(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2])))
                stage = time.perf_counter()
                face = backend.embed_faces(frame, [selected])[0]
                synchronize(); embedding_time = time.perf_counter() - stage
                stage = time.perf_counter()
                match = people.match(face.embedding, backend.settings.confidence, backend.settings.recognition_min_margin)
                search_time = time.perf_counter() - stage
                if match:
                    prediction, score = match[0].name.removeprefix('Subject '), match[1]
            results.append({**row, 'detected': bool(suitable), 'prediction': prediction, 'score': score,
                            'detection_seconds': detection_time, 'embedding_seconds': embedding_time,
                            'search_seconds': search_time, 'total_seconds': time.perf_counter() - start})
            if len(results) % 20 == 0:
                print(f'{args.device}: {len(results)}/{len(manifest["cases"])} probes', flush=True)
        client.close()
        report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'device': args.device,
                  'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  'torch': torch.__version__, 'threads': 2, 'fixture_sha256': sha256(manifest_path),
                  'detector_policy': {'final_threshold': args.mtcnn_final_threshold,
                                      'scale_factor': args.mtcnn_scale_factor},
                  'model_signature': people.signature, 'model_startup_seconds': startup,
                  'confidence': backend.settings.confidence, 'min_margin': backend.settings.recognition_min_margin,
                  'gallery': gallery_rows, 'overall': summarize(results),
                  'by_split': {split: summarize([r for r in results if r['split'] == split])
                               for split in sorted({r['split'] for r in results})},
                  'samples': results, 'limitations': manifest['limitations']}
    finally:
        sampler.stop.set(); sampler.thread.join()
    report['resources'] = sampler.summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'overall': report['overall'], 'resources': report['resources']}, indent=2))


if __name__ == '__main__':
    main()
