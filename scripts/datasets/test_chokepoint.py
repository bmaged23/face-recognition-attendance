"""Recognize multiple faces in real crowded ChokePoint camera-2 videos.

Uses one labeled reference from S1 per enrolled ID, held-out crowded S5 probes.
No crowded ground truth is supplied; counts are observations, not accuracy.
See input/chokepoint/SOURCE_AND_LICENSE.html and MODIFIED_DATA_NOTICE.txt.
"""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from qdrant_client import QdrantClient
from face_recognition.config import Settings
from face_recognition.core.backend import FaceBackend
from face_recognition.core.engine import RecognitionEngine
from face_recognition.enrollment.service import EnrollmentService
from face_recognition.storage.people import PeopleStore
from face_recognition.video.pipeline import StreamPipeline
from face_recognition.video.tracking import IoURecognitionEngine

NAMES = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Robin', 'Sam',
         'Jamie', 'Charlie', 'Avery', 'Riley', 'Cameron', 'Drew', 'Quinn',
         'Blake', 'Harper', 'Skyler', 'Rowan', 'Reese', 'Emerson', 'Dakota',
         'Finley', 'Sage', 'Parker', 'Hayden', 'Kendall', 'Logan', 'Ellis', 'Remy']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color-references', action='store_true', help='Use the quality-selected color stills; preserve baseline outputs')
    args = parser.parse_args()
    root = ROOT / 'input' / 'chokepoint'
    out = ROOT / 'output' / ('chokepoint_color_test' if args.color_references else 'chokepoint_test')
    out.mkdir(parents=True, exist_ok=True)
    for name in ['SOURCE_AND_LICENSE.html', 'MODIFIED_DATA_NOTICE.txt']:
        shutil.copy2(root / name, out / name)
    archive_path = root / 'P2E.tar.xz'
    with archive_path.open('rb') as file:
        assert hashlib.file_digest(file, 'md5').hexdigest() == 'bd0cd0719b625f4927e9450a9053d8d1'
    with tarfile.open(archive_path) as archive:
        archive.extractall(root / 'references', filter='data')
    settings = Settings.from_env()
    backend = FaceBackend(settings)
    client = QdrantClient(path=str(out / 'qdrant'))
    store = PeopleStore(settings, client=client)
    if client.count(store.collection).count:
        client.delete_collection(store.collection)
        store = PeopleStore(settings, client=client)
    enrollment = EnrollmentService(backend, store, settings.max_image_bytes)
    report = {'model': settings.model_name, 'device': backend.device,
              'sample_every': settings.sample_every,
              'show_recognition_confidence': settings.show_recognition_confidence,
              'use_iou_tracking': settings.use_iou_tracking,
              'iou_threshold': settings.iou_threshold,
              'recognition_refresh_seconds': settings.recognition_refresh_seconds,
              'confidence': settings.confidence, 'accuracy_measured': False,
              'reference_sequence': 'P2E_S1_C2', 'enrollment': [],
              'failed_enrollment': [], 'videos': []}
    # The original gallery consists of labeled, aligned 96x96 grayscale PGM faces.
    # Upsample and pad only to allow the shared detector to locate those tight crops.
    source = root / 'references' / 'P2E_S1_C2'
    folders = sorted(p for p in source.iterdir() if p.is_dir())
    for index, folder in enumerate(folders):
        if args.color_references:
            manifest = json.loads((root / 'color_enrollment/manifest.json').read_text())
            row = next(p for p in manifest['people'] if p['dataset_id'] == folder.name)
            if manifest['model_signature'] != settings.model_signature:
                raise RuntimeError('Color references must be evaluated with the active model')
            person = enrollment.enroll_frame(row['name'], cv2.imread(row['photo']))
            report['enrollment'].append(dict(person, dataset_id=folder.name, source=row['source'], photo=row['photo']))
            report['reference_sequence'] = 'ChokePoint color stills (one photo per person)'
            continue
        files = sorted(folder.glob('*.pgm'))
        person = None
        for candidate in np.linspace(0, len(files) - 1, min(12, len(files)), dtype=int):
            photo = files[candidate]
            frame = cv2.imread(str(photo))
            frame = cv2.resize(frame, (288, 288), interpolation=cv2.INTER_CUBIC)
            frame = cv2.copyMakeBorder(frame, 48, 48, 48, 48, cv2.BORDER_REPLICATE)
            try:
                person = enrollment.enroll_frame(NAMES[index % len(NAMES)], frame)
            except ValueError:
                continue
            destination = root / f'{folder.name}_enrollment.jpg'
            cv2.imwrite(str(destination), frame)
            report['enrollment'].append(dict(person, dataset_id=folder.name,
                                              source=str(photo.relative_to(root)), photo=str(destination)))
            print(f'Enrolled {folder.name} as {person["name"]}', flush=True)
            break
        if person is None:
            report['failed_enrollment'].append(folder.name)
    if len(report['enrollment']) < 2:
        raise RuntimeError('Need at least two enrolled references for the crowded test')
    report['recognition_min_margin'] = settings.recognition_min_margin
    engine = RecognitionEngine(backend, store, settings.confidence, settings.recognition_min_margin)
    for sequence in ['P2E_S5_C2', 'P2L_S5_C2']:
        tracked = IoURecognitionEngine(engine, settings.iou_threshold,
                    settings.sample_every or settings.recognition_refresh_seconds) if settings.use_iou_tracking else None
        stats = {'sampled_frames': 0, 'detected_faces': 0, 'recognized_faces': 0,
                 'unknown_faces': 0, 'frames_with_multiple_faces': 0,
                 'frames_with_multiple_recognized_people': 0, 'max_faces_in_frame': 0}
        class ObservedEngine:
            def recognize(self, frame):
                matches = tracked.recognize(frame, stats['sampled_frames'] / 30) if tracked else engine.recognize(frame)
                recognized = [m for m in matches if m.person]
                stats['sampled_frames'] += 1
                stats['detected_faces'] += len(matches)
                stats['recognized_faces'] += len(recognized)
                stats['unknown_faces'] += len(matches) - len(recognized)
                stats['frames_with_multiple_faces'] += int(len(matches) > 1)
                stats['frames_with_multiple_recognized_people'] += int(len({m.person.id for m in recognized}) > 1)
                stats['max_faces_in_frame'] = max(stats['max_faces_in_frame'], len(matches))
                return matches
        records = StreamPipeline(ObservedEngine(), settings=settings).run_video(
            root / f'{sequence}.mp4', output=out / f'{sequence}.json',
            sample_every=0 if tracked else settings.sample_every,
            annotated_video=out / f'{sequence}_annotated.mp4')
        report['videos'].append(dict(sequence=sequence, records=records, **stats))
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['videos'][-1]), flush=True)
    client.close()
    print(f'Report: {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
