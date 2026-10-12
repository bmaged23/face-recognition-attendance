"""GPU smoke test: one enrollment image per subject, held-out session videos.

Download fadg0.zip and mabw0.zip from the official VidTIMIT links into
input/vidtimit first. Uses an isolated local Qdrant database, no server needed.
Generated videos use 25 fps for this test; source audio is omitted.
"""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from qdrant_client import QdrantClient
from face_recognition.config import Settings
from face_recognition.core.backend import FaceBackend
from face_recognition.core.engine import RecognitionEngine
from face_recognition.enrollment.service import EnrollmentService
from face_recognition.storage.people import PeopleStore
from face_recognition.video.pipeline import StreamPipeline


def main():
    settings = Settings.from_env()
    backend = FaceBackend(settings)
    out = ROOT / 'output' / 'vidtimit_test'
    out.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(out / 'qdrant'))
    store = PeopleStore(settings, client=client)
    # Re-running replaces only this script's isolated test collection.
    if client.count(store.collection).count:
        client.delete_collection(store.collection)
        store = PeopleStore(settings, client=client)
    enrollment = EnrollmentService(backend, store, settings.max_image_bytes)
    manifest, videos = [], []
    # Fictional display names for testing; dataset codes remain source identifiers.
    for subject, display_name in [('fadg0', 'Emma'), ('mabw0', 'James')]:
        with zipfile.ZipFile(ROOT / 'input' / 'vidtimit' / f'{subject}.zip') as archive:
            sequences = {}
            for name in archive.namelist():
                parts = name.split('/')
                if len(parts) == 4 and parts[1] == 'video' and parts[3] and not parts[2].startswith('head'):
                    sequences.setdefault(parts[2], []).append(name)
            sentences = sorted(sequences)
            reference = sorted(sequences[sentences[0]])[len(sequences[sentences[0]]) // 2]
            image = archive.read(reference)
            photo = ROOT / 'input' / 'vidtimit' / f'{subject}_enrollment.jpg'
            photo.write_bytes(image)
            person = enrollment.enroll(display_name, image)
            manifest.append(dict(person, source=reference, photo=str(photo)))
            for sentence in sentences[6:8]:
                path = ROOT / 'input' / 'vidtimit' / f'{subject}_{sentence}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (512, 384))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create {path}')
                for name in sorted(sequences[sentence]):
                    frame = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                    writer.write(frame)
                writer.release()
                videos.append((subject, person['id'], path))
    engine = RecognitionEngine(backend, store, settings.confidence, settings.recognition_min_margin)
    report = {'model': settings.model_name, 'device': backend.device,
              'gpu': torch.cuda.get_device_name(0), 'confidence': settings.confidence,
              'enrollment': manifest, 'videos': []}
    for subject, expected, path in videos:
        stats = {'sampled_frames': 0, 'detected_faces': 0, 'correct_matches': 0,
                 'wrong_matches': 0, 'unknown_faces': 0, 'scores': []}
        class ObservedEngine:
            def recognize(self, frame):
                matches = engine.recognize(frame)
                stats['sampled_frames'] += 1
                stats['detected_faces'] += len(matches)
                for match in matches:
                    if match.person is None:
                        stats['unknown_faces'] += 1
                    elif match.person.id == expected:
                        stats['correct_matches'] += 1
                        stats['scores'].append(match.confidence)
                    else:
                        stats['wrong_matches'] += 1
                return matches
        records = StreamPipeline(ObservedEngine()).run_video(
            path, output=out / f'{path.stem}.json', sample_every=0.5,
            annotated_video=out / f'{path.stem}_annotated.mp4')
        report['videos'].append(dict(subject=subject, video=str(path), records=records, **stats))
        print(json.dumps(report['videos'][-1]), flush=True)
    # Use a fresh state file per run without removing any previous report.
    from datetime import datetime
    state = out / f'attendance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    pipeline = StreamPipeline(engine)
    for action, timestamp in [('checkin', '2026-10-12T09:00:00+03:00'),
                              ('checkout', '2026-10-12T17:00:00+03:00')]:
        for subject, _, path in videos[::2]:
            pipeline.run_video(path, start_time=timestamp, action=action,
                               attendance_file=state, output=out / f'{subject}_{action}.json',
                               sample_every=1.0)
    report['attendance'] = json.loads(state.read_text())
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    client.close()
    print(f'Report: {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
