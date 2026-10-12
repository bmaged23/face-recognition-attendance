"""Re-enroll retained color references and render both full attendance videos.

Uses the approved AdaFace model, threshold, ambiguity margin, and IoU policy.
Source videos, original color stills, and the enrollment manifest must exist.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageOps
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from face_recognition.config import Settings
from face_recognition.core.backend import FaceBackend
from face_recognition.core.engine import RecognitionEngine
from face_recognition.storage.people import PeopleStore
from face_recognition.video.pipeline import StreamPipeline


def main():
    data = ROOT / 'input/chokepoint'
    destination = ROOT / 'output/chokepoint_accuracy/improved'
    destination.mkdir(parents=True, exist_ok=True)
    settings = replace(Settings.from_env(), device='cuda', confidence=0.4,
                       recognition_min_margin=0.05, sample_every=0,
                       use_iou_tracking=True, iou_threshold=0.5,
                       recognition_refresh_seconds=0.5)
    manifest = json.loads((data / 'color_enrollment/manifest.json').read_text())
    if manifest['model_signature'] != settings.model_signature:
        raise RuntimeError('Color reference model signature mismatch')
    backend = FaceBackend(settings)
    print(f'Model: {settings.model_name}; device: {backend.device}', flush=True)
    client = QdrantClient(':memory:')
    try:
        store = PeopleStore(settings, client=client)
        points = []
        for row in manifest['people']:
            # Match evaluate_chokepoint.gallery exactly, before its JPEG export.
            photo = data / 'stills' / row['expression'] / f"ID{row['dataset_id']}.JPG"
            with Image.open(photo) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1600, 1600))
                frame = np.asarray(image)[:, :, ::-1].copy()
            faces = backend.detect_and_embed(frame)
            if len(faces) != 1:
                raise RuntimeError(f'Reference must have exactly one face: {photo}')
            points.append(PointStruct(id=row['id'], vector=faces[0].embedding.tolist(),
                         payload={'person_id': row['id'], 'name': row['name'],
                                  'model_signature': store.signature}))
        client.upsert(store.collection, points=points)
        print(f'Enrolled {len(points)} retained identities', flush=True)
        # Progress messages do not affect inference, tracking, or source timestamps.
        detect = backend.detect_faces
        state = {'frames': 0, 'started': time.monotonic()}
        def progress(frame):
            faces = detect(frame)
            state['frames'] += 1
            if state['frames'] % 100 == 0:
                print(f"Processed {state['frames']} frames in {time.monotonic() - state['started']:.1f}s", flush=True)
            return faces
        backend.detect_faces = progress
        engine = RecognitionEngine(backend, store, 0.4, 0.05)
        attendance = destination / 'attendance.json'
        attendance.write_text('[]')
        for sequence, action, hour in [('P2E_S5_C2', 'checkin', '09'), ('P2L_S5_C2', 'checkout', '17')]:
            state.update(frames=0, started=time.monotonic())
            print(f'Rendering {sequence}: {action}', flush=True)
            records = StreamPipeline(engine, settings).run_video(
                data / f'{sequence}.mp4', output=destination / f'{sequence}.json',
                annotated_video=destination / f'{sequence}_annotated.mp4',
                action=action, attendance_file=attendance,
                start_time=f'2026-10-12T{hour}:00:00+03:00')
            print(f"Finished {sequence}: {state['frames']} frames, {len(records)} identities", flush=True)

    finally:
        client.close()


if __name__ == '__main__':
    main()
