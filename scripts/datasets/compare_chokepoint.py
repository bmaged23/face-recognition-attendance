"""Compare policies using cached real GPU embeddings; render improved full videos."""
import json
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from face_recognition.config import Settings
from face_recognition.core.models import FaceDetection, FaceSample
from face_recognition.core.engine import RecognitionEngine
from face_recognition.storage.people import PeopleStore
from face_recognition.video.tracking import IoURecognitionEngine
from face_recognition.video.pipeline import StreamPipeline

OUT = ROOT / 'output/chokepoint_accuracy'
DATA = ROOT / 'input/chokepoint'


class CachedBackend:
    """Replay ordered detections and GPU embeddings, with no model inference."""
    def __init__(self, data):
        self.data, self.index, self.embedded = data, 0, 0

    def detect_faces(self, frame):
        start, stop = self.data['offsets'][self.index:self.index + 2]
        self.index += 1
        return [FaceDetection(tuple(map(int, box)), feature)
                for box, feature in zip(self.data['boxes'][start:stop], self.data['embeddings'][start:stop])]

    def embed_faces(self, frame, detections):
        self.embedded += len(detections)
        return [FaceSample(d.bbox, d.aligned) for d in detections]

    def detect_and_embed(self, frame):
        return self.embed_faces(frame, self.detect_faces(frame))


def main():
    settings = Settings.from_env()
    signature = settings.model_signature
    manifest = json.loads((OUT / 'gallery.json').read_text())
    if manifest['model_signature'] != signature:
        raise RuntimeError('Gallery model mismatch')
    galleries = np.load(OUT / 'gallery.npz')
    destination = OUT / 'improved'
    destination.mkdir(exist_ok=True)
    attendance = destination / 'attendance.json'
    attendance.write_text('[]')
    report = {'accuracy_measured_on_crowded_videos': False, 'videos': []}
    for sequence in ('P2E_S5_C2', 'P2L_S5_C2'):
        meta = json.loads((OUT / f'{sequence}_faces.json').read_text())
        if meta['model_signature'] != signature:
            raise RuntimeError('Video cache model mismatch')
        with np.load(OUT / f'{sequence}_faces.npz') as archive:
            data = {key: archive[key] for key in archive.files}
        result = {'sequence': sequence, 'frames': meta['frames'], 'policies': {}}
        for gallery, margin in [('old', 0), ('improved', 0.05)]:
            client = QdrantClient(':memory:')
            store = PeopleStore(settings, client=client)
            client.upsert(store.collection, points=[PointStruct(id=p['id'], vector=e.tolist(),
                payload={'person_id': p['id'], 'name': p['name'], 'model_signature': signature})
                for p, e in zip(manifest['people'], galleries[gallery])])
            reference = None
            for refresh in (None, 1.0, 0.5):
                backend = CachedBackend(data)
                engine = RecognitionEngine(backend, store, 0.4, margin)
                tracker = IoURecognitionEngine(engine, 0.5, refresh) if refresh else None
                labels, named, disagreements = [], 0, 0
                for i in range(meta['frames']):
                    matches = tracker.recognize(None, i / meta['fps']) if tracker else engine.recognize(None)
                    labels.extend(m.person.id if m.person else None for m in matches)
                    named += sum(m.person is not None for m in matches)
                if reference is None:
                    reference = labels
                else:
                    disagreements = sum(a != b for a, b in zip(reference, labels))
                key = f'{gallery}_refresh_{refresh}'
                result['policies'][key] = {'named_faces': named, 'total_faces': len(labels),
                    'embedding_requests': backend.embedded, 'disagreements_with_every_frame': disagreements}
                print(sequence, key, result['policies'][key], flush=True)
            if gallery == 'improved':
                destination = OUT / 'improved'
                destination.mkdir(exist_ok=True)
                for notice in ('SOURCE_AND_LICENSE.html', 'MODIFIED_DATA_NOTICE.txt'):
                    shutil.copy2(DATA / notice, destination / notice)
                backend = CachedBackend(data)
                engine = RecognitionEngine(backend, store, 0.4, margin)
                # Shorter refresh limits the time a mistaken IoU association persists.
                config = replace(settings, recognition_refresh_seconds=0.5, sample_every=0)
                action = 'checkin' if sequence.startswith('P2E') else 'checkout'
                StreamPipeline(engine, config).run_video(DATA / f'{sequence}.mp4',
                    output=destination / f'{sequence}.json', sample_every=0,
                    annotated_video=destination / f'{sequence}_annotated.mp4', action=action,
                    attendance_file=attendance,
                    start_time=f'2026-10-12T{"09" if action == "checkin" else "17"}:00:00+03:00')
                if backend.index != meta['frames']:
                    raise RuntimeError('Cache/video frame counts differ')
            client.close()
        report['videos'].append(result)
        (OUT / 'crowded_comparison.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
