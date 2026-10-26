"""Real HTTP smoke: pretrained enrollment, recognition, jobs and attendance.

Requires a running server, the prepared local research fixtures and an isolated
empty gallery. This writes synthetic attendance only under the chosen workdir.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import cv2
import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7323')
    parser.add_argument('--cases', type=Path, default=ROOT / 'input/recognition-evaluation')
    parser.add_argument('--workdir', type=Path, default=ROOT / 'output/api-validation')
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cases / 'manifest.json').read_text())
    events = []
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        def request(method, path, **kwargs):
            start = time.perf_counter()
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            events.append({'endpoint': path, 'status': response.status_code, 'seconds': time.perf_counter() - start})
            return response.json()
        assert request('GET', '/health')['status'] == 'ok'
        ready = request('GET', '/ready')
        assert ready['status'] == 'ready'
        # Enroll two independent neutral stills through the real upload decoder.
        persons = []
        for row in manifest['gallery'][:2]:
            raw = (args.cases / row['path']).read_bytes()
            person = request('POST', '/enroll', data={'name': f"Subject {row['identity']}"},
                             files={'image': ('reference.jpg', raw, 'image/jpeg')})
            persons.append((row['identity'], person))
        identity, person = persons[0]
        known = next(row for row in manifest['cases'] if row['identity'] == identity and row['camera'] == 2)
        probe = args.cases / known['source']
        response = request('POST', '/recognize/image', files={'image': ('probe.jpg', probe.read_bytes(), 'image/jpeg')})
        assert person['id'] in {row['id'] for row in response}, response
        unknown = next(row for row in manifest['cases'] if not row['known'] and row['camera'] == 2)
        response = request('POST', '/recognize/image', files={'image': ('unknown.jpg', (args.cases / unknown['source']).read_bytes(), 'image/jpeg')})
        assert response == [], response
        image = cv2.imread(str(probe))
        height, width = image.shape[:2]
        video = (args.workdir / 'known-person.avi').resolve()
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
        assert writer.isOpened()
        for _ in range(4):
            writer.write(image)
        writer.release()
        attendance = (args.workdir / 'synthetic-attendance.json').resolve()
        if attendance.exists():
            raise RuntimeError('Use a new isolated workdir to avoid overwriting an earlier test')
        for action, hour in [('checkin', '09'), ('checkout', '17')]:
            submitted = request('POST', '/video-recognition', json={
                'video_path': str(video), 'mode': 'attendance', 'action': action,
                'start_time': f'2026-10-26T{hour}:00:00+00:00',
                'attendance_file': str(attendance), 'create_annotated_video': True})
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                job = request('GET', '/jobs/' + submitted['job_id'])
                if job['status'] in {'done', 'failed'}:
                    break
                time.sleep(.25)
            assert job['status'] == 'done', job
            assert person['id'] in {row['id'] for row in job['result']['records']}, job
            downloaded = request('GET', '/jobs/' + submitted['job_id'] + '/artifacts/results')
            assert downloaded == job['result']['records']
            partial = client.get('/jobs/' + submitted['job_id'] + '/artifacts/video',
                                 headers={'Range': 'bytes=0-127'})
            assert partial.status_code == 206 and len(partial.content) == 128
            annotated = Path(job['result']['annotated_video'])
            capture = cv2.VideoCapture(str(annotated))
            try:
                assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
            finally:
                capture.release()
        rows = json.loads(attendance.read_text())
        row = next(row for row in rows if row['id'] == person['id'])
        assert row['checkin'] and row['checkout']
        assert set(row) == {'id', 'name', 'checkin', 'checkout'}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'readiness': ready,
              'checks': ['liveness', 'real model readiness', 'two photo enrollments', 'known recognition',
                         'unseen identity rejection', 'checkin job', 'checkout job', 'four-frame annotations',
                         'persistent attendance schema', 'result downloads', 'video byte ranges'],
              'enrolled_people': persons, 'events': events,
              'scope': 'Synthetic repeated-frame clips from real research probes; not a temporal accuracy benchmark'}
    (args.workdir / 'api-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
