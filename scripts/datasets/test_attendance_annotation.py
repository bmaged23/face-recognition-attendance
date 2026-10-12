"""Render attendance panels on existing GPU-recognized crowded test videos.

Replays the saved first-confirmation events; does not rerun face inference.
The existing video already contains face boxes, names and percentages.
"""
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from face_recognition.core.models import Match, Person
from face_recognition.storage.attendance import AttendanceStore, parse_timestamp
from face_recognition.video.attendance_panel import annotate_attendance, panel_layout
from face_recognition.video.encoding import finalize_video
from face_recognition.video.results import SessionResults


def seconds(timestamp):
    hours, minutes, seconds = timestamp.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main():
    source = ROOT / 'output' / 'chokepoint_test'
    out = ROOT / 'output' / 'chokepoint_attendance_test'
    out.mkdir(parents=True, exist_ok=True)
    for notice in ('SOURCE_AND_LICENSE.html', 'MODIFIED_DATA_NOTICE.txt'):
        shutil.copy2(source / notice, out / notice)
    state = out / f'attendance_{uuid.uuid4().hex[:8]}.json'
    report = {'method': 'Replay saved recognition events over already GPU-annotated video',
              'synthetic_start_times': True, 'attendance_file': str(state), 'videos': []}
    for sequence, action, start in [
            ('P2E_S5_C2', 'checkin', '2026-10-12T09:00:00+03:00'),
            ('P2L_S5_C2', 'checkout', '2026-10-12T17:00:00+03:00')]:
        events = sorted(json.loads((source / f'{sequence}.json').read_text()), key=lambda r: seconds(r['time']))
        capture = cv2.VideoCapture(str(source / f'{sequence}_annotated.mp4'))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open annotated source: {sequence}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = out / f'{sequence}_{action}_annotated.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f'Cannot create {video}')
        results = SessionResults(AttendanceStore(state), action)
        begin = parse_timestamp(start)
        index, count, last = 0, 0, None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                elapsed = count / fps
                while index < len(events) and seconds(events[index]['time']) <= elapsed + 1e-6:
                    event = events[index]
                    person = Person(event['id'], event['name'])
                    results.observe([Match((0, 0, 0, 0), person)], begin + timedelta(seconds=seconds(event['time'])))
                    index += 1
                last = annotate_attendance(frame, results.records, action)
                writer.write(last)
                count += 1
        finally:
            writer.release()
            capture.release()
        if index != len(events):
            raise RuntimeError('Some saved recognition events exceed the video duration')
        finalize_video(video)
        (out / f'{sequence}_{action}.json').write_text(json.dumps(results.records, indent=2))
        cv2.imwrite(str(out / f'{sequence}_{action}_preview.jpg'), last)
        report['videos'].append({'video': str(video), 'action': action, 'frames': count,
                                 'names_added': index, 'visible_capacity': panel_layout(width, height).capacity,
                                 'overflowed': index > panel_layout(width, height).capacity})
        print(json.dumps(report['videos'][-1]), flush=True)
    report['attendance'] = json.loads(state.read_text())
    (out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
