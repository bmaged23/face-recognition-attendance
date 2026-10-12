"""Per-stream IoU association with periodic recognition and current-frame boxes."""
from dataclasses import dataclass
import math
from face_recognition.core.models import Match


def intersection_over_union(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


@dataclass
class Track:
    match: Match
    recognized_at: float


class IoURecognitionEngine:
    def __init__(self, engine, threshold=0.5, refresh_seconds=1.0):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('IoU threshold must be in (0, 1]')
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0:
            raise ValueError('Recognition refresh interval must be positive')
        self.engine = engine
        self.threshold = threshold
        self.refresh_seconds = refresh_seconds
        self.tracks = []

    def recognize(self, frame, timestamp):
        detections = self.engine.backend.detect_faces(frame)
        pairs = []
        for index, detection in enumerate(detections):
            for previous, track in enumerate(self.tracks):
                score = intersection_over_union(detection.bbox, track.match.bbox)
                if score >= self.threshold:
                    pairs.append((score, index, previous))
        assignments, used = {}, set()
        # A previous recognition can be associated with only one current detection.
        for _, index, previous in sorted(pairs, reverse=True):
            if index not in assignments and previous not in used:
                assignments[index] = self.tracks[previous]
                used.add(previous)
        fresh, pending = {}, []
        for index, detection in enumerate(detections):
            track = assignments.get(index)
            if track is None or timestamp - track.recognized_at >= self.refresh_seconds:
                pending.append(index)
            else:
                fresh[index] = Track(Match(detection.bbox, track.match.person, track.match.confidence),
                                     track.recognized_at)
        recognized = self.engine.recognize_detections(frame, [detections[i] for i in pending]) if pending else []
        if len(recognized) != len(pending):
            raise RuntimeError('Recognition must return one result per selected face')
        for index, match in zip(pending, recognized):
            fresh[index] = Track(match, timestamp)
        # Drop unmatched tracks immediately; no stale boxes/labels after disappearance.
        self.tracks = [fresh[index] for index in range(len(detections))]
        return [track.match for track in self.tracks]
