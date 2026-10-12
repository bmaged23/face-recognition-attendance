"""One recognition flow used by images, videos, and live cameras."""
import math
from face_recognition.core.models import Match


def validate_confidence(confidence: float):
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")


class RecognitionEngine:
    def __init__(self, backend, people_store, confidence: float, min_margin: float = 0.0):
        validate_confidence(confidence)
        self.backend = backend
        self.people_store = people_store
        self.confidence = confidence
        if not math.isfinite(min_margin) or not 0 <= min_margin <= 1:
            raise ValueError('min_margin must be in [0, 1]')
        self.min_margin = min_margin

    def recognize(self, frame) -> list[Match]:
        return self.match_faces(self.backend.detect_and_embed(frame))

    def recognize_detections(self, frame, detections) -> list[Match]:
        return self.match_faces(self.backend.embed_faces(frame, detections))

    def match_faces(self, faces) -> list[Match]:
        results = []
        for face in faces:
            candidate = self.people_store.match(face.embedding, self.confidence, self.min_margin) if self.min_margin else self.people_store.match(face.embedding, self.confidence)
            if candidate is None:
                results.append(Match(face.bbox, None))
            else:
                person, score = candidate
                # Compare the original score; rounding must never admit a low match.
                if math.isfinite(score) and score >= self.confidence:
                    results.append(Match(face.bbox, person, score))
                else:
                    results.append(Match(face.bbox, None))
        return results
