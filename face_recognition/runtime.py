"""Construct one shared backend and repository for a workflow."""
from dataclasses import dataclass
from face_recognition.config import Settings
from face_recognition.core.engine import RecognitionEngine
from face_recognition.enrollment.service import EnrollmentService


@dataclass
class Runtime:
    settings: Settings
    backend: object
    people: object

    def engine(self, confidence: float | None = None):
        return RecognitionEngine(self.backend, self.people,
                                 self.settings.confidence if confidence is None else confidence,
                                 self.settings.recognition_min_margin)

    def enrollment(self):
        return EnrollmentService(self.backend, self.people, self.settings.max_image_bytes,
                                 self.settings.max_image_dimension)


def create_runtime(settings: Settings) -> Runtime:
    from face_recognition.core.backend import FaceBackend
    from face_recognition.storage.people import PeopleStore
    backend = FaceBackend(settings)
    return Runtime(settings, backend, PeopleStore(settings))
