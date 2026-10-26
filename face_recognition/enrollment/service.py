"""One photo is sufficient; an explicit UUID updates that person's photo."""
from face_recognition.core.images import decode_photo
from face_recognition.storage.people import validate_name, validate_person_id


class EnrollmentService:
    def __init__(self, backend, people_store, max_image_bytes: int, max_image_dimension: int = 1600):
        self.backend = backend
        self.people_store = people_store
        self.max_image_bytes = max_image_bytes
        self.max_image_dimension = max_image_dimension

    def enroll(self, name: str, image: bytes, person_id: str | None = None) -> dict:
        name = validate_name(name)
        if person_id is not None:
            person_id = validate_person_id(person_id)
        frame = decode_photo(image, self.max_image_bytes, self.max_image_dimension)
        return self.enroll_frame(name, frame, person_id)

    def enroll_frame(self, name: str, frame, person_id: str | None = None) -> dict:
        name = validate_name(name)
        faces = self.backend.detect_and_embed(frame)
        if len(faces) != 1:
            raise ValueError(f"enrollment requires exactly one face; detected {len(faces)}")
        return self.people_store.enroll(name, faces[0].embedding, person_id).to_dict()
