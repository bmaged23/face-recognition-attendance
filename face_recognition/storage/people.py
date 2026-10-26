"""Qdrant person records: UUID, English name, and one reference embedding."""
import uuid
from face_recognition.core.models import Person


def validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("name must be an English string")
    name = name.strip()
    if not name or not name.isascii() or not name.isprintable():
        raise ValueError("name must be a non-empty English name using printable ASCII characters")
    return name


def validate_person_id(person_id: str) -> str:
    try:
        return str(uuid.UUID(person_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("person ID must be a UUID returned by enrollment") from exc


class PeopleStore:
    def __init__(self, settings, client=None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.client = client if client is not None else (
            QdrantClient(path=str(settings.qdrant_path)) if settings.qdrant_path is not None
            else QdrantClient(url=settings.qdrant_url, timeout=60)
        )
        self.collection = settings.collection
        self.signature = settings.model_signature
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
        else:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if getattr(vectors, "size", None) != 512 or getattr(vectors, "distance", None) != Distance.COSINE:
                raise ValueError("Collection must contain unnamed 512-dimensional cosine vectors")
            # Old project records and another model's embeddings require re-enrollment.
            points, _ = self.client.scroll(self.collection, limit=1, with_payload=True, with_vectors=False)
            if points:
                self._person(points[0])

    def _person(self, point):
        payload = point.payload or {}
        if payload.get("model_signature") != self.signature:
            raise ValueError("Collection uses an incompatible model/schema; select a new collection and enroll people again")
        person_id = validate_person_id(payload.get("person_id"))
        if person_id != str(point.id):
            raise ValueError("Stored person ID does not match its embedding ID")
        return Person(person_id, validate_name(payload.get("name", "")))

    def enroll(self, name: str, embedding, person_id: str | None = None) -> Person:
        from qdrant_client.models import PointStruct
        name = validate_name(name)
        if person_id is None:
            person_id = str(uuid.uuid4())
        else:
            person_id = validate_person_id(person_id)
            previous = self.client.retrieve(self.collection, ids=[person_id], with_payload=True)
            if not previous:
                raise ValueError("person ID does not exist in this collection")
            self._person(previous[0])
        values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        self.client.upsert(self.collection, wait=True, points=[PointStruct(
            id=person_id, vector=values,
            payload={"person_id": person_id, "name": name, "model_signature": self.signature},
        )])
        return Person(person_id, name)

    def match(self, embedding, confidence: float, min_margin: float = 0.0):
        import math
        if not math.isfinite(min_margin) or not 0 <= min_margin <= 1:
            raise ValueError('min_margin must be in [0, 1]')
        values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        result = self.client.query_points(
            collection_name=self.collection, query=values, limit=2 if min_margin else 1,
            # The second candidate must not be filtered out by the absolute threshold.
            score_threshold=None if min_margin else confidence,
            with_payload=["person_id", "name", "model_signature"],
        )
        if not result.points:
            return None
        point = result.points[0]
        if not math.isfinite(point.score) or point.score < confidence:
            return None
        if min_margin and len(result.points) > 1:
            runner_up = result.points[1]
            self._person(runner_up)
            if not math.isfinite(runner_up.score) or point.score - runner_up.score < min_margin:
                return None
        return self._person(point), float(point.score)
