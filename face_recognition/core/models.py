"""Internal values; public JSON records are deliberately smaller."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    aligned: Any


@dataclass(frozen=True)
class FaceSample:
    bbox: tuple[int, int, int, int]
    embedding: Any


@dataclass(frozen=True)
class Person:
    id: str
    name: str

    def to_dict(self):
        return {"id": self.id, "name": self.name}


@dataclass(frozen=True)
class Match:
    bbox: tuple[int, int, int, int]
    person: Person | None
    confidence: float = 0.0
