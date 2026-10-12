"""Optional attendance state, independent of face detection and matching."""
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
from face_recognition.storage.json_io import write_json


def parse_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("timestamp must be ISO 8601 with a timezone") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone, for example +03:00")
    return timestamp


class AttendanceStore:
    """Use one file per attendance session/day; repeated sightings are idempotent."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _locked(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def record(self, person, action: str, timestamp: datetime) -> dict:
        if action not in {"checkin", "checkout"}:
            raise ValueError("attendance action must be checkin or checkout")
        parse_timestamp(timestamp.isoformat())
        with self._locked():
            records = json.loads(self.path.read_text()) if self.path.exists() else []
            if not isinstance(records, list):
                raise ValueError("attendance file must contain a JSON list")
            by_id = {}
            for row in records:
                if not isinstance(row, dict) or set(row) != {"id", "name", "checkin", "checkout"}:
                    raise ValueError("attendance file must contain only id, name, checkin, and checkout")
                if row["id"] in by_id:
                    raise ValueError("attendance file contains a duplicate person ID")
                by_id[row["id"]] = dict(row)
            row = by_id.setdefault(person.id, {**person.to_dict(), "checkin": None, "checkout": None})
            row["name"] = person.name
            # Each action records the first confirmed sighting, never toggles state.
            current = parse_timestamp(row[action]) if row[action] else None
            if current is None or timestamp < current:
                row[action] = timestamp.isoformat(timespec="milliseconds")
            if row["checkin"] and row["checkout"] and parse_timestamp(row["checkout"]) < parse_timestamp(row["checkin"]):
                raise ValueError("checkout precedes checkin; verify recording times and attendance file")
            write_json(self.path, list(by_id.values()))
            return dict(row)
