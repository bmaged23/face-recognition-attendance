"""Small, stable output records for both video and camera sessions."""
from datetime import datetime


class SessionResults:
    def __init__(self, attendance_store=None, action: str | None = None):
        if attendance_store is not None and action not in {"checkin", "checkout"}:
            raise ValueError("attendance requires checkin or checkout")
        if attendance_store is None and action is not None:
            raise ValueError("action is only valid in attendance mode")
        self.attendance_store = attendance_store
        self.action = action
        self._records = {}

    def observe(self, matches, timestamp: datetime | str):
        for match in matches:
            person = match.person
            if person is None or person.id in self._records:
                continue
            if self.attendance_store is None:
                time = timestamp.isoformat(timespec="milliseconds") if isinstance(timestamp, datetime) else timestamp
                record = {**person.to_dict(), "time": time}
            else:
                if not isinstance(timestamp, datetime):
                    raise ValueError("attendance requires an absolute recording timestamp")
                record = self.attendance_store.record(person, self.action, timestamp)
            self._records[person.id] = record
        return self.records

    @property
    def records(self):
        return list(self._records.values())
