"""Validation for the general video workflow, without course-specific fields."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from face_recognition.storage.attendance import parse_timestamp


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_path: str
    mode: Literal["recognize", "attendance"] = "recognize"
    action: Literal["checkin", "checkout"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    start_time: str | None = None
    sample_every: float | None = Field(default=None, ge=0)
    attendance_file: str | None = None
    create_annotated_video: bool = False

    @model_validator(mode="after")
    def validate_workflow(self):
        if not Path(self.video_path).is_file():
            raise ValueError("video_path must point to an existing video file")
        if self.start_time:
            parse_timestamp(self.start_time)
        if self.mode == "attendance":
            if not self.action or not self.start_time:
                raise ValueError("attendance requires action and recording start_time with timezone")
        elif self.action or self.attendance_file:
            raise ValueError("action and attendance_file are only valid in attendance mode")
        return self
