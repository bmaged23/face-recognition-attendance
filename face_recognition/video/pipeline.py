"""Process a file or a live camera with exactly the same recognition engine."""
from datetime import datetime, timedelta
import math
from pathlib import Path
import time

from face_recognition.storage.attendance import AttendanceStore, parse_timestamp
from face_recognition.storage.json_io import write_json
from face_recognition.video.results import SessionResults
from face_recognition.config import Settings


def relative_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class StreamPipeline:
    def __init__(self, engine, settings=None):
        self.engine = engine
        self.settings = settings if settings is not None else Settings()

    def run_video(self, video_path, *, start_time: str | None = None, **options):
        video_path = Path(video_path)
        if not video_path.is_file():
            raise ValueError(f"video does not exist: {video_path}")
        start = parse_timestamp(start_time) if start_time else None
        if options.get("action") and start is None:
            raise ValueError("video attendance requires start_time with a timezone")
        self._validate_paths(video_path, options)
        return self._run(str(video_path), live=False, start=start, **options)

    def run_camera(self, source=0, **options):
        return self._run(source, live=True, start=None, **options)

    @staticmethod
    def _validate_paths(source, options):
        # Input files must never be replaced by an output or attendance file.
        targets = [options.get(key) for key in ("output", "annotated_video", "attendance_file")]
        if any(Path(target).resolve() == source.resolve() for target in targets if target):
            raise ValueError("output paths must differ from the input video")

    def _run(self, source, *, live, start, output,
             action=None, attendance_file=None, sample_every=None,
             annotated_video=None, display=False, max_seconds=None):
        if sample_every is None:
            sample_every = self.settings.sample_every
        if not math.isfinite(sample_every) or sample_every < 0:
            raise ValueError("sample_every must be non-negative; 0 means every frame")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be a positive number")
        if action not in {None, "checkin", "checkout"}:
            raise ValueError("action must be checkin or checkout")
        if bool(action) != bool(attendance_file):
            raise ValueError("attendance requires both action and attendance_file")
        paths = [Path(p).resolve() for p in (output, attendance_file, annotated_video) if p]
        if len(paths) != len(set(paths)):
            raise ValueError("JSON output, attendance state, and annotated video must use different paths")
        if annotated_video:
            from face_recognition.video.encoding import require_ffmpeg
            require_ffmpeg()
        import cv2
        from face_recognition.core.engine import RecognitionEngine
        from face_recognition.video.tracking import IoURecognitionEngine
        tracker = None
        if (self.settings.use_iou_tracking and isinstance(self.engine, RecognitionEngine)
                and callable(getattr(self.engine.backend, 'detect_faces', None))):
            tracker = IoURecognitionEngine(self.engine, self.settings.iou_threshold,
                    sample_every if sample_every > 0 else self.settings.recognition_refresh_seconds)
        capture = cv2.VideoCapture(source)
        writer = None
        records = SessionResults(AttendanceStore(attendance_file) if action else None, action)
        count = 0
        next_sample = 0.0
        matches = []
        started = time.monotonic()
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {'camera' if live else 'video'}: {source}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                if not live:
                    raise ValueError("video has no valid frame rate; timestamps cannot be calculated")
                fps = 30.0
            write_json(output, [])
            while True:
                if max_seconds is not None and live and time.monotonic() - started >= max_seconds:
                    break
                ok, frame = capture.read()
                if not ok:
                    if count == 0 or live:
                        raise RuntimeError("camera/video did not return a readable frame")
                    break
                # File times use source presentation timestamps, falling back to frame index.
                elapsed = time.monotonic() - started if live else count / fps
                if not live:
                    reported = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    if math.isfinite(reported) and reported > 0:
                        elapsed = reported
                if max_seconds is not None and not live and elapsed >= max_seconds:
                    break
                if tracker is not None or elapsed + 1e-9 >= next_sample:
                    matches = tracker.recognize(frame, elapsed) if tracker is not None else self.engine.recognize(frame)
                    timestamp = datetime.now().astimezone() if live else (
                        start + timedelta(seconds=elapsed) if start else relative_timestamp(elapsed)
                    )
                    records.observe(matches, timestamp)
                    write_json(output, records.records)
                    next_sample = elapsed + sample_every
                if annotated_video or display:
                    from face_recognition.video.annotation import annotate
                    shown = annotate(frame.copy(), matches,
                                     show_confidence=self.settings.show_recognition_confidence)
                    if action:
                        from face_recognition.video.attendance_panel import annotate_attendance
                        shown = annotate_attendance(shown, records.records, action)
                    if annotated_video and writer is None:
                        path = Path(annotated_video)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot create annotated video: {path}")
                    if writer is not None:
                        writer.write(shown)
                    if display:
                        cv2.imshow("Face Recognition - press Q to stop", shown)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            count += 1
                            break
                count += 1
        except KeyboardInterrupt:
            # Ctrl+C ends a live session normally; observations have already been saved.
            if not live:
                raise
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        if writer is not None:
            from face_recognition.video.encoding import finalize_video
            finalize_video(annotated_video)
        return records.records
