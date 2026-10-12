"""Small CLI adapters; business logic lives in the package."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

from face_recognition.config import Settings
from face_recognition.core.engine import validate_confidence
from face_recognition.storage.attendance import parse_timestamp


def _model_options(parser):
    parser.add_argument("--collection", help="Qdrant collection; use the same one for enrollment and recognition")
    parser.add_argument("--family", choices=["adaface", "insightface"])
    parser.add_argument("--model", help="Model name from the selected family")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])


def _settings(args):
    settings = Settings.from_env()
    updates = {key: value for key, value in {
        "family": args.family, "model_name": args.model,
        "collection": args.collection, "device": args.device,
    }.items() if value is not None}
    if args.family and not args.model and args.family != settings.family:
        updates["model_name"] = "buffalo_l" if args.family == "insightface" else "ir_100_webface12m"
    if (args.family or args.model) and not args.collection:
        updates["collection"] = f"people_{updates.get('family', settings.family)}_{updates.get('model_name', settings.model_name)}"
    return replace(settings, **updates)


def _report_error(exc):
    print(f"Error: {exc}", file=sys.stderr)
    return 1


def enrollment_main(argv=None):
    parser = argparse.ArgumentParser(description="Enroll one person from one photo and generate a UUID.")
    parser.add_argument("--name", required=True, help="English display name")
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--person-id", help="Existing enrollment UUID to update; omit to create a person")
    _model_options(parser)
    args = parser.parse_args(argv)
    try:
        from face_recognition.storage.people import validate_name, validate_person_id
        validate_name(args.name)
        if args.person_id:
            validate_person_id(args.person_id)
        settings = _settings(args)
        if args.photo.stat().st_size > settings.max_image_bytes:
            raise ValueError(f"photo exceeds {settings.max_image_bytes} bytes")
        photo = args.photo.read_bytes()
        from face_recognition.runtime import create_runtime
        result = create_runtime(settings).enrollment().enroll(args.name, photo, args.person_id)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        return _report_error(exc)


def stream_main(live: bool, argv=None):
    parser = argparse.ArgumentParser(description=f"{'Live camera' if live else 'Video'} face recognition with optional attendance.")
    if live:
        parser.add_argument("--source", default="0", help="Camera index, RTSP URL, or stream URL")
    else:
        parser.add_argument("--video", required=True, type=Path)
        parser.add_argument("--start-time", help="Recording start as ISO 8601 with timezone; required for attendance")
    parser.add_argument("--mode", choices=["recognize", "attendance"], default="recognize")
    parser.add_argument("--action", choices=["checkin", "checkout"])
    parser.add_argument("--confidence", type=float, help="Minimum cosine similarity in [0, 1]")
    parser.add_argument("--sample-every", type=float, help="Recognition interval; with IoU tracking detection runs every frame. 0 uses the tracking refresh setting, or recognizes every frame when tracking is off")
    parser.add_argument("--output", type=Path, help="Result JSON file")
    parser.add_argument("--attendance-file", type=Path, help="Shared attendance state for checkin and checkout runs")
    parser.add_argument("--annotated-video", type=Path, help="Optional annotated MP4 (without audio)")
    parser.add_argument("--display", action="store_true", help="Show a preview; press Q to stop")
    parser.add_argument("--max-seconds", type=float, help="Stop after this many source seconds (video) or wall seconds (camera)")
    _model_options(parser)
    args = parser.parse_args(argv)
    if args.mode == "recognize" and (args.action or args.attendance_file):
        parser.error("--action and --attendance-file require --mode attendance")
    if args.mode == "attendance" and not args.action:
        parser.error("attendance requires --action checkin or checkout")
    if not live and args.mode == "attendance" and not args.start_time:
        parser.error("video attendance requires --start-time with a timezone")
    try:
        settings = _settings(args)
        confidence = settings.confidence if args.confidence is None else args.confidence
        validate_confidence(confidence)
        if not live:
            if not args.video.is_file():
                raise ValueError(f"video does not exist: {args.video}")
            if args.start_time:
                parse_timestamp(args.start_time)
        from face_recognition.runtime import create_runtime
        from face_recognition.video.pipeline import StreamPipeline
        day = datetime.now().astimezone().date() if live or not args.start_time else parse_timestamp(args.start_time).date()
        output = args.output or Path("output") / ("camera_results.json" if live else "video_results.json")
        attendance_file = None
        if args.mode == "attendance":
            attendance_file = args.attendance_file or Path("output") / "attendance" / f"{day}.json"
        options = dict(output=output, action=args.action, attendance_file=attendance_file,
                       sample_every=args.sample_every, annotated_video=args.annotated_video,
                       display=args.display, max_seconds=args.max_seconds)
        pipeline = StreamPipeline(create_runtime(settings).engine(confidence), settings=settings)
        if live:
            source = int(args.source) if args.source.isdecimal() else args.source
            results = pipeline.run_camera(source, **options)
        else:
            results = pipeline.run_video(args.video, start_time=args.start_time, **options)
        print(json.dumps(results, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        return _report_error(exc)
