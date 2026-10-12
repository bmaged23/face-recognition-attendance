"""Main entry point for webcams and camera streams."""
from face_recognition.cli import stream_main

if __name__ == "__main__":
    raise SystemExit(stream_main(live=True))
