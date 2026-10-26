"""Real image codecs, in-memory Qdrant, and HTTP API; deterministic fake inference."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("cv2", "PIL", "qdrant_client", "fastapi", "httpx", "multipart"))

if DEPENDENCIES:
    import cv2
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient
    from face_recognition.api.app import create_app
    from face_recognition.config import Settings
    from face_recognition.core.images import decode_photo
    from face_recognition.core.models import FaceSample
    from face_recognition.runtime import Runtime
    from face_recognition.storage.people import PeopleStore
    from face_recognition.video.pipeline import StreamPipeline


@unittest.skipUnless(DEPENDENCIES, "Install requirements-dev.txt to run real-library integration tests")
class LibraryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.checkpoint = self.directory / "model.ckpt"
        self.checkpoint.write_bytes(b"test-model-fingerprint")
        self.settings = Settings(model_path=self.checkpoint)
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.people = PeopleStore(self.settings, self.client)
        self.vector = np.zeros(512, dtype=np.float32)
        self.vector[0] = 1.0
        outer = self
        class FakeBackend:
            device = "cpu"
            def detect_and_embed(self, frame):
                return [FaceSample((0, 0, 10, 10), outer.vector.copy())]
        self.runtime = Runtime(self.settings, FakeBackend(), self.people)
        image = Image.new("RGB", (32, 32), (255, 0, 0))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        self.photo = stream.getvalue()

    def test_photo_decoding_produces_bgr_pixels(self):
        frame = decode_photo(self.photo, 1_000_000)
        self.assertEqual(frame[0, 0].tolist(), [0, 0, 255])
        self.assertTrue(frame.flags.c_contiguous)

    def test_real_decoder_rejects_non_image(self):
        with self.assertRaisesRegex(ValueError, "valid image"):
            decode_photo(b"this is not an image", 100)

    def test_large_photo_is_bounded_before_detector_inference(self):
        stream = io.BytesIO()
        Image.new('RGB', (2400, 1200), 'red').save(stream, format='JPEG')
        frame = decode_photo(stream.getvalue(), 1_000_000, max_dimension=800)
        self.assertEqual(frame.shape, (400, 800, 3))

    def test_real_qdrant_enrollment_update_and_recognition(self):
        person = self.runtime.enrollment().enroll("Alice Smith", self.photo)
        rows = self.runtime.engine(0.9).recognize(decode_photo(self.photo, 100000))
        self.assertEqual(rows[0].person.id, person["id"])
        self.runtime.enrollment().enroll("Alice Jones", self.photo, person["id"])
        self.assertEqual(self.client.count(self.settings.collection).count, 1)
        match = self.people.match(self.vector, 0.9)
        self.assertEqual(match[0].name, "Alice Jones")

    def test_api_enrollment_and_image_recognition(self):
        app = create_app(runtime_factory=lambda _: self.runtime, settings=self.settings)
        with TestClient(app) as http:
            self.assertEqual(http.get("/health").json(), {"status": "ok"})
            result = http.post("/enroll", data={"name": "Alice Smith"}, files={"image": ("alice.png", self.photo, "image/png")})
            self.assertEqual(result.status_code, 200, result.text)
            person = result.json()
            self.assertEqual(set(person), {"id", "name"})
            recognition = http.post("/recognize/image?confidence=0.9", files={"image": ("probe.png", self.photo, "image/png")})
            self.assertEqual(recognition.status_code, 200, recognition.text)
            self.assertEqual(recognition.json()[0]["id"], person["id"])
            self.assertEqual(set(recognition.json()[0]), {"id", "name", "time"})
            bad_name = http.post("/enroll", data={"name": "أحمد"}, files={"image": ("alice.png", self.photo, "image/png")})
            self.assertEqual(bad_name.status_code, 422)
            bad_image = http.post("/recognize/image", files={"image": ("bad.png", b"not-an-image", "image/png")})
            self.assertEqual(bad_image.status_code, 422)

    def test_readiness_initializes_model_and_checks_gallery(self):
        from unittest.mock import Mock
        factory = Mock(return_value=self.runtime)
        with TestClient(create_app(runtime_factory=factory, settings=self.settings)) as http:
            self.assertEqual(http.get('/health').status_code, 200)
            factory.assert_not_called()
            for _ in range(2):
                response = http.get('/ready')
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['device'], 'cpu')
            factory.assert_called_once_with(self.settings)
            self.client.delete_collection(self.settings.collection)
            self.assertEqual(http.get('/ready').status_code, 503)

    def test_readiness_does_not_report_missing_model_as_ready(self):
        def unavailable(_):
            raise FileNotFoundError('/private/model/path')
        with TestClient(create_app(runtime_factory=unavailable, settings=self.settings)) as http:
            self.assertEqual(http.get('/health').status_code, 200)
            response = http.get('/ready')
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('/private/', response.text)

    def test_embedded_gallery_survives_client_restart(self):
        from dataclasses import replace
        settings = replace(self.settings, qdrant_path=self.directory / 'gallery')
        first = PeopleStore(settings)
        try:
            person = first.enroll('Alice Smith', self.vector)
        finally:
            first.client.close()
        second = PeopleStore(settings)
        try:
            found = second.match(self.vector, 0.9)
            self.assertEqual(found[0].id, person.id)
        finally:
            second.client.close()

    def make_video(self):
        video = self.directory / "test.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 32))
        self.assertTrue(writer.isOpened())
        for index in range(4):
            writer.write(np.full((32, 32, 3), index * 40, dtype=np.uint8))
        writer.release()
        return video

    def test_real_video_decode_and_annotation(self):
        person = self.runtime.enrollment().enroll("Alice Smith", self.photo)
        video = self.make_video()
        annotated = self.directory / "annotated.mp4"
        rows = StreamPipeline(self.runtime.engine(0.9)).run_video(
            video, output=self.directory / "results.json", annotated_video=annotated,
        )
        self.assertEqual(rows[0]["id"], person["id"])
        self.assertEqual(rows[0]["time"], "00:00:00.000")
        capture = cv2.VideoCapture(str(annotated))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
        finally:
            capture.release()
        import json
        import subprocess
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name,pix_fmt', '-of', 'json', str(annotated)],
            check=True, capture_output=True, text=True,
        )
        stream = json.loads(probe.stdout)['streams'][0]
        self.assertEqual(stream['codec_name'], 'h264')
        self.assertEqual(stream['pix_fmt'], 'yuv420p')

    def test_attendance_sidebar_is_enabled_only_for_attendance(self):
        from face_recognition.video.attendance_panel import annotate_attendance
        person = self.runtime.enrollment().enroll('Alice Smith', self.photo)
        source = self.make_video()
        for action in ('checkin', 'checkout', None):
            options = {}
            if action:
                options = dict(action=action, attendance_file=self.directory / 'attendance.json',
                               start_time=f'2026-10-12T{9 if action == "checkin" else 17:02d}:00:00+03:00')
            with patch('face_recognition.video.attendance_panel.annotate_attendance',
                       wraps=annotate_attendance) as panel:
                rows = StreamPipeline(self.runtime.engine(0.4)).run_video(
                    source, output=self.directory / f'{action}.json',
                    annotated_video=self.directory / f'{action}.mp4', **options)
            self.assertEqual(rows[0]['id'], person['id'])
            self.assertEqual(panel.call_count, 4 if action else 0)
            if action:
                self.assertEqual(panel.call_args.args[1][0]['name'], 'Alice Smith')
                self.assertEqual(panel.call_args.args[2], action)

    def test_background_api_video_attendance_uses_simple_json(self):
        person = self.runtime.enrollment().enroll("Alice Smith", self.photo)
        video = self.make_video()
        state = self.directory / "attendance.json"
        app = create_app(runtime_factory=lambda _: self.runtime, settings=self.settings)
        with patch("face_recognition.api.app.PROJECT_ROOT", self.directory), TestClient(app) as http:
            invalid = http.post("/video-recognition", json={"video_path": str(video), "mode": "attendance", "action": "checkin"})
            self.assertEqual(invalid.status_code, 422)
            unknown_field = http.post("/video-recognition", json={"video_path": str(video), "national_id": "removed"})
            self.assertEqual(unknown_field.status_code, 422)
            for action, hour in (("checkin", "09"), ("checkout", "17")):
                submitted = http.post("/video-recognition", json={
                    "video_path": str(video), "mode": "attendance", "action": action,
                    "start_time": f"2026-10-12T{hour}:00:00+03:00", "confidence": 0.9,
                    "attendance_file": str(state),
                })
                self.assertEqual(submitted.status_code, 202, submitted.text)
                job_id = submitted.json()["job_id"]
                deadline = time.monotonic() + 5
                while True:
                    job = http.get(f"/jobs/{job_id}").json()
                    if job["status"] in {"done", "failed"} or time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
                self.assertEqual(job["status"], "done", job)
                artifact = http.get(f'/jobs/{job_id}/artifacts/results')
                self.assertEqual(artifact.status_code, 200)
                self.assertEqual(artifact.json(), job['result']['records'])
                self.assertEqual(http.get(f'/jobs/{job_id}/artifacts/video').status_code, 404)
            record = job["result"]["records"][0]
            self.assertEqual(record["id"], person["id"])
            self.assertEqual(set(record), {"id", "name", "checkin", "checkout"})
            self.assertIsNotNone(record["checkin"])
            self.assertIsNotNone(record["checkout"])
            self.assertEqual(json.loads(state.read_text()), [record])


if __name__ == "__main__":
    unittest.main()
