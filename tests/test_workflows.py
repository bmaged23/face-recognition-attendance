"""Behavior tests with fake inference/storage; no model downloads or GPU required."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import uuid

from face_recognition.config import Settings
from face_recognition.core.engine import RecognitionEngine
from face_recognition.core.images import decode_photo
from face_recognition.core.models import FaceSample, Match, Person
from face_recognition.enrollment.service import EnrollmentService
from face_recognition.storage.attendance import AttendanceStore, parse_timestamp
from face_recognition.storage.people import PeopleStore, validate_name
from face_recognition.video.pipeline import StreamPipeline
from face_recognition.video.results import SessionResults

ALICE = Person(str(uuid.uuid4()), "Alice Smith")
BOB = Person(str(uuid.uuid4()), "Alice Smith")  # Same name, different identity.
BOX = (1, 2, 30, 40)


class MatchingTests(unittest.TestCase):
    def test_confidence_boundary_uses_unrounded_value(self):
        backend = Mock()
        backend.detect_and_embed.return_value = [FaceSample(BOX, [1]), FaceSample(BOX, [2])]
        people = Mock()
        people.match.side_effect = [(ALICE, 0.599999), (BOB, 0.6)]
        matches = RecognitionEngine(backend, people, 0.6).recognize(object())
        self.assertIsNone(matches[0].person)
        self.assertEqual(matches[1].person, BOB)

    def test_unknown_does_not_create_a_person(self):
        backend = Mock()
        backend.detect_and_embed.return_value = [FaceSample(BOX, [1])]
        people = Mock()
        people.match.return_value = None
        self.assertIsNone(RecognitionEngine(backend, people, 0.4).recognize(object())[0].person)

    def test_zero_threshold_is_allowed(self):
        backend = Mock()
        backend.detect_and_embed.return_value = [FaceSample(BOX, [1])]
        people = Mock()
        people.match.return_value = (ALICE, 0.0)
        self.assertEqual(RecognitionEngine(backend, people, 0).recognize(None)[0].person, ALICE)

    def test_invalid_thresholds_rejected(self):
        for confidence in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                RecognitionEngine(Mock(), Mock(), confidence)

    def test_nonfinite_match_cannot_confirm_identity(self):
        backend = Mock()
        backend.detect_and_embed.return_value = [FaceSample(BOX, [1])]
        store = Mock()
        store.match.return_value = (ALICE, float("nan"))
        self.assertIsNone(RecognitionEngine(backend, store, 0).recognize(None)[0].person)


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.store = Mock()
        self.store.enroll.return_value = ALICE
        self.service = EnrollmentService(self.backend, self.store, 100)

    def test_single_photo_creates_person(self):
        self.backend.detect_and_embed.return_value = [FaceSample(BOX, [0.1])]
        result = self.service.enroll_frame("Alice Smith", object())
        self.assertEqual(result, ALICE.to_dict())
        self.store.enroll.assert_called_once_with("Alice Smith", [0.1], None)

    def test_no_face_and_group_photo_cannot_enroll(self):
        for count in (0, 2):
            self.backend.detect_and_embed.return_value = [FaceSample(BOX, [1])] * count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "exactly one face"):
                self.service.enroll_frame("Alice", object())
        self.store.enroll.assert_not_called()

    def test_english_names_are_required(self):
        for name in ("", "   ", "أحمد", "Alice\nSmith"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.service.enroll_frame(name, object())
        self.backend.detect_and_embed.assert_not_called()

    def test_image_size_checked_before_inference_dependencies(self):
        for image in (b"", b"x" * 101):
            with self.subTest(size=len(image)), self.assertRaises(ValueError):
                self.service.enroll("Alice", image)
        self.backend.detect_and_embed.assert_not_called()

    def test_update_uuid_is_forwarded(self):
        self.backend.detect_and_embed.return_value = [FaceSample(BOX, [0.1])]
        self.service.enroll_frame("Alice Smith", object(), ALICE.id)
        self.store.enroll.assert_called_once_with("Alice Smith", [0.1], ALICE.id)

    def test_invalid_uuid_rejected_before_decoding(self):
        with self.assertRaisesRegex(ValueError, "UUID"):
            self.service.enroll("Alice", b"photo", "not-a-uuid")


class ResultTests(unittest.TestCase):
    def test_recognition_has_no_attendance_fields(self):
        results = SessionResults()
        results.observe([Match(BOX, ALICE, 0.7), Match(BOX, None)], "00:00:02.000")
        results.observe([Match(BOX, ALICE, 0.8), Match(BOX, BOB, 0.8)], "00:00:03.000")
        self.assertEqual(len(results.records), 2)
        self.assertEqual(results.records[0], {**ALICE.to_dict(), "time": "00:00:02.000"})
        self.assertNotEqual(results.records[0]["id"], results.records[1]["id"])

    def test_live_recognition_uses_absolute_timezone(self):
        stamp = parse_timestamp("2026-10-12T09:00:00+03:00")
        result = SessionResults().observe([Match(BOX, ALICE)], stamp)
        self.assertEqual(result[0]["time"], "2026-10-12T09:00:00.000+03:00")

    def test_action_cannot_be_used_in_recognition_mode(self):
        with self.assertRaises(ValueError):
            SessionResults(action="checkin")

    def test_repeated_sightings_call_attendance_once(self):
        store = Mock()
        store.record.return_value = {**ALICE.to_dict(), "checkin": "now", "checkout": None}
        results = SessionResults(store, "checkin")
        stamp = parse_timestamp("2026-10-12T09:00:00Z")
        for _ in range(5):
            results.observe([Match(BOX, ALICE)], stamp)
        store.record.assert_called_once_with(ALICE, "checkin", stamp)


class AttendanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "attendance.json"
        self.checkin = parse_timestamp("2026-10-12T09:00:00+03:00")
        self.checkout = parse_timestamp("2026-10-12T17:00:00+03:00")

    def test_checkin_checkout_persist_across_instances(self):
        first = AttendanceStore(self.path).record(ALICE, "checkin", self.checkin)
        self.assertIsNone(first["checkout"])
        final = AttendanceStore(self.path).record(ALICE, "checkout", self.checkout)
        self.assertEqual(set(final), {"id", "name", "checkin", "checkout"})
        self.assertEqual(final["checkin"], "2026-10-12T09:00:00.000+03:00")
        self.assertEqual(final["checkout"], "2026-10-12T17:00:00.000+03:00")
        self.assertEqual(json.loads(self.path.read_text()), [final])

    def test_checkout_without_checkin_is_allowed(self):
        row = AttendanceStore(self.path).record(ALICE, "checkout", self.checkout)
        self.assertIsNone(row["checkin"])

    def test_reruns_preserve_first_qualifying_time(self):
        store = AttendanceStore(self.path)
        store.record(ALICE, "checkin", self.checkout)
        store.record(ALICE, "checkin", self.checkin)
        row = store.record(ALICE, "checkin", self.checkout)
        self.assertEqual(row["checkin"], self.checkin.isoformat(timespec="milliseconds"))
        self.assertEqual(len(json.loads(self.path.read_text())), 1)

    def test_same_name_people_remain_distinct(self):
        store = AttendanceStore(self.path)
        store.record(ALICE, "checkin", self.checkin)
        store.record(BOB, "checkin", self.checkin)
        self.assertEqual(len(json.loads(self.path.read_text())), 2)

    def test_checkout_before_checkin_does_not_corrupt_file(self):
        store = AttendanceStore(self.path)
        store.record(ALICE, "checkin", self.checkout)
        original = self.path.read_text()
        with self.assertRaisesRegex(ValueError, "precedes"):
            store.record(ALICE, "checkout", self.checkin)
        self.assertEqual(self.path.read_text(), original)

    def test_naive_and_invalid_timestamps_rejected(self):
        for stamp in ("2026-10-12T09:00:00", "bad-time", None):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                parse_timestamp(stamp)
        with self.assertRaises(ValueError):
            AttendanceStore(self.path).record(ALICE, "checkin", datetime(2026, 10, 12))

    def test_old_schema_rejected_without_overwriting(self):
        self.path.write_text('[{"name":"Alice","national_id":"old-id"}]')
        original = self.path.read_text()
        with self.assertRaises(ValueError):
            AttendanceStore(self.path).record(ALICE, "checkin", self.checkin)
        self.assertEqual(self.path.read_text(), original)

    def test_concurrent_updates_do_not_lose_people(self):
        people = [Person(str(uuid.uuid4()), f"Person {index}") for index in range(20)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda person: AttendanceStore(self.path).record(person, "checkin", self.checkin), people))
        rows = json.loads(self.path.read_text())
        self.assertEqual({row["id"] for row in rows}, {person.id for person in people})


class FakeCapture:
    def __init__(self, frames, fps=1.0):
        self.frames = iter(frames)
        self.fps = fps
        self.position = -1
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return self.fps if key == 1 else max(0, self.position) / self.fps * 1000 if self.fps else 0

    def read(self):
        try:
            frame = next(self.frames)
        except StopIteration:
            return False, None
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "source.mp4"
        self.source.touch()
        self.output = self.directory / "result.json"
        self.engine = Mock()

    def fake_cv(self, capture):
        cv = types.ModuleType("cv2")
        cv.CAP_PROP_FPS = 1
        cv.CAP_PROP_POS_MSEC = 2
        cv.VideoCapture = lambda _: capture
        return patch.dict(sys.modules, {"cv2": cv})

    def test_video_recognition_with_unknown_then_confirmed_face(self):
        capture = FakeCapture([0, 1, 2])
        self.engine.recognize.side_effect = [[Match(BOX, None)], [Match(BOX, ALICE, 0.7)], [Match(BOX, ALICE, 0.8)]]
        with self.fake_cv(capture):
            rows = StreamPipeline(self.engine).run_video(self.source, output=self.output)
        self.assertEqual(rows, [{**ALICE.to_dict(), "time": "00:00:01.000"}])
        self.assertEqual(json.loads(self.output.read_text()), rows)
        self.assertTrue(capture.released)
        self.assertFalse((self.directory / "attendance.json").exists())

    def test_video_attendance_combines_separate_runs(self):
        state = self.directory / "attendance.json"
        self.engine.recognize.return_value = [Match(BOX, ALICE, 0.7)]
        for action, start in (("checkin", "2026-10-12T09:00:00+03:00"), ("checkout", "2026-10-12T17:00:00+03:00")):
            with self.fake_cv(FakeCapture([0, 1])):
                rows = StreamPipeline(self.engine).run_video(self.source, output=self.output, action=action,
                            attendance_file=state, start_time=start)
        self.assertEqual(rows[0]["checkin"], "2026-10-12T09:00:00.000+03:00")
        self.assertEqual(rows[0]["checkout"], "2026-10-12T17:00:00.000+03:00")

    def test_video_attendance_requires_recording_start(self):
        with self.assertRaisesRegex(ValueError, "start_time"):
            StreamPipeline(self.engine).run_video(self.source, output=self.output, action="checkin")
        self.engine.recognize.assert_not_called()

    def test_sampling_controls_inference_frequency(self):
        capture = FakeCapture([0, 1, 2, 3], fps=2)
        self.engine.recognize.return_value = []
        with self.fake_cv(capture):
            StreamPipeline(self.engine).run_video(self.source, output=self.output, sample_every=1)
        self.assertEqual(self.engine.recognize.call_count, 2)

    def test_default_recognizes_every_frame_at_video_frame_rate(self):
        capture = FakeCapture([0, 1, 2, 3], fps=30)
        self.engine.recognize.return_value = []
        with self.fake_cv(capture):
            StreamPipeline(self.engine).run_video(self.source, output=self.output)
        self.assertEqual(self.engine.recognize.call_count, 4)

    def test_zero_overrides_configured_sampling(self):
        capture = FakeCapture([0, 1, 2, 3], fps=30)
        self.engine.recognize.return_value = []
        with self.fake_cv(capture):
            StreamPipeline(self.engine, settings=Settings(sample_every=1)).run_video(
                self.source, output=self.output, sample_every=0)
        self.assertEqual(self.engine.recognize.call_count, 4)

    def test_tracking_detects_every_frame_and_only_embeds_new_or_due_faces(self):
        from face_recognition.core.models import FaceDetection
        backend = Mock(spec=['detect_faces', 'embed_faces'])
        backend.detect_faces.side_effect = lambda frame: [FaceDetection((frame, 0, frame + 100, 100), None)]
        backend.embed_faces.side_effect = lambda frame, faces: [FaceSample(d.bbox, [1.0]) for d in faces]
        store = Mock()
        store.match.return_value = (ALICE, 0.822)
        engine = RecognitionEngine(backend, store, 0.4)
        with self.fake_cv(FakeCapture([0, 1, 2, 3], fps=30)):
            rows = StreamPipeline(engine, Settings(use_iou_tracking=True)).run_video(
                self.source, output=self.output, sample_every=1)
        self.assertEqual(backend.detect_faces.call_count, 4)
        self.assertEqual(backend.embed_faces.call_count, 1)
        self.assertEqual(store.match.call_count, 1)
        self.assertEqual(rows[0]['id'], ALICE.id)

    def test_annotation_percentage_can_be_enabled_and_disabled(self):
        import numpy as np
        from face_recognition.video.annotation import annotate
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        matches = [Match(BOX, ALICE, 0.8216), Match(BOX, None)]
        for enabled, expected in [(True, 'Alice Smith (82.2%)'), (False, 'Alice Smith')]:
            with patch('cv2.putText') as put_text:
                annotate(frame.copy(), matches, show_confidence=enabled)
            self.assertEqual(put_text.call_args_list[0].args[1], expected)
            self.assertEqual(put_text.call_args_list[1].args[1], 'Unknown')

    def test_live_camera_interrupt_preserves_results_and_releases_capture(self):
        capture = FakeCapture([0, 1])
        original_read = capture.read
        def read():
            if capture.position == 0:
                raise KeyboardInterrupt
            return original_read()
        capture.read = read
        self.engine.recognize.return_value = [Match(BOX, ALICE, 0.7)]
        with self.fake_cv(capture):
            rows = StreamPipeline(self.engine).run_camera(0, output=self.output)
        self.assertEqual(set(rows[0]), {"id", "name", "time"})
        parse_timestamp(rows[0]["time"])
        self.assertEqual(json.loads(self.output.read_text()), rows)
        self.assertTrue(capture.released)

    def test_camera_disconnect_reports_error_and_keeps_results(self):
        capture = FakeCapture([0])
        self.engine.recognize.return_value = [Match(BOX, ALICE, 0.7)]
        with self.fake_cv(capture), self.assertRaises(RuntimeError):
            StreamPipeline(self.engine).run_camera(0, output=self.output)
        self.assertEqual(json.loads(self.output.read_text())[0]["id"], ALICE.id)
        self.assertTrue(capture.released)

    def test_invalid_fps_does_not_replace_existing_results(self):
        self.output.write_text("previous results")
        capture = FakeCapture([], fps=0)
        with self.fake_cv(capture), self.assertRaisesRegex(ValueError, "frame rate"):
            StreamPipeline(self.engine).run_video(self.source, output=self.output)
        self.assertEqual(self.output.read_text(), "previous results")
        self.assertTrue(capture.released)

    def test_output_cannot_replace_input_or_attendance_state(self):
        with self.assertRaisesRegex(ValueError, "input video"):
            StreamPipeline(self.engine).run_video(self.source, output=self.source)
        with self.assertRaisesRegex(ValueError, "different paths"):
            StreamPipeline(self.engine).run_video(self.source, output=self.output, action="checkin",
                          attendance_file=self.output, start_time="2026-10-12T09:00:00Z")


class FakeQdrant:
    def __init__(self):
        self.points = {}
        self.exists = False
        self.score = 0.8

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, **kwargs):
        self.exists = True

    def get_collection(self, name):
        return types.SimpleNamespace(config=types.SimpleNamespace(params=types.SimpleNamespace(
            vectors=types.SimpleNamespace(size=512, distance="Cosine"))))

    def scroll(self, *args, **kwargs):
        return list(self.points.values())[:1], None

    def upsert(self, collection, *, wait, points):
        for point in points:
            self.points[point.id] = point

    def retrieve(self, collection, *, ids, with_payload):
        return [self.points[key] for key in ids if key in self.points]

    def query_points(self, **kwargs):
        self.last_query = kwargs
        points = list(self.points.values())[:1] if self.score >= kwargs["score_threshold"] else []
        return types.SimpleNamespace(points=[types.SimpleNamespace(id=point.id, payload=point.payload, score=self.score) for point in points])


class PeopleStorageTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType("qdrant_client")
        module.QdrantClient = Mock()
        models = types.ModuleType("qdrant_client.models")
        models.Distance = types.SimpleNamespace(COSINE="Cosine")
        models.VectorParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
        models.PointStruct = lambda **kwargs: types.SimpleNamespace(**kwargs)
        self.patch = patch.dict(sys.modules, {"qdrant_client": module, "qdrant_client.models": models})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.settings = types.SimpleNamespace(collection="people", qdrant_url="http://test", model_signature="model-A")
        self.client = FakeQdrant()
        self.store = PeopleStore(self.settings, self.client)

    def test_enrollment_uuid_contains_no_legacy_identity_fields(self):
        person = self.store.enroll("Alice", [0.1] * 512)
        self.assertEqual(str(uuid.UUID(person.id)), person.id)
        self.assertEqual(set(self.client.points[person.id].payload), {"person_id", "name", "model_signature"})

    def test_update_keeps_uuid_and_one_photo(self):
        person = self.store.enroll("Alice", [0.1] * 512)
        updated = self.store.enroll("Alice Smith", [0.2] * 512, person.id)
        self.assertEqual(person.id, updated.id)
        self.assertEqual(len(self.client.points), 1)
        self.assertEqual(updated.name, "Alice Smith")

    def test_nonexistent_update_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.store.enroll("Alice", [0.1] * 512, str(uuid.uuid4()))
        self.assertFalse(self.client.points)

    def test_match_returns_person_and_full_precision_confidence(self):
        person = self.store.enroll("Alice", [0.1] * 512)
        self.client.score = 0.600001
        self.assertEqual(self.store.match([0.2] * 512, 0.6), (person, 0.600001))
        self.assertEqual(self.client.last_query["with_payload"], ["person_id", "name", "model_signature"])
        self.assertIsNone(self.store.match([0.2] * 512, 0.7))

    def test_incompatible_model_or_old_schema_is_rejected(self):
        self.store.enroll("Alice", [0.1] * 512)
        other = types.SimpleNamespace(collection="people", qdrant_url="test", model_signature="model-B")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PeopleStore(other, self.client)
        point = next(iter(self.client.points.values()))
        point.payload = {"label": "Alice", "national_id": "old-id"}
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PeopleStore(self.settings, self.client)


class EntryPointTests(unittest.TestCase):
    def test_all_three_help_commands_work_without_inference_dependencies(self):
        root = Path(__file__).resolve().parent.parent
        for program in ("video_recognition.py", "live_camera_recognition.py", "enroll.py"):
            with self.subTest(program=program):
                result = subprocess.run([sys.executable, str(root / program), "--help"], capture_output=True, text=True, cwd=root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("national_id", result.stdout)

    def test_cli_requires_explicit_attendance_action_and_start_time(self):
        root = Path(__file__).resolve().parent.parent
        invalid = [
            ["--video", "video.mp4", "--mode", "attendance"],
            ["--video", "video.mp4", "--mode", "attendance", "--action", "checkin"],
            ["--video", "video.mp4", "--action", "checkout"],
        ]
        for options in invalid:
            with self.subTest(options=options):
                result = subprocess.run([sys.executable, str(root / "video_recognition.py"), *options], capture_output=True, text=True, cwd=root)
                self.assertEqual(result.returncode, 2)

    def test_model_fingerprint_changes_with_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.ckpt"
            checkpoint.write_bytes(b"weights-A")
            settings = Settings(model_path=checkpoint)
            signature = settings.model_signature
            checkpoint.write_bytes(b"weights-B")
            self.assertNotEqual(signature, settings.model_signature)


if __name__ == "__main__":
    unittest.main()
