"""IoU reuse must update boxes without duplicating or indefinitely retaining labels."""
import unittest
from unittest.mock import Mock
from face_recognition.core.models import FaceDetection, Match, Person
from face_recognition.video.tracking import IoURecognitionEngine, intersection_over_union

ALICE = Person('alice', 'Alice')
BOB = Person('bob', 'Bob')


class TrackingTests(unittest.TestCase):
    def make_tracker(self):
        engine = Mock()
        engine.recognize_detections.side_effect = lambda frame, faces: [Match(d.bbox, ALICE, 0.822) for d in faces]
        return engine, IoURecognitionEngine(engine, threshold=0.5, refresh_seconds=1)

    def detect(self, engine, boxes):
        engine.backend.detect_faces.return_value = [FaceDetection(b, None) for b in boxes]

    def test_iou_geometry(self):
        self.assertEqual(intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)), 0)
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(intersection_over_union((0, 0, 0, 0), (0, 0, 0, 0)), 0)

    def test_high_overlap_reuses_exact_score_but_current_box(self):
        engine, tracker = self.make_tracker()
        self.detect(engine, [(0, 0, 100, 100)])
        tracker.recognize(None, 0)
        self.detect(engine, [(5, 0, 105, 100)])
        result = tracker.recognize(None, 0.1)[0]
        self.assertEqual(result, Match((5, 0, 105, 100), ALICE, 0.822))
        self.assertEqual(engine.recognize_detections.call_count, 1)
        self.assertEqual(engine.backend.detect_faces.call_count, 2)

    def test_low_overlap_recognizes_new_face_immediately(self):
        engine, tracker = self.make_tracker()
        self.detect(engine, [(0, 0, 100, 100)])
        tracker.recognize(None, 0)
        engine.recognize_detections.side_effect = lambda frame, faces: [Match(d.bbox, BOB, 0.9) for d in faces]
        self.detect(engine, [(200, 0, 300, 100)])
        self.assertEqual(tracker.recognize(None, 0.1)[0].person, BOB)
        self.assertEqual(engine.recognize_detections.call_count, 2)

    def test_refresh_can_remove_previous_identity(self):
        engine, tracker = self.make_tracker()
        self.detect(engine, [(0, 0, 100, 100)])
        tracker.recognize(None, 0)
        tracker.recognize(None, 0.9)
        engine.recognize_detections.side_effect = lambda frame, faces: [Match(d.bbox, None) for d in faces]
        self.assertIsNone(tracker.recognize(None, 1)[0].person)
        self.assertEqual(engine.recognize_detections.call_count, 2)

    def test_one_previous_track_is_not_reused_for_two_detections(self):
        engine, tracker = self.make_tracker()
        self.detect(engine, [(0, 0, 100, 100)])
        tracker.recognize(None, 0)
        self.detect(engine, [(0, 0, 100, 100), (5, 0, 105, 100)])
        tracker.recognize(None, 0.1)
        self.assertEqual(len(engine.recognize_detections.call_args.args[1]), 1)

    def test_disappearance_drops_track_and_return_requires_recognition(self):
        engine, tracker = self.make_tracker()
        self.detect(engine, [(0, 0, 100, 100)])
        tracker.recognize(None, 0)
        self.detect(engine, [])
        self.assertEqual(tracker.recognize(None, 0.1), [])
        self.detect(engine, [(0, 0, 100, 100)])
        tracker.recognize(None, 0.2)
        self.assertEqual(engine.recognize_detections.call_count, 2)

    def test_invalid_tracking_settings_are_rejected(self):
        for threshold in (0, -1, 2, float('nan')):
            with self.assertRaises(ValueError):
                IoURecognitionEngine(Mock(), threshold=threshold)
        for refresh in (0, -1, float('inf')):
            with self.assertRaises(ValueError):
                IoURecognitionEngine(Mock(), refresh_seconds=refresh)
