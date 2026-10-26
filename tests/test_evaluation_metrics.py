"""Report errors explicitly rather than masking detector misses as rejections."""
import unittest
from face_recognition.config import Settings
from scripts.evaluation.evaluate_recognition import summarize


class EvaluationMetricTests(unittest.TestCase):
    def test_missed_detection_is_in_end_to_end_denominator(self):
        rows = []
        for identity, known, detected, prediction in [
            ('a', True, True, 'a'), ('b', True, False, None),
            ('c', False, True, 'a'), ('d', False, False, None),
        ]:
            rows.append(dict(identity=identity, known=known, detected=detected, prediction=prediction,
                             total_seconds=1., detection_seconds=.5, embedding_seconds=.4, search_seconds=.1))
        report = summarize(rows)
        self.assertEqual(report['known_end_to_end_correct_rate'], .5)
        self.assertEqual(report['known_correct_rate_given_detection'], 1.)
        self.assertEqual(report['unknown_false_accept_rate_given_detection'], 1.)
        self.assertEqual(report['known_rejected_after_detection'], 0)
        self.assertEqual(report['unknown_false_accepts'], 1)
        self.assertEqual(report['wrong_known_accepted'], 0)

    def test_invalid_detector_and_queue_configuration_fails_early(self):
        for options in [dict(mtcnn_final_threshold=float('nan')), dict(mtcnn_scale_factor=1.),
                        dict(embedding_batch_size=0), dict(inference_threads=0),
                        dict(mtcnn_min_face_size=1), dict(job_capacity=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                Settings(**options)
