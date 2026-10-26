"""A failed, changed or incomplete recognition run must block publication."""
from copy import deepcopy
import unittest
from scripts.release.assert_quality import check


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(fixture_sha256='fixtures', model_signature='model', required_case_count=1,
                              required_gallery_count=1, min_correct_known=1, min_detected_unknown=0,
                              max_wrong_known=0, max_unknown_false_accepts=0,
                              outcomes=[dict(case_id='one', detected=True, prediction='alice')])
        self.report = dict(fixture_sha256='fixtures', model_signature='model', device='cuda',
                           samples=[dict(case_id='one', detected=True, prediction='alice')],
                           gallery=[dict(enrolled=True)],
                           overall=dict(correct_known_accepted=1, detected_unknown=0,
                                        wrong_known_accepted=0, unknown_false_accepts=0))

    def test_valid_exact_image_result_passes(self):
        self.assertEqual(check(self.report, self.reference), [])

    def test_missing_wrong_device_changed_outcome_and_false_accept_fail(self):
        variants = []
        for key, value in [('fixture_sha256', 'changed'), ('model_signature', 'changed'),
                           ('device', 'cpu'), ('samples', []), ('gallery', [])]:
            report = deepcopy(self.report); report[key] = value; variants.append(report)
        report = deepcopy(self.report); report['samples'][0]['prediction'] = 'bob'; variants.append(report)
        report = deepcopy(self.report); report['overall']['unknown_false_accepts'] = 1; variants.append(report)
        for report in variants:
            with self.subTest(report=report):
                self.assertTrue(check(report, self.reference))
