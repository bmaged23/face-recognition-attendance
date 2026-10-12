import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from qdrant_client import QdrantClient
from face_recognition.config import Settings
from face_recognition.storage.people import PeopleStore


class MatchMarginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        checkpoint = Path(self.temp.name) / 'model.ckpt'
        checkpoint.write_bytes(b'margin-test')
        self.client = QdrantClient(':memory:')
        self.addCleanup(self.client.close)
        self.store = PeopleStore(Settings(model_path=checkpoint), self.client)
        self.query = np.zeros(512)
        self.query[0] = 1

    def vector(self, score, axis):
        vector = np.zeros(512)
        vector[0] = score
        vector[axis] = math.sqrt(1 - score * score)
        return vector

    def test_close_candidates_rejected_and_clearer_margin_accepted(self):
        alice = self.store.enroll('Alice', self.vector(0.82, 1))
        self.store.enroll('Bob', self.vector(0.80, 2))
        self.assertIsNone(self.store.match(self.query, 0.4, 0.03))
        self.assertEqual(self.store.match(self.query, 0.4, 0.015)[0], alice)
        self.assertEqual(self.store.match(self.query, 0.4)[0], alice)

    def test_runner_up_below_absolute_threshold_is_still_compared(self):
        self.store.enroll('Alice', self.vector(0.41, 1))
        self.store.enroll('Bob', self.vector(0.39, 2))
        self.assertIsNone(self.store.match(self.query, 0.4, 0.05))

    def test_single_candidate_still_requires_absolute_threshold(self):
        self.store.enroll('Alice', self.vector(0.39, 1))
        self.assertIsNone(self.store.match(self.query, 0.4, 0.05))
        self.assertIsNotNone(self.store.match(self.query, 0.3, 0.05))

    def test_invalid_margin_is_rejected(self):
        for margin in (-0.1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.store.match(self.query, 0.4, margin)
