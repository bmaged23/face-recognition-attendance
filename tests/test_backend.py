"""Test actual tensor preprocessing separately from learned model accuracy."""
import importlib.util
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "PIL", "cv2", "numpy"))


@unittest.skipUnless(AVAILABLE, "Install the project inference dependencies")
class BackendTests(unittest.TestCase):
    def test_adaface_preprocessing_uses_bgr_and_unit_embeddings(self):
        import numpy as np
        import torch
        from PIL import Image
        from face_recognition.config import Settings
        from face_recognition.core.backend import FaceBackend

        class ChannelProbe(torch.nn.Module):
            def forward(self, images):
                features = torch.zeros((len(images), 512), dtype=images.dtype, device=images.device)
                features[:, :3] = images.mean(dim=(2, 3))
                return features, None

        backend = object.__new__(FaceBackend)
        backend.settings = Settings(device="cpu")
        backend.device = "cpu"
        backend._lock = threading.RLock()
        backend.model = ChannelProbe()
        backend.detector = Mock()
        backend.detector.align_multi.return_value = ([np.array([0, 0, 111, 111])], [Image.new("RGB", (112, 112), (255, 0, 0))])
        faces = backend.detect_and_embed(np.zeros((112, 112, 3), dtype=np.uint8))
        np.testing.assert_allclose(faces[0].embedding[:3], np.array([-1, -1, 1]) / np.sqrt(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(faces[0].embedding)), 1.0, places=6)

    def test_native_mtcnn_loads_weights_without_changing_cwd(self):
        import os
        from PIL import Image
        from face_recognition.config import PROJECT_ROOT
        directory = str(PROJECT_ROOT / "vendor" / "adaface")
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from face_alignment.mtcnn import MTCNN
        original = os.getcwd()
        detector = MTCNN(device="cpu", crop_size=(112, 112))
        self.assertEqual(os.getcwd(), original)
        _, faces = detector.align_multi(Image.new("RGB", (32, 32), (0, 0, 0)))
        self.assertEqual(faces, [])


if __name__ == "__main__":
    unittest.main()
