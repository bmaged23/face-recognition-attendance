"""Native face detectors and embedding models, with no attendance or storage."""
import sys
import threading
import types

from face_recognition.config import ADAFACE_MODELS, PROJECT_ROOT, Settings
from face_recognition.core.models import FaceDetection, FaceSample


def _checkpoint_compatibility():
    """Older AdaFace checkpoints reference a Lightning callback when unpickled."""
    if "pytorch_lightning" in sys.modules:
        return
    lightning = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    module = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:
        pass

    module.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = module
    lightning.callbacks = callbacks
    sys.modules.setdefault("pytorch_lightning", lightning)
    sys.modules.setdefault("pytorch_lightning.callbacks", callbacks)
    sys.modules.setdefault("pytorch_lightning.callbacks.model_checkpoint", module)


class FaceBackend:
    """Detect and embed every face, using the same alignment for all workflows."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        if settings.family == "adaface":
            self._load_adaface()
        else:
            self._load_insightface()

    def _load_adaface(self):
        import torch
        torch.set_num_threads(self.settings.inference_threads)
        checkpoint_path = self.settings.checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint_path}")
        self.device = self.settings.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
        vendor_path = str(PROJECT_ROOT / "vendor" / "adaface")
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        import net
        from face_alignment.mtcnn import MTCNN
        self.model = net.build_model(ADAFACE_MODELS[self.settings.model_name][0])
        _checkpoint_compatibility()
        # Training classifier tensors are large and are never used for inference.
        # Load on CPU, then move only the selected backbone onto the GPU.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        # Training checkpoints also contain classifier weights; load only the backbone.
        if any(key.startswith("model.") for key in state):
            state = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.detector = MTCNN(device="cuda:0" if self.device == "cuda" else "cpu", crop_size=(112, 112))
        self.detector.thresholds[-1] = self.settings.mtcnn_final_threshold
        self.detector.factor = self.settings.mtcnn_scale_factor
        self.detector.min_face_size = self.settings.mtcnn_min_face_size

    def _load_insightface(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        available = "CUDAExecutionProvider" in ort.get_available_providers()
        if self.settings.device == "cuda" and not available:
            raise RuntimeError("CUDA was requested but ONNX Runtime cannot access a GPU")
        gpu = available and self.settings.device != "cpu"
        self.device = 'cuda' if gpu else 'cpu'
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.model = FaceAnalysis(
            name=self.settings.model_name, root=str(self.settings.insightface_root),
            providers=providers, allowed_modules=["detection", "recognition"],
        )
        size = self.settings.detector_size
        self.model.prepare(ctx_id=0 if gpu else -1, det_size=(size, size))

    def detect_and_embed(self, frame) -> list[FaceSample]:
        with self._lock:
            return self.embed_faces(frame, self.detect_faces(frame))

    def detect_faces(self, frame) -> list[FaceDetection]:
        with self._lock:
            if self.settings.family == "insightface":
                from insightface.app.common import Face
                boxes, landmarks = self.model.det_model.detect(frame, max_num=0, metric='default')
                return [FaceDetection(tuple(int(v) for v in box[:4]),
                        Face(bbox=box[:4], det_score=box[4],
                             kps=landmarks[index] if landmarks is not None else None))
                        for index, box in enumerate(boxes)]
            from PIL import Image
            boxes, faces = self.detector.align_multi(Image.fromarray(frame[:, :, ::-1]))
            return [FaceDetection(tuple(int(v) for v in box[:4]), face)
                    for box, face in zip(boxes, faces)]

    def embed_faces(self, frame, detections) -> list[FaceSample]:
        # Only new tracks or tracks due for recognition need embeddings.
        with self._lock:
            if not detections:
                return []
            if self.settings.family == 'insightface':
                samples = []
                for detection in detections:
                    face = detection.aligned
                    self.model.models['recognition'].get(frame, face)
                    samples.append(FaceSample(detection.bbox, face.normed_embedding))
                return samples
            import numpy as np
            import torch
            tensors = []
            for detection in detections:
                face = detection.aligned
                # AdaFace's vendor inference.py expects BGR channels after RGB alignment.
                array = (np.asarray(face, dtype=np.float32)[:, :, ::-1].copy() / 255.0 - 0.5) / 0.5
                tensors.append(torch.from_numpy(array).permute(2, 0, 1))
            batches = []
            with torch.inference_mode():
                batch_size = self.settings.embedding_batch_size
                for index in range(0, len(tensors), batch_size):
                    features, _ = self.model(torch.stack(tensors[index:index + batch_size]).to(self.device))
                    batches.append(features.cpu().numpy())
            embeddings = np.concatenate(batches)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            return [FaceSample(detection.bbox, embedding)
                    for detection, embedding in zip(detections, embeddings)]
