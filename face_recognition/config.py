"""Portable settings shared by enrollment, video, and camera recognition."""
from dataclasses import dataclass
import os
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAFACE_MODELS = {
    "ir_50_ms1mv2": ("ir_50", "adaface_ir50_ms1mv2.ckpt"),
    "ir_100_webface12m": ("ir_101", "adaface_ir100_webface12m.ckpt"),
    "ir_101_webface4m": ("ir_101", "adaface_ir101_webface4m.ckpt"),
    "ir_101_ms1mv3": ("ir_101", "adaface_ir101_ms1mv3.ckpt"),
    "ir_50_webface4m": ("ir_50", "adaface_ir50_webface4m.ckpt"),
    "ir_101_ms1mv2": ("ir_101", "adaface_ir101_ms1mv2.ckpt"),
}
INSIGHTFACE_MODELS = {"buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"}


@dataclass(frozen=True)
class Settings:
    family: str = "adaface"
    model_name: str = "ir_100_webface12m"
    collection: str = "people_adaface_ir_100_webface12m"
    qdrant_url: str = "http://localhost:5333"
    device: str = "auto"
    confidence: float = 0.4
    recognition_min_margin: float = 0.05  # Calibrate for your enrollment gallery; 0 disables.
    show_recognition_confidence: bool = True
    sample_every: float = 0.0  # 0: per-frame recognition without tracking; default refresh with tracking.
    use_iou_tracking: bool = True
    iou_threshold: float = 0.5
    recognition_refresh_seconds: float = 0.5
    detector_size: int = 640
    model_path: Path | None = None
    insightface_root: Path = Path.home() / ".insightface"
    max_image_bytes: int = 10 * 1024 * 1024

    def __post_init__(self):
        registry = ADAFACE_MODELS if self.family == "adaface" else INSIGHTFACE_MODELS
        if self.family not in {"adaface", "insightface"} or self.model_name not in registry:
            raise ValueError("Select a supported recognition family and model_name")
        if not self.collection.strip():
            raise ValueError("collection must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not math.isfinite(self.recognition_min_margin) or not 0 <= self.recognition_min_margin <= 1:
            raise ValueError('recognition_min_margin must be in [0, 1]')
        if not math.isfinite(self.sample_every) or self.sample_every < 0:
            raise ValueError("sample_every must be finite and non-negative; 0 means every frame")
        if not math.isfinite(self.iou_threshold) or not 0 < self.iou_threshold <= 1:
            raise ValueError('iou_threshold must be in (0, 1]')
        if not math.isfinite(self.recognition_refresh_seconds) or self.recognition_refresh_seconds <= 0:
            raise ValueError('recognition_refresh_seconds must be positive')
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.detector_size <= 0 or self.max_image_bytes <= 0:
            raise ValueError("detector_size and max_image_bytes must be positive")

    @property
    def model_signature(self) -> str:
        # Alignment and weights must agree between enrollment and recognition.
        import hashlib
        signature = f"v1:{self.family}:{self.model_name}:native-alignment:bgr"
        paths = []
        if self.family == "adaface":
            checkpoint = self.checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint}")
            paths = [checkpoint]
        else:
            paths = sorted((self.insightface_root / "models" / self.model_name).glob("*.onnx"))
            if not paths:
                raise FileNotFoundError("InsightFace model files were not found after model initialization")
        digest = hashlib.sha256()
        for path in paths:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        signature += f":{digest.hexdigest()}"
        return signature

    @property
    def checkpoint(self) -> Path:
        return self.model_path or PROJECT_ROOT / "vendor" / "adaface" / "pretrained" / ADAFACE_MODELS[self.model_name][1]

    @classmethod
    def from_env(cls):
        try:
            from dotenv import load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv(PROJECT_ROOT / ".env")
        family = os.getenv("RECOGNITION_FAMILY", "adaface")
        model = os.getenv("MODEL_NAME", "ir_100_webface12m" if family == "adaface" else "buffalo_l")
        path = os.getenv("ADAFACE_MODEL_PATH")
        show_confidence = os.getenv('SHOW_RECOGNITION_CONFIDENCE', str(cls.show_recognition_confidence)).lower()
        if show_confidence not in {'true', 'false'}:
            raise ValueError('SHOW_RECOGNITION_CONFIDENCE must be true or false')
        tracking = os.getenv('USE_IOU_TRACKING', str(cls.use_iou_tracking)).lower()
        if tracking not in {'true', 'false'}:
            raise ValueError('USE_IOU_TRACKING must be true or false')
        return cls(
            family=family, model_name=model,
            collection=os.getenv("QDRANT_COLLECTION", f"people_{family}_{model}"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:5333"),
            device=os.getenv("DEVICE", "auto"),
            confidence=float(os.getenv("RECOGNITION_CONFIDENCE", "0.4")),
            recognition_min_margin=float(os.getenv('RECOGNITION_MIN_MARGIN', str(cls.recognition_min_margin))),
            show_recognition_confidence=show_confidence == 'true',
            sample_every=float(os.getenv('SAMPLE_EVERY', str(cls.sample_every))),
            use_iou_tracking=tracking == 'true',
            iou_threshold=float(os.getenv('IOU_THRESHOLD', str(cls.iou_threshold))),
            recognition_refresh_seconds=float(os.getenv('RECOGNITION_REFRESH_SECONDS', str(cls.recognition_refresh_seconds))),
            detector_size=int(os.getenv("DETECTOR_SIZE", "640")),
            model_path=Path(path).expanduser() if path else None,
            insightface_root=Path(os.getenv("INSIGHTFACE_ROOT", str(Path.home() / ".insightface"))).expanduser(),
            max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
        )
