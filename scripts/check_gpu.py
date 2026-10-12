"""Verify CUDA kernels and model architecture; does not test recognition accuracy."""
import json
import sys
from pathlib import Path
import torch
from PIL import Image

print("PyTorch:", torch.__version__, flush=True)
print("CUDA runtime:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "adaface"))
import net
from face_alignment.mtcnn import MTCNN
model = net.build_model("ir_101").cuda().eval()
with torch.inference_mode():
    embedding, norm = model(torch.randn(1, 3, 112, 112, device="cuda"))
    assert tuple(embedding.shape) == (1, 512)
    assert torch.isfinite(embedding).all().item()
detector = MTCNN(device="cuda:0", crop_size=(112, 112))
_, faces = detector.align_multi(Image.new("RGB", (32, 32), (0, 0, 0)))
assert len(faces) == 0
torch.cuda.synchronize()
print(json.dumps({"gpu_kernel_check": "passed", "adaface_backbone": "ir_101", "embedding_shape": list(embedding.shape), "mtcnn": "passed", "note": "Random backbone weights; recognition accuracy not tested"}), flush=True)
