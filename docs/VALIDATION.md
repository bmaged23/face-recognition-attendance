# Validation

The tested environment uses Python 3.12.15, PyTorch 2.5.1+cu124, and the exact
installed versions pinned in `requirements.txt`. Real GPU execution used an
NVIDIA GTX 1660 Ti (6 GiB), driver 560.76, and the pretrained AdaFace WebFace12M
checkpoint. Model SHA-256:
`0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0`.

## Software checks

- All 70 application regression tests passed after the presentation cleanup,
  including HTTP range handling and private-file exclusion for the demo server.
- `pip check` reported no broken requirements.
- Tests use actual in-memory Qdrant searches, Pillow/OpenCV media operations,
  and FastAPI multipart/background-job flows, with controlled model outputs.
- Matching tests verify the original, unrounded threshold and top-2 ambiguity
  margin, including a runner-up below the absolute threshold.
- Tracking tests verify current-frame boxes, one-to-one association, scheduled
  recognition, missing-track removal, and exact name/score reuse.
- Annotation tests exercise percentages and rolling attendance panels.
- Both approved videos decode fully and use H.264/yuv420p with fast-start MP4.
- A headless Chromium playback check verified the final order: check-in seeks
  to 3 seconds, its real end event advances to check-out at 28 seconds, and
  check-out's real end event wraps to check-in at 3 seconds. Next/restart,
  mobile layout, and JavaScript/private-file checks passed. The local server
  supports HTTP byte ranges so seeking does not reset playback to zero.
- After reducing the vendor checkout to inference code, the pretrained GPU
  backend generated three real detected/aligned face examples and finite,
  unit-normalized 512-dimensional embeddings for the README.
- All 57 installed Python distributions matched the exact requirements file;
  an offline `pip install --dry-run --no-index` required no package changes.

These are behavior checks, not a recognition accuracy benchmark. The optional
InsightFace weights, actual camera/RTSP access, and end-to-end Docker deployment
have not been tested. The Starlette test client emits an upstream deprecation
warning for its httpx integration; endpoint tests pass.

## Real labeled recognition evaluation

The gallery has 29 people, each represented by exactly one photo. Original
references are 96×96 grayscale crops; improved references are independent color
stills selected by aligned-image sharpness, without looking at probe scores.
EXIF orientation is applied. All inference uses the same model/preprocessing.

| Detected labeled samples | Grayscale, score ≥0.4 | Color, score ≥0.4 and gap ≥0.05 |
|---|---|---|
| S2 camera 2, development | 195/201 correct accepted | 200/201 correct accepted |
| S2 camera 1, held-out view | 129/135 correct accepted | 134/135 correct accepted |

Both configurations accepted zero wrong identities in these samples. Rejected
known faces account for the remaining detected probes. Camera 1 had 90 samples
without a usable detector match; camera 2 had 31. Some recorded misses represent
an identity with no labeled frames, so these counts must not be presented as a
universal detector-recall calculation. Recognition percentages are conditional
on detected labeled faces. Camera 1 is another view of the same recording as
camera 2, not a fully independent recording.

The threshold search found 0.32 on this narrow development set. That result was
not adopted as a general threshold: the application retains 0.4 and the 0.05
ambiguity guard. Raising the threshold to 0.5 accepted fewer correct faces.
The leave-one-identity-out false-accept count is a proxy, not an actual
unseen-person evaluation. Complete results: [calibration.json](evaluation/calibration.json).

## Crowded-video policy comparison

The S5 videos contain 842 entry frames and 1,513 exit frames, with 2,398 and
2,319 detected faces respectively. Real GPU detections/embeddings were cached
for every frame to compare policies on identical observations.

| Color gallery + ambiguity guard | Entry | Exit |
|---|---:|---:|
| Disagreements with every-frame matching at 1.0 s refresh | 397 | 435 |
| Disagreements at 0.5 s refresh | 263 | 199 |
| Embedding requests at 0.5 s refresh | 363 | 318 |
| Embedding requests with every-frame matching | 2,398 | 2,319 |

The default refresh changed to 0.5 seconds. These numbers measure consistency
with fresh recognition and reduced embedding workload; they do not measure
identity accuracy or wall-clock speed. S5 has no supplied identity ground truth.
The approved annotated clips are qualitative examples and must not be described
as a measured 99.3% crowded-video benchmark. Report:
[crowded_comparison.json](evaluation/crowded_comparison.json).

## Repeat checks

```sh
python -m pip check
python -m unittest discover -s tests -v
python scripts/download_model.py --verify
python scripts/check_gpu.py
```

The GPU diagnostic builds a random-weight backbone to check the architecture,
tensor shape, finite values, and detector kernels. Real-model behavior is
established by the separate pretrained checkpoint/video tests.
