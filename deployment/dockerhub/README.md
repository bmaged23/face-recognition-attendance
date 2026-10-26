# Face Recognition & Attendance — GPU AdaFace API

[GitHub source, looping demo and measured results](https://github.com/bmaged23/face-recognition-attendance)

Enroll from one photo, recognize people in images/videos and optionally record
explicit check-in/check-out. AdaFace creates 512-dimensional embeddings; MTCNN
handles detection/alignment; an embedded Qdrant gallery persists locally.

## Pull and run

Linux x86-64 containers with NVIDIA GPU support are required. The native tested
GPU is a GTX 1660 Ti with 6 GiB VRAM. This image includes the inference checkpoint,
detector weights and pinned environment. No Git clone, first-run model download,
cloud model or API key is needed. Allow several GB for the CUDA/PyTorch image.

```sh
docker pull bmaged23/face-recognition-attendance:gpu
docker run --rm --gpus all --name face-attendance \
  -p 127.0.0.1:7323:7323 \
  -v face-attendance-data:/app/data \
  bmaged23/face-recognition-attendance:gpu
```

Open http://localhost:7323/docs. `GET /ready` initializes the real model/gallery;
`GET /health` is liveness only. The named volume retains enrolled embeddings and
outputs. The application runs as UID/GID 1000. Stop with Ctrl+C. Upload your own
photo with the interactive API or curl:

```sh
curl --fail -F 'name=Alice Smith' -F image=@alice.jpg http://localhost:7323/enroll
curl --fail -F image=@probe.jpg http://localhost:7323/recognize/image
```

For videos, mount a local input folder read-only at `/app/input` and submit that
container path to `POST /video-recognition`. Poll `GET /jobs/{id}`; download
completed JSON or MP4 using `/jobs/{id}/artifacts/results` and
`/jobs/{id}/artifacts/video`. MP4 responses support HTTP byte ranges for seeking.
The API is local/trusted and has no authentication; keep the port on localhost.

[Full Docker instructions](https://github.com/bmaged23/face-recognition-attendance/blob/main/docs/deployment/DOCKER.md) ·
[API guide](https://github.com/bmaged23/face-recognition-attendance/blob/main/docs/api/README.md)

## Evaluation and delivery

The candidate image runs the shipped regression suite, all 228 frozen real GPU
recognition probes, real HTTP enrollment/known/unknown recognition, attendance,
annotation/download checks and persistent recognition after restart. A trusted
local GPU runner supplies the private research fixtures. Only passing candidates
are published; the public tag is then pulled anonymously and started/tested again.
[CI/CD flow and workflow](https://github.com/bmaged23/face-recognition-attendance/blob/main/docs/deployment/README.md)

Native CPU/GPU runs had identical outcomes: 113/152 known probes recognized,
all 113 detected known probes correct, and 0/56 detected unknown probes falsely
accepted. Detection misses are the main weakness; this is not 100% end-to-end
accuracy. Median probe processing was 1,287.20 ms CPU versus 110.95 ms CUDA
(11.60x in this matched study, excluding startup/enrollment/video encoding).

The compact checkpoint retains all 917 inference tensors exactly, shrinking
1.53 GB to 261 MB. In native runs it preserved all 228 outcomes/scores and reduced
sampled peak process RAM from 2,135.55 to 1,034.34 MiB. These are native measurements;
see separate exact-container evidence and small-set limitations in the
[evaluation report](https://github.com/bmaged23/face-recognition-attendance/blob/main/docs/evaluation/README.md).

Raw research images, personal galleries/attendance, credentials and source demo
footage are excluded from the runtime image. AdaFace/MTCNN notices remain under
`vendor/`; upstream model/data terms still apply. The research dataset has
noncommercial terms and is acquired separately. This is a local prototype without
liveness detection or a biometric security/fairness certification. Similarity is
not a calibrated probability.
