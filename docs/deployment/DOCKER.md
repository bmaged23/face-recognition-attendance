# Docker GPU application

The public GPU image passed exact-image evaluation and anonymous pull/startup.
[Verified measurements and digest](../evaluation/README.md#verified-container-release).
The maintained image recipe is
`deployment/docker/Dockerfile`; no root Dockerfile is required.

The complete GPU image includes Python, exact dependencies, the AdaFace inference
backbone, MTCNN detector weights, CLI programs, and the HTTP API. It uses a local
embedded Qdrant gallery by default. No model download, Git clone, cloud account or
API key is needed after pulling. Data/models for research evaluation and the
prerecorded source demo are not bundled in the runtime image.

## Pull and start

Requires Linux x86-64 containers and an NVIDIA GPU exposed to Docker. Native
CUDA was tested on GTX 1660 Ti 6 GiB. Native Windows/macOS/ARM and CPU-only use of
the public GPU image are not certified. Docker startup/evaluation evidence is
separate from native results. The registry download is approximately 3.64 GB and Docker reports 10.26 GB
locally. Allow 20 GiB free disk headroom for downloading, unpacking and outputs.

```sh
docker pull bmaged23/face-recognition-attendance:gpu
docker run --rm --gpus all --name face-attendance \
  -p 127.0.0.1:7323:7323 \
  -v face-attendance-data:/app/data \
  bmaged23/face-recognition-attendance:gpu
```

Open http://localhost:7323/docs. `GET /ready` initializes the actual model and
checks the embedded gallery; `GET /health` is liveness only. Swagger supports
photo enrollment and recognition. It is an API interface, not a camera web UI.
The image runs as UID/GID 1000. The named volume persists enrollment and outputs;
stop with Ctrl+C and repeat the command to reuse it. Do not remove the volume
unless you intend to delete its biometric gallery and attendance records.

## Photo requests

In another terminal:

```sh
curl --fail http://localhost:7323/ready
curl --fail -F 'name=Alice Smith' -F image=@alice.jpg http://localhost:7323/enroll
curl --fail -F image=@probe.jpg http://localhost:7323/recognize/image
```

## Video requests

To recognize your own saved video, also mount a local input folder read-only:

```sh
mkdir -p input
# Place clip.mp4 in input/ first.
docker run --rm --gpus all --name face-attendance \
  -p 127.0.0.1:7323:7323 \
  -v face-attendance-data:/app/data \
  -v "$PWD/input:/app/input:ro" \
  bmaged23/face-recognition-attendance:gpu
```

```sh
curl --fail -H 'Content-Type: application/json' \
  -d '{"video_path":"/app/input/clip.mp4","create_annotated_video":true}' \
  http://localhost:7323/video-recognition
curl --fail http://localhost:7323/jobs/RETURNED_JOB_ID
curl --fail -o results.json http://localhost:7323/jobs/RETURNED_JOB_ID/artifacts/results
curl --fail -o annotated.mp4 http://localhost:7323/jobs/RETURNED_JOB_ID/artifacts/video
```

Wait for `status: "done"` before downloading. The host and container paths differ;
use `/app/input/clip.mp4` in JSON. This local API has no authentication: keep the
published port on localhost. Camera devices/RTSP are separate native CLI routes.

## Updates and source builds

Pulling a newer tag does not replace an existing container. Stop it, pull, and
run the same command/volume again. Fingerprints reject incompatible model
galleries; enroll again into a fresh collection when changing checkpoint files.

```sh
git clone https://github.com/bmaged23/face-recognition-attendance.git
cd face-recognition-attendance
docker build -f deployment/docker/Dockerfile -t face-recognition-attendance:gpu .
```

Compose is an optional external-Qdrant route: copy `.env.example` to `.env` and
run `docker compose up --build -d`. It selects the maintained recipe and clears
`QDRANT_PATH` so `QDRANT_URL` selects its database service. This route must be
qualified separately from the single-container embedded-store release.

[API guide](../api/README.md) · [CI/CD flow](README.md) ·
[GitHub source](https://github.com/bmaged23/face-recognition-attendance) ·
[Docker Hub](https://hub.docker.com/r/bmaged23/face-recognition-attendance).

For the immutable qualified release:

```sh
docker pull bmaged23/face-recognition-attendance@sha256:9a2fcd1a0de9c082d58dcc787814896fe75cec297a7d74a04f9c5d7f1a5e289f
```
