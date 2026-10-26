# Recognition API

The FastAPI service wraps the same enrollment, recognition and attendance code
as the command-line programs. Run it locally with one worker:

```sh
.venv/bin/python -m uvicorn face_recognition.api.app:app --host 127.0.0.1 --port 7323
```

Open http://localhost:7323/docs for the interactive OpenAPI schema. This local
API has no authentication. Keep it bound to localhost; video paths and optional
attendance output paths refer to files on the server. It is not a public,
multi-tenant service or a real-time video streaming API.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Liveness only; does not load models or certify inference readiness |
| `GET /ready` | Loads the configured recognition backend, checks the gallery, returns actual device; HTTP 503 if unavailable |
| `POST /enroll` | Multipart `name`, `image`, optional existing `person_id`; requires exactly one face |
| `POST /recognize/image` | Multipart `image`, optional `confidence` query parameter; recognized people only |
| `POST /video-recognition` | JSON file path and workflow settings; returns an asynchronous job ID |
| `GET /jobs/{job_id}` | Pending/running/done/failed status, result or error |
| `GET /jobs/{job_id}/artifacts/results` | Completed JSON result download |
| `GET /jobs/{job_id}/artifacts/video` | Completed annotated MP4; HTTP byte ranges support seeking |

```sh
curl --fail http://localhost:7323/ready
curl --fail -F 'name=Alice Smith' -F image=@alice.jpg http://localhost:7323/enroll
curl --fail -F image=@probe.jpg http://localhost:7323/recognize/image
curl --fail -H 'Content-Type: application/json' \
  -d '{"video_path":"/absolute/path/to/clip.mp4","create_annotated_video":true}' \
  http://localhost:7323/video-recognition
curl --fail http://localhost:7323/jobs/RETURNED_JOB_ID
```

Attendance requests additionally set `mode: "attendance"`, `action: "checkin"`
or `"checkout"`, and a timezone-aware `start_time`. Unknown faces never create
attendance records. Similarity values are not calibrated probabilities.

By default the app connects to `QDRANT_URL`. To use one persistent embedded
Qdrant store instead, set `QDRANT_PATH=/absolute/path/to/gallery` before startup.
This selects embedded storage regardless of `QDRANT_URL`; use one application
process per local store. Keep gallery embeddings and attendance records private.

The queue defaults to one worker, capacity four (running + queued), and 128
completed metadata entries (`JOB_WORKERS`, `JOB_CAPACITY`, `JOB_HISTORY`).
Full queues return HTTP 429 with Retry-After. Metadata expires on restart or
history pruning; files remain under `OUTPUT_DIR` (default `output/`).
