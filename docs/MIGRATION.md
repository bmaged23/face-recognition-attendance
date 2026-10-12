# Migration from the previous attendance project

The active application is now `face_recognition/`, with two main programs:
`video_recognition.py` and `live_camera_recognition.py`. `enroll.py` is a utility
that both workflows rely on. Model code is shared, and attendance receives
recognition results rather than owning detection and recognition.

| Previous component | Active replacement |
|---|---|
| `Attendance_full_pipeline_api.py` | `face_recognition/api/app.py` |
| `Det_Rec_Attendance_full_pipeline.py` | `face_recognition/video/pipeline.py` + `storage/attendance.py` |
| `video_detection_pipeline.py` | Same shared stream pipeline in recognition mode |
| `Face Rec/Test_live_recognition.py` | `live_camera_recognition.py` |
| `Face Rec/enrollment.py`, `enroll_trainees.py` | `face_recognition/enrollment/service.py` + `enroll.py` |
| `Face Rec/adaface.py` | `face_recognition/core/backend.py` + `storage/people.py` |
| `config.py` | `face_recognition/config.py` + `.env.example` |
| Arabic annotation, images in attendance JSON | English OpenCV labels and minimal person records |
| Course scheduler, spreadsheet assumptions, camera crawler | Archived under `archive/legacy/` |

The previous source files, documentation, dependency list, and deployment files
were moved into `archive/legacy/` for reference. They are **not active entry points** and
their original relative paths are not maintained. The previous vendored AdaFace
repository moved to `vendor/adaface/`. Its MTCNN weight loading now uses absolute
paths relative to the module instead of changing the process working directory,
so initializing a detector cannot change another job's file paths.
Two alignment-method comparisons also use value equality instead of Python
object identity, avoiding incorrect selection for dynamically constructed strings.

Old national-ID collections and JSON records are not automatically migrated.
Create a fresh collection and re-enroll each person with an English name and
one photo. Enrollment creates a new UUID, and stores no national ID or legacy
metadata. Existing Qdrant data is not deleted. An old or incompatible collection
is rejected instead of silently mixing embeddings or personal identifiers.

AdaFace preprocessing now follows the vendored `inference.py`: align the RGB
face, then feed **BGR** channels normalized to `[-1, 1]` to the model. The previous
wrapper fed RGB channels. Fresh enrollment ensures reference and probe inputs
use identical preprocessing; old embeddings cannot be reused directly. Model
weights are fingerprinted alongside the schema/alignment identifier.

Attendance JSON is a list of `{id, name, checkin, checkout}` records. It contains
no images, national IDs, course fields, room metadata, or nested person objects.
Recognition JSON is a list of `{id, name, time}` records. Both identify a person
by UUID, never by name. Unknown faces are omitted, and repeated sightings create
one record per person per run.

The API contract changed: `/video-recognition` supports both modes. The old
course/batch/annotator endpoints are archived rather than continuing to expose
the project-specific attendance contract. Live cameras use the camera CLI.
Existing clients must update their endpoint, request, and response handling.

The Qdrant adapter uses the current [point API](https://qdrant.tech/documentation/manage-data/points/)
and [query API](https://qdrant.tech/documentation/search/) rather than deprecated
search endpoints.

Environment changes: use `MODEL_NAME` instead of separate family-specific model
variables, `QDRANT_URL` instead of host/port, `DEVICE` instead of `YOLO_DEVICE`,
and `RECOGNITION_CONFIDENCE` instead of `SCORE_THRESHOLD`. Legacy cutting,
tracking, course, and Arabic-font settings are not used. There is no YOLO or
RetinaFace crop stage in the active pipeline; native MTCNN or SCRFD detects and
aligns faces directly from each detected frame.
