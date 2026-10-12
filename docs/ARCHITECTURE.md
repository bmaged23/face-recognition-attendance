# Architecture

The two recognition entry points differ only in their input adapter. Both
create the same `Runtime`, `RecognitionEngine`, and `StreamPipeline`.

| Component | Responsibility |
|---|---|
| `config.py` | Validate environment-derived model, device, collection and thresholds |
| `runtime.py` | Construct the face backend and Qdrant repository |
| `core/backend.py` | Load the pretrained backbone, detect/align faces, produce embeddings |
| `core/images.py` | Decode and validate images, including EXIF orientation |
| `core/engine.py` | Match embeddings and return known or unknown faces |
| `enrollment/service.py` | Require one face; create/update a person reference |
| `storage/people.py` | Store UUID/name/embedding; validate model signature; cosine top-2 search |
| `video/tracking.py` | One-to-one previous-frame IoU association and timed recognition refresh |
| `video/pipeline.py` | Capture frames, calculate time, observe results, render/encode video |
| `video/results.py` | Deduplicate first sightings per UUID within a run |
| `storage/attendance.py` | Persist first check-in/check-out with a file lock and atomic writes |
| `video/annotation.py` | Draw current face boxes, names and optional cosine percentages |
| `video/attendance_panel.py` | Draw a rolling list of attendance names |
| `video/encoding.py` | Convert output to H.264/yuv420p MP4 with fast-start metadata |
| `api/` | Validate HTTP inputs and manage background saved-video jobs |

## Identity and matching

One reference embedding is stored per UUID. Names are display values, not keys.
Passing an existing UUID to enrollment replaces that person's reference; a new
enrollment creates a new UUID. Qdrant records include a signature of the model
weights and alignment/color convention. Incompatible collections are rejected.

For a fresh recognition result, Qdrant returns the best two cosine candidates.
Top-1 must meet the confidence threshold and exceed top-2 by the configured
margin. The runner-up is not filtered by the absolute threshold. A collection
with only one identity uses the absolute threshold alone.

## Time and tracking

Video observations use source presentation times, with frame index/FPS as a
fallback. An absolute `--start-time` maps video offsets into real datetimes;
attendance requires this start time with a timezone. Live observations use the
current local time. The attendance time represents the first accepted sighting
for the chosen action, not the last frame someone is visible.

With default tracking, detection and alignment run every frame. Greedy IoU
association links each current detection to at most one previous detection.
New/unmatched faces and tracks due for recognition receive fresh embeddings;
others reuse the previous name and score with the current bounding box.
Unmatched previous tracks are dropped immediately. Tracking state is local to
one stream/run. This is a lightweight box association, not a learned identity
tracker, and can confuse people during crossings.

## Attendance and annotation

Recognition emits `{id, name, time}` once per recognized UUID per run.
Attendance emits `{id, name, checkin, checkout}` and updates a shared state file
for the selected action. Unknown identities do not update either output.
Recognition mode never touches attendance state. Check-in and check-out are
caller-selected actions; repeated sightings never toggle a state.

All detected faces are eligible for annotation, including unknown faces.
Attendance adds a heading and a bounded list of names on the left. Only the
visible list scrolls; persisted results retain every confirmed person.
Video encoding is finalized after capture stops and removes audio.

## Resources and boundaries

The AdaFace backend uses a lock around GPU model operations; the API initializes
its runtime lazily and reuses it across jobs. Job metadata is in memory; result
files are on disk. Qdrant's Docker volume persists references. Camera/RTSP
sources are handled by the CLI, not an API endpoint that opens a server camera.

The inference subset under `vendor/adaface/` and its licenses are part of the
source distribution. The large recognition checkpoint, source datasets,
generated output, previous implementation, and working notes are local-only
ignored files. Dataset utility scripts are separate from application imports.

## Diagram review

The animated GIF and static SVG are built from one node/edge definition in
`scripts/build_pipeline_diagram.py`. Their branches were checked against
`EnrollmentService`, `IoURecognitionEngine`, `RecognitionEngine`,
`SessionResults`, and `AttendanceStore`: enrollment is separate from stream
inference, IoU reuse bypasses fresh embedding/search, unknown results are
omitted from persisted records, and attendance actions are explicit.
