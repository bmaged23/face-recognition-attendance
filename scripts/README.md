# Project tools

Run commands from the repository root with the project environment active.
Application code lives in `face_recognition/`; these files support setup,
evaluation, validation, release and documentation.

| Tool | Purpose |
| --- | --- |
| `download_model.py` | Download the pinned original checkpoint and verify its SHA256 |
| `check_gpu.py` | Check CUDA kernels and model architecture using random backbone weights; not an accuracy test |
| `serve_demo.py` | Serve the approved local video playlist with seeking and private-path exclusion |
| `build_video_demo.py` | Rebuild the combined approved MP4 and looping GIF |
| `build_pipeline_diagram.py` | Generate the pipeline SVG and animated GIF |
| `setup/export_inference_checkpoint.py` | Remove unused training tensors and verify exact inference-tensor equality |
| `evaluation/prepare_recognition_cases.py` | Verify archives and prepare deterministic labeled research cases |
| `evaluation/evaluate_recognition.py` | Run real detector/embedding/search evaluation and sample resources |
| `evaluation/publish_native_results.py` | Publish native reports/chart and deliberately regenerate frozen regression references; never run automatically in CI |
| `evaluation/monitor_container_resources.py` | Sample Docker CPU/RAM and whole-device GPU/VRAM during qualification |
| `validation/validate_real_api.py` | Test real HTTP enrollment, recognition, attendance, annotations and downloads against an isolated gallery |
| `release/assert_quality.py` | Enforce fixture/model identity, complete execution and frozen recognition quality thresholds |
| `release/qualify_image.py` | Test the exact candidate/public image, API, downloads and gallery persistence after restart |
| `release/update_dockerhub.py` | Synchronize the maintained public image overview using a private environment credential |
| `documentation/summarize_component_timings.py` | Summarize recorded native stage timings, excluding skipped calls |
| `datasets/` | Earlier optional research preparation, comparisons and demo reproduction; see its [tool index](datasets/README.md) |

Raw inputs and transient outputs belong in ignored `input/` and `output/`.
Only intentionally reviewed, measured reports belong in `docs/evaluation/`.
Keep credentials in local environment configuration, never in scripts or reports.
