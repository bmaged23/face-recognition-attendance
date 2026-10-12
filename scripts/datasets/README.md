# Optional dataset utilities

These scripts are experiments, not prerequisites for the application. Run them
from the project root using the project `.venv`. Source archives, labels,
reference photos, and model weights must already be available locally.
Large datasets are not committed. Retained source data lives under `input/`.
Generated experiment results live under `output/` and are ignored.

| Script | Purpose |
|---|---|
| `prepare_chokepoint.py` | Extract camera-2 S5 JPEG frames and create full-frame MP4s, holding missing frames to retain timing |
| `test_chokepoint.py` | Enroll the grayscale baseline and recognize crowded clips in an isolated local Qdrant store |
| `test_chokepoint.py --color-references` | Enroll one selected color reference per person and regenerate outputs in `output/chokepoint_color_test/` |
| `evaluate_chokepoint.py` | Compare galleries, extract labeled probes, calibrate matching and cache actual GPU S5 embeddings |
| `compare_chokepoint.py` | Compare recognition policies against identical cached observations and render improved videos |
| `test_vidtimit.py` | Enroll and test two VidTIMIT participants using separate capture sessions |
| `test_attendance_annotation.py` | Replay first-confirmation events over baseline annotated clips to examine the rolling panel |
| `build_face_examples.py` | Use the real GPU backend to render detected face crops, aligned 112×112 inputs, and embedding examples for the README |

## ChokePoint prerequisites

Acquire the dataset from the [official page](https://arma.sourceforge.net/chokepoint/)
or its [Zenodo record](https://zenodo.org/records/815657). Keep the license and
modified-data notices. The local evaluation used the following archives under
`input/chokepoint/`: `P2E.tar.xz`, `P2E_S2.tar.xz`, `P2E_S5.tar.xz`,
`P2L_S5.tar.xz`, `Still.tar.xz`, and `groundtruth.tar.xz`.

Extract `Still.tar.xz` under `input/chokepoint/stills/` and the ground-truth
archive under `input/chokepoint/groundtruth/`. Preparation/evaluation handles
the nested camera archives with safe tar extraction and expected checksums.
The color-gallery comparison needs a baseline manifest. This workspace retains
it in `input/chokepoint/baseline_enrollment.json`; on a fresh dataset setup,
create baseline references using `test_chokepoint.py` first.

```sh
python scripts/datasets/prepare_chokepoint.py
python scripts/datasets/test_chokepoint.py
python scripts/datasets/evaluate_chokepoint.py --stage gallery
python scripts/datasets/evaluate_chokepoint.py --stage probes
python scripts/datasets/evaluate_chokepoint.py --stage calibrate
python scripts/datasets/evaluate_chokepoint.py --stage crowded
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/datasets/compare_chokepoint.py
```

The `crowded` stage runs detection and embedding inference on the GPU for all
frames. Policy comparison replays cached vectors; it does not perform AdaFace
inference on CPU. Its generated test attendance dates are synthetic.

`compare_chokepoint.py` **regenerates** the improved videos and resets its demo
attendance file. Do not run it merely to view the approved videos; use
`python scripts/serve_demo.py` instead. Baseline scripts recreate disposable
results that were removed during the presentation cleanup.

## Evidence and limitations

The saved evaluation reports are under `docs/evaluation/`. The labeled
recognition result is conditional on detection. Development camera 2 and
held-out camera 1 share the same S2 recording. S5 crowded clips lack supplied
identity ground truth; policy disagreements and accepted-face counts are not
accuracy metrics. Test names are aliases.

ChokePoint is limited to noncommercial research and personal experimentation.
Refer to `docs/licenses/SOURCE_AND_LICENSE.html` and
`MODIFIED_DATA_NOTICE.txt` for its original notice and derivative description.
