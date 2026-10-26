# Model and runtime provenance

- AdaFace inference source: official `mk-minchul/AdaFace`, commit
  `c60eaa786a42c03444f3df7096dbaf9d57ae010d`. Retained MIT notice:
  `vendor/adaface/LICENSE` (Minchul Kim, 2022).
- MTCNN implementation/weights: retained vendored inference dependency and
  `vendor/adaface/face_alignment/mtcnn_pytorch/LICENSE` (MIT, Dan Antoshchenko,
  2017). Local changes are listed in `vendor/README.md`.
- Recognition checkpoint: AdaFace WebFace12M R100 (`ir_101` upstream backbone),
  pinned mirror revision `534fa44f9499437645bd18eee376e52fd60bd931` at
  https://huggingface.co/VishalMishraTss/AdaFace . Original SHA256:
  `0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0`.
- Docker includes only the inference tensor export, SHA256
  `2bafbe30750677b7165236dbe05054a179c716f29fcbd2f47475aaeaa61acd95`.
  All 917 retained tensors are exactly equal; unused training tensors and
  classifier are omitted. Original model provenance and upstream terms remain
  applicable; do not interpret a source-code notice as a biometric deployment
  certification or as a new license for upstream training data.

AdaFace paper: Minchul Kim, Anil K. Jain, Xiaoming Liu, *AdaFace: Quality Adaptive
Margin for Face Recognition*, CVPR 2022. Official source and method:
https://github.com/mk-minchul/AdaFace .

ChokePoint fixtures and source demo are separate research assets covered by the
retained [source license](SOURCE_AND_LICENSE.html) and
[modification notice](MODIFIED_DATA_NOTICE.txt). They are not included in the
runtime image. The image includes dependency distributions and their notices;
no ownership of upstream libraries/models is claimed. Optional InsightFace
models are not included or certified by the AdaFace release.
