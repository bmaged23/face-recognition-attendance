# AdaFace inference dependency

`adaface/` contains the inference subset of the official
[mk-minchul/AdaFace repository](https://github.com/mk-minchul/AdaFace), based on
commit `c60eaa786a42c03444f3df7096dbaf9d57ae010d`. It includes `net.py`, face
alignment and the small MTCNN detector weights. Training scripts, example
images, notebooks, duplicate checkouts and nested Git metadata are omitted.
Cloning the main project supplies these source files; no second AdaFace clone
or installation of its training requirements is needed.

Local adaptations retained from the working project:

- MTCNN includes the existing batch-alignment helpers.
- Detector `.npy` weights load relative to the module; initialization does not
  change the process working directory.
- Alignment-method comparisons use string equality rather than object identity.

AdaFace's MIT notice is in `adaface/LICENSE`. The nested MTCNN implementation
retains its own `face_alignment/mtcnn_pytorch/LICENSE` and upstream README.
See the [AdaFace paper](https://arxiv.org/abs/2204.00964) for the training method.
The active application wrapper is `face_recognition/core/backend.py`.

The large recognition checkpoint is not bundled. Run
`python scripts/download_model.py` from the project root after installing the
main requirements. It downloads the pinned WebFace12M mirror into
`adaface/pretrained/adaface_ir100_webface12m.ckpt` and checks SHA-256. The local
configuration calls this model `ir_100_webface12m`; upstream builds its backbone
as `ir_101`. These names refer to the same selected architecture in this project.

For research into the full upstream training implementation, clone it into a
separate location:

```sh
git clone https://github.com/mk-minchul/AdaFace.git /tmp/AdaFace-research
git -C /tmp/AdaFace-research checkout c60eaa786a42c03444f3df7096dbaf9d57ae010d
```

That research checkout is not required to run this application's inference.
