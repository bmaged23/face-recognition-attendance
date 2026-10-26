# CI/CD and release qualification

The `Checks` workflow installs the exact CPU test environment on GitHub-hosted
Ubuntu and runs all application tests plus dependency/media-tool checks. No
recognition checkpoint or private gallery is needed for that offline suite.

The `Evaluated GPU image release` workflow handles trusted `main` runtime changes:

1. Run the offline suite with pinned CPU dependencies.
2. Build the Linux amd64 GPU candidate on a hosted runner. Download the verified
   training checkpoint, export the exact inference tensors, and include only
   that compact artifact in the final image. Push the candidate to GHCR.
3. The trusted local `face-gpu` runner pulls that exact digest and executes the
   shipped tests inside it. Real GPU MTCNN/AdaFace then evaluates all 228 frozen
   probes and 19 enrollment photos using the local research fixtures.
4. Require the frozen fixture/model fingerprints, all case outcomes unchanged,
   at least 113 correct known matches, at least 56 detected unknown probes,
   zero wrong-known matches and zero accepted unknowns. These are development
   regression floors, not population accuracy guarantees.
5. Test actual HTTP enrollment, known/unknown recognition, check-in/check-out,
   annotations, result downloads, video byte ranges, published host HTTP access,
   and persistent gallery recognition after restarting the exact image.
6. Publish the qualified manifest to Docker Hub using digest-preserving registry
   copy. Check anonymous public manifest identity and update the Hub overview.
7. Pull the public `gpu` tag with an empty Docker credential configuration and
   repeat normal GPU API startup, workflow and restart/persistence checks.

Any failed test/gate blocks the dependent publication stage. Verification after
publication can still fail and must be reported; publication alone is not a
successful release. Runs preserve aggregate artifacts and detailed local test
outputs without uploading source photos, enrolled embeddings or credentials.

## Runner and secrets

GPU stages require the trusted local runner to be online, NVIDIA support in
Docker, and prepared, checksum-matched research fixtures. Hosted runners cannot
access the local GPU or fixture directory. Pull requests run hosted tests only;
untrusted PR code must never execute on the GPU runner. Trusted builds require
the owner actor on `main`. `GITHUB_TOKEN` grants scoped candidate-registry access;
`DOCKERHUB_TOKEN` is an encrypted repository secret for read/write publication.
No key belongs in source files or image layers.

Source/runtime/test/fixture/recipe changes trigger image qualification.
Documentation-only changes run lightweight checks without rebuilding models.
`workflow_dispatch` provides a deliberate full-release rerun. Each attempt uses
an isolated empty gallery and synthetic attendance folder. CI does not replace
users' running applications: CD delivers the image; users pull and run it.

[Checks workflow](../../.github/workflows/checks.yml) ·
[GPU release workflow](../../.github/workflows/release.yml) ·
[Docker instructions](DOCKER.md) · [Evaluation](../evaluation/README.md).
