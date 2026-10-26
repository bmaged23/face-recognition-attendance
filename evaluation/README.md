# Frozen recognition references

`cases.json` contains the human identity/eye annotations, deterministic sampling
choices and image hashes for the 228-case ChokePoint research evaluation. The
actual images are kept locally under ignored `input/recognition-evaluation/`.
SHA256 of the byte-identical fixture manifest is checked by the release gate.

`reference.json` is a model regression baseline with explicit quality floors.
Its predictions are not ground truth. Identity correctness is measured using
the independent dataset labels. Regenerating references is an intentional,
reviewed development operation, never an automatic CI step.

Raw images, personal identities, biometric embeddings and attendance state must
not be committed. These annotation derivatives retain ChokePoint's
[noncommercial research notice](../docs/licenses/SOURCE_AND_LICENSE.html) and
[modification notice](../docs/licenses/MODIFIED_DATA_NOTICE.txt). See the full
[protocol, citation, results and limitations](../docs/evaluation/README.md).
