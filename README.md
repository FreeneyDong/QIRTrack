# QIRTrack

Quality–Identity Routing for Single Object Tracking.

This is a **release-preparation source tree**, extracted from the implemented research code. It is not a promise of a one-command reproduction of all paper tables. ARTrackV2, OSTrack and SUTrack implementations are stored separately under `lib/trackers/`; BQP, QIC and PIC are shared under `lib/models/qirtrack/`.

## Important boundaries

- MLLM prompt files are deliberately empty. No original prompts or responses are distributed. Online inference must supply prompts and pass the prompt preflight; exact semantic reproduction is not possible without them.
- Only the legacy ARTrackV2-Base256-seq-GOT S9 64k diagnostic checkpoint is locally included. **It is not the final benchmark model.** It was trained on LaSOT test280 + extension150 and must not be presented as independent LaSOT evaluation.
- Original-code licensing and final citation metadata are pending. Third-party licences are preserved. Do not publish this tree as fully licensed open-source until this is resolved.
- No videos, annotations, experiment outputs, remote credentials, request caches or service launch scripts are included.
- Historical code was namespace/path adapted, not retrained. Short GPU integration checks do not certify a full benchmark reproduction. See `docs/REPRODUCIBILITY.md`.

## Layout

`lib/models/qirtrack/{quality_predictor,trigger,semantic_verifier,evidence_fusion,recovery,state,history}.py` expose the actual core implementations. `core/` retains source-module boundaries for traceability. `lib/trackers/{artrackv2,ostrack,sutrack}/native/` contains each tracker separately, with upstream configs under `experiments/`. `external/` contains required CUDA sources. `prompts/` is empty by design.

## Checks

From the repository root: `python tracking/preflight.py --offline` and `python -m unittest discover -s tests -v`. Online prompt preflight: `python tracking/preflight.py` (expected to fail until prompts are supplied).

See [installation](docs/INSTALL.md), [model zoo](docs/MODEL_ZOO.md), [training](docs/TRAINING.md), [evaluation](docs/EVALUATION.md) and [provenance](docs/SOURCE_PROVENANCE.json).

## Release checklist

See [release scope](docs/RELEASE_SCOPE.md) for the six author-requested packaging constraints, retained interfaces and explicit limitations. Original research archives are maintained separately; their private inventories are not part of this repository.

See [executable commands](docs/QUICKSTART.md) for local path configuration, legacy paired fine-tuning and offline predictor evaluation. For complete ordered-frame inference, use [online tracking](docs/ONLINE_TRACKING.md): configure paths once, then run `python tracking/test.py`. Encoded video decoding and official submission packaging are separate.
