# Reproducibility boundaries

The source tree retains the actual implementations with namespace/path
adaptation, pinned weight identities, private-prompt injection and transactional
trial state. `tracking/test.py` supplies the complete ordered-frame loop for
ARTrackV2, OSTrack and SUTrack. `tracking/predictor.py` supplies legacy paired
fine-tuning/evaluation; it is not a new raw-data construction pipeline.

## Integration checks

- All three adapters completed separate 12-frame checks with native models,
  matching S9 predictors and a separate CPU CLIP environment. These checks
  issued no MLLM calls.
- Each adapter also completed a separate live three-frame check with five 4B
  requests. These checks audited every frame; AR used Q=.79. The AR check
  exercised an accepted global-trial correction and guarded bbox feedback.
  The OS and SU checks exercised rejection and native fallback.
- A separate trial on each tracker computed a candidate box and quality score
  and restored native fallback state exactly under the snapshot contract.
  Synthetic accepted-box tests checked that only bbox state changed.
- The legacy paired CLI completed one CUDA training update and four evaluation
  pairs. [Historical check counts](VALIDATION_SUMMARY.json) record the tested
  export, not an independent benchmark rerun.

These checks are not a guarantee of bitwise equality or clean-machine
installation, and shared-server integration timings are not paper FPS.

## Release boundaries

The [Model Zoo](MODEL_ZOO.md) provides a legacy S9 trained with
ARTrackV2-Base256-seq-GOT for diagnostics, not the final benchmark checkpoint.
Native weights, final predictors, original prompts, datasets and request caches
are not distributed in this repository. The prompt templates are empty; the
online entry point requires separately supplied prompts and matching resources.
This source preview may change and is not the final implementation described in
the paper.
Original QIRTrack code is licensed under Apache-2.0; nested third-party
restrictions are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
[Source provenance](SOURCE_PROVENANCE.json) records source/export hashes.
Internal experiment archives and raw outputs are kept separately. Review the
rights to third-party files, figures and external weights before redistributing
this repository or its assets.
