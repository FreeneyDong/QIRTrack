# Reproducibility boundaries

The source tree retains the actual implementations with namespace/path
adaptation, pinned weight identities, private-prompt injection and transactional
trial state. `tracking/test.py` supplies the complete ordered-frame loop for
ARTrackV2, OSTrack and SUTrack. `tracking/predictor.py` supplies legacy paired
fine-tuning/evaluation; it is not a new raw-data construction pipeline.

Short CUDA tests, unit tests, trial state isolation and real-MLLM request checks
are recorded in INTEGRATION_STATUS.md. They are not full benchmark re-runs or
a guarantee of bitwise equality on different hardware/software. Timings from
short shared-server integration tests are not paper FPS measurements.

Only the legacy test-trained AR S9 is included locally. Native weights, final
predictors, original prompts, datasets and request caches are not distributed.
Thus complete online execution is possible after providing those resources,
but exact turnkey reproduction of all paper tables is deliberately not offered.
Original-code licensing remains pending; nested third-party restrictions are
documented in THIRD_PARTY.md. SOURCE_PROVENANCE.json tracks source/export hashes.
