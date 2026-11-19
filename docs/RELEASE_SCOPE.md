# Release scope and repository organisation

1. The research repositories on both servers remain separate rebuttal archives. They are not overwritten by this source distribution. Raw data, outputs, request caches and historical versions are not imported here.
2. Shared BQP/QIC/PIC implementations are under `lib/models/qirtrack/`. Source submodule names are retained inside `core/` to preserve traceability; avoid treating them as independent proposed methods.
3. Tracker-specific code is separate: `lib/trackers/artrackv2/`, `lib/trackers/ostrack/`, `lib/trackers/sutrack/`. Each has a controller and its own native namespace. OSTrack/SUTrack include observation adapters, and native-response training/model contracts are under `lib/models/qirtrack/native/`. Native training/evaluation support stays in each tracker's native tree.
4. The [Model Zoo](MODEL_ZOO.md) links to the legacy S9 checkpoint trained with ARTrackV2-Base256-seq-GOT. It is a diagnostic checkpoint and is not the one used for the paper's final benchmark evaluation. Do not pair it with a different tracker. No predictor, native tracker or MLLM checkpoint is tracked by Git.
5. All 56 MLLM prompt text files are empty. Empty/missing prompts fail before an online request. No prompt responses or research request caches are distributed.
6. Original QIRTrack source code is licensed under Apache-2.0; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). Third-party licences, including non-commercial terms for identified MAE-derived portions, remain intact. Citation author/venue metadata is not fabricated.
7. No raw results, credentials, private server addresses or service launch scripts are included. The README displays selected paper figures; the diagnostic checkpoint is hosted separately.

This repository is a source preview. Files may change and do not represent the final implementation described in the paper. Portable paired-predictor and ordered-frame entry points are documented in [Quickstart](QUICKSTART.md) and [Online Tracking](ONLINE_TRACKING.md). The prompts and final weights are not provided; short integration checks do not establish full benchmark reproducibility.
