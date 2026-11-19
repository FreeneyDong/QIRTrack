# Evaluation

See [complete frame-sequence inference](ONLINE_TRACKING.md) for the executable
three-tracker entry point. It uses image frames and a first-frame box, not
cached predictions as a substitute for native tracking. Run each complete
sequence in a separate process/output directory.

See [paired predictor commands](QUICKSTART.md) for offline BQP evaluation.
Those metrics are distinct from final tracking metrics. Dataset evaluation and
official submission packaging must use their official protocol; hidden GT is
not provided or inferred here. No benchmark result tables are fabricated.

The downloadable legacy S9 was trained with ARTrackV2-Base256-seq-GOT. It is
for diagnostics only, not the checkpoint used for the paper's final benchmark
results. Other trackers require their own trained predictors, which are not
part of the released weight selection.
