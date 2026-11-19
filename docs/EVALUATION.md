# Evaluation

See [complete frame-sequence inference](ONLINE_TRACKING.md) for the executable
three-tracker entry point. It uses image frames and a first-frame box, not
cached predictions as a substitute for native tracking. Run each complete
sequence in a separate process/output directory.

See [paired predictor commands](QUICKSTART.md) for offline BQP evaluation.
Those metrics are distinct from final tracking metrics. Dataset evaluation and
official submission packaging must use their official protocol; hidden GT is
not provided or inferred here. No benchmark result tables are fabricated.

The only bundled S9 is the old ARTrackV2-Base256-seq-GOT diagnostic checkpoint.
It was trained on LaSOT test280 plus extension150; do not describe its LaSOT
evaluation as independent held-out performance. Other trackers require their
own trained predictors, which are not part of the released weight selection.
