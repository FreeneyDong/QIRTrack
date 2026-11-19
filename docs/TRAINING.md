# Predictor training and offline evaluation

The [README](../README.md#predictor-fine-tuning-and-offline-evaluation) provides
configuration, input validation, fine-tuning and offline evaluation commands.
These use the [legacy S9 diagnostic checkpoint](MODEL_ZOO.md), trained with
ARTrackV2-Base256-seq-GOT, not the paper's final benchmark weights. The legacy
schedule used 2,000 raw/U1 updates followed by 64,000 paired updates.

## Paired data contract

Each labelled JSONL entry contains `sequence`, a 1-based `frame`, a unique
`sample_id`, `label_valid`, `origin` (`actual_base_local_trial` or
`actual_mllm_global_base_trial`), `raw_iou`, `candidate_iou`, `true_gain`, and
`raw_inputs`/`candidate_inputs` objects with `file` and `sha256`.
The cached tensors use the 14-field Base256-seq observation contract, including
K16 history, 768-dimensional tokens, 4×800 coordinate logits, 16×16 search tokens
and recent-template/candidate fields. `--validate-only` checks the first pair;
other hashes and shapes are checked when loaded. These observations are not
bundled. Raw videos are not a substitute for the cached inputs.

## Scope

- Training fine-tunes the supplied legacy predictor with raw/U1 frozen, using
  stratified sampling, 50-bin absolute IoU supervision and a 10× relative
  Smooth L1 loss. Updates are additional, not a resume of optimiser state.
- Evaluation reports raw-deduplicated MAE, Recall, FPR, F1 and pair ranking.
  Classification uses true IoU below .4 for bad boxes and at least .5 for good
  boxes. The grey band is excluded only from classification. Ranking excludes
  true-gain ties; predicted ties count as incorrect and are reported separately.
- The configuration generator and run outputs refuse to overwrite existing
  paths. Use a new config to evaluate a fine-tuned checkpoint. `path_map` can
  remap cached path prefixes, and the CLI does not choose a validation split.
- Neither command evaluates online tracking. For that, use
  [Online Tracking](ONLINE_TRACKING.md). OSTrack and SUTrack require their own
  predictors; this paired training CLI accepts only the legacy AR variant.
