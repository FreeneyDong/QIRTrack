# Executable commands and scope

Run from the repository root with the dependencies installed. Only load trusted PyTorch checkpoints/input files. CUDA is required for S9 forward/backward (PreciseRoIPooling).

```bash
python tracking/configure.py --data-root /absolute/path/to/pair_data --output-root /absolute/path/to/new_outputs --checkpoint checkpoints/old_s9_b256_seq_got_diagnostic_64k.pt
python tracking/preflight.py --offline
python -m unittest discover -s tests -v
python tracking/predictor.py evaluate --config configs/local.json --manifest /absolute/path/evaluation_pairs.jsonl --run-name input_check --validate-only
CUDA_VISIBLE_DEVICES=0 python tracking/predictor.py train --config configs/local.json --manifest /absolute/path/train_pairs.jsonl --run-name s9_finetune --steps 4000 --batch-size 8
CUDA_VISIBLE_DEVICES=0 python tracking/predictor.py evaluate --config configs/local.json --manifest /absolute/path/evaluation_pairs.jsonl --run-name legacy_s9_eval --q 0.4
```

The configuration generator refuses to overwrite an existing file. To evaluate the newly trained model, generate a second config using its `checkpoint.pt` and pass that config. A run name must be new. Local paths/configs and outputs are ignored by Git. `path_map` in the local JSON maps an old absolute directory prefix to its new local location without rewriting any cached tensors or source code.

## Exact data contract

The JSONL uses the original labelled-pair schema: `sequence`, 1-based `frame`, unique `sample_id`, `label_valid`, `origin` (`actual_base_local_trial` or `actual_mllm_global_base_trial`), `raw_iou`, `candidate_iou`, `true_gain`, and `raw_inputs`/`candidate_inputs` objects with `file` and `sha256`. The tensor files retain the original 14-field Base256-seq observation contract (K16, 768-dimensional tokens, 4x800 coordinate logits, full 16x16 search tokens and recent-template/candidate fields). The first pair is checked by validate-only; remaining hashes/shapes are checked as loaded, not claimed verified upfront. These cached observations are not bundled; raw videos cannot be passed as tensor caches.

## What these commands do, and do not do

- Train performs paired **fine-tuning from the included old S9**, not raw/U1 pretraining or exact replay of the original 64k training schedule. It preserves the original stratified sampler, frozen base, absolute distribution loss + 10x relative SmoothL1, AdamW settings and gradient clipping. Updates are explicitly additional, not a hidden resume of optimiser state.
- Evaluate reports raw-deduplicated offline predictor MAE/Recall/FPR/F1 and pair ranking. Good IoU >= .5, bad IoU < .4; grey samples are excluded from classification, not MAE. Ranking excludes true-gain ties; predicted ties count as incorrect and are separately reported.
- The paired predictor CLI is not a final tracking test. Use the separate complete sequence runner in ONLINE_TRACKING.md. OSTrack/SUTrack need their own predictors; this paired training CLI only accepts the legacy AR variant.
- The supplied checkpoint was trained on LaSOT test280 + extension150. No independent generalisation claim can be made from evaluating it on those identities. The CLI never invents or silently chooses a validation split.
- Empty MLLM prompts prevent turnkey online reproduction by design. The complete sequence entry point is implemented but requires privately supplied prompts and matching native/predictor weights.

## Complete online tracking

The native tracker/controller loop is now available in `tracking/test.py`; see [ONLINE_TRACKING.md](ONLINE_TRACKING.md). The paired training/evaluation commands above remain unchanged and do not replace this loop.
