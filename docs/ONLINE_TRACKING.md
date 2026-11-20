# Complete frame-sequence inference

The supported input is an ordered directory of JPG/PNG frames and one initial
`x y width height` box. Subsequent GT is never read. Encoded video files are not
the interface: extract their frames first. Run from the repository root.

## Required resources

1. Compatible CUDA PyTorch, native tracker checkpoint and matching S9.
2. Local Hugging Face CLIP ViT-B/32 directory and a Python environment that can
   import `transformers`, `torch`, PIL and NumPy (`--clip-python`).
3. A populated **private** prompt directory matching `prompts/index.json`.
   Public prompt text is deliberately withheld; empty files fail before any
   HTTP request. Exact semantic reproduction requires the original prompts.
4. An OpenAI-compatible multimodal endpoint supporting the actual logprob
   scoring/image payloads. Merely serving a text chat endpoint is insufficient.

No backend address, API key or credentials are supplied. The conditional
YES/NO scores are not calibrated probabilities.

Only trusted checkpoint files should be loaded: upstream files use Python
pickle. The compatibility loader maps old module names, not model tensors.

## Configure ARTrackV2 without editing code

```bash
python tracking/configure_online.py \
  --tracker artrackv2 --domain lasot \
  --native-checkpoint checkpoints/official_base256_got.pth.tar \
  --predictor checkpoints/s9_b256_seq_got_64k.pt \
  --clip checkpoints/clip-vit-base-patch32 \
  --prompts configs/local_prompts \
  --mllm-url http://127.0.0.1:8001/v1/chat/completions \
  --mllm-model YOUR_SERVED_MODEL --q 0.4 \
  --out configs/local_artrackv2.json

CUDA_VISIBLE_DEVICES=0 python tracking/test.py \
  --config configs/local_artrackv2.json \
  --sequence data/sequence/img --init-box 367 101 41 16 \
  --output outputs/demo_artrackv2
```

The box above is only an example: replace it with the first frame's target box.
Pass `--clip-python` if CLIP uses a separate Python environment. New configs store
paths inside the checkout relative to the repository root. Paths to external
resources supplied by the user retain their absolute locations.
Add `--max-frames 12` for a smoke test; omit it to process the entire sequence.
The native Base256-seq-GOT checkpoint identity is checked. The legacy S9 from
the [Model Zoo](MODEL_ZOO.md) was trained with ARTrackV2-Base256-seq-GOT and is
for diagnostics only, **not the final benchmark checkpoint**.

## OSTrack and SUTrack

Use the same two commands, setting `--tracker ostrack` or `--tracker sutrack`,
and supply that tracker's native checkpoint **and its own trained S9**. Do not
reuse the AR S9. The current adapters support the pinned OSTrack ViT-B/384 CE
and SUTrack-L384 checkpoints, respectively. The matching native-response S9
contract is `NativeCachedPastS9`, GOT-train, complete 4k paired stage; its weights
are not distributed in this release. The generated configs select the matching
upstream YAML automatically. See MODEL_ZOO.md for pinned hashes.

`--q` is explicit and is not automatically calibrated. Defaults include
`p_high=.7`, `q_trusted=.8`; if Q is higher, specify `--q-trusted` explicitly,
with `0 < Q < q_trusted < 1`. Changing it changes the protocol. These examples
are executable integration profiles, not a claim that every historical paper
run used one common threshold.

## Actual runtime

Native prediction -> causal BQP -> quality/identity routing -> optional global
or local proposal -> isolated native trial -> candidate checks -> guarded
bbox feedback -> history update. ATOM is disabled. Independent trials restore
native fallback state, including AR recursive features and mutable embedding
weights. Accepted feedback changes the bbox state only; it does not replace
native recursive prompts or the native dynamic template bank.

The default controller uses the fused representative/recent trusted history,
periodic audit interval 90, and semantic conflict routing. Uncalibrated codebook
cosine is not used as an automatic identity acceptance threshold. All resolved
FrameworkConfig values are written into the run protocol.

## Outputs and failure handling

- `predictions.txt`: one full-image xywh per input frame, including initial box.
- `decisions.jsonl`, `memory.jsonl`: causal decisions and memory updates.
- `evidence/`: actual request metadata, images and replies; keep private.
- `protocol.json`: ordered frame names, config, model hashes and feedback mode.
- `completion.json`: frame/call counts and wall time only after success.
- `failure.json`: errors, with all partial outputs preserved.

Existing output directories are refused. There is no unsafe mid-sequence
resume; use a new output directory for an explicit retry. HTTP failures are
not silently treated as successful decisions. This launcher does not package
GOT/TrackingNet submissions or invent official metrics. Evaluate only with the
appropriate dataset protocol and genuine authorised annotations.

The paired predictor metrics in [Training](TRAINING.md) are not tracking
metrics. For offline tracking metrics or submission files, evaluate the final
predictions with the benchmark's official protocol. Run each sequence in a
separate process and output directory.
