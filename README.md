# QIRTrack

**Quality–Identity Routing for Single Object Tracking**

[Installation](docs/INSTALL.md) · [Quickstart](docs/QUICKSTART.md) · [Model Zoo](docs/MODEL_ZOO.md) · [Online Tracking](docs/ONLINE_TRACKING.md)

A tracker can remain confident while persistently mislocalising its target. QIRTrack addresses this failure mode, **confident drift**, by checking localisation quality and target identity separately. It uses a multimodal large language model (MLLM) selectively and verifies proposed corrections before accepting them.

## Framework

![QIRTrack framework: quality estimation, identity-conditioned routing, and verified recovery](assets/framework.png)

- **Bounding Box Quality Predictor (BQP)** estimates the quality of boxes already supplied by the tracker or recovery trial. It uses target-conditioned visual features, past observations and box geometry to help decide when to query; it does not predict a second tracking trajectory.
- **Quadrant-based Intervention Controller (QIC)** uses quality and semantic identity evidence to choose whether to retain, refine or recover a box. For recovery, the MLLM provides identity evidence or a coarse proposal; an isolated tracker trial and candidate postcheck validate the correction.
- **Protected Identity Codebook (PIC)** maintains diverse, screened references for later identity checks. Accepting a corrected output and writing that correction back to the tracker are separate decisions.

## Qualitative examples

**Detecting and interrupting drift (`dog-19`, LaSOT).** The native tracker stays confident after switching to a distractor. The timeline aligns the quality estimate, semantic queries and verified recovery with the two tracking trajectories.

<img src="assets/confident-drift-dog19.png" alt="LaSOT dog-19: native trajectory, QIRTrack trajectory, and evidence timeline" width="660">

<details>
<summary>Inspect the tracker after a correction (umbrella-19)</summary>

This case compares output-only correction with bounding-box feedback from the same correction point. The panels show subsequent attention and boundary readout; they illustrate one sequence, not an aggregate performance result.

<img src="assets/feedback-umbrella19.png" alt="LaSOT umbrella-19: output-only versus bbox feedback, with layer attention and boundary readout" width="740">

</details>

## Get started

Tracker adapters are provided separately for [ARTrackV2](lib/trackers/artrackv2/), [OSTrack](lib/trackers/ostrack/) and [SUTrack](lib/trackers/sutrack/). The shared QIRTrack components are under [`lib/models/qirtrack/`](lib/models/qirtrack/).

1. Follow the [installation instructions](docs/INSTALL.md). The [Model Zoo](docs/MODEL_ZOO.md) identifies the matching native tracker profiles and lists the available predictor checkpoint.
2. Use the [Quickstart](docs/QUICKSTART.md) for configuration, offline predictor evaluation and legacy S9 fine-tuning. Check the installation with `python tracking/preflight.py --offline`.
3. To run the full tracker on ordered image frames with a first-frame box, follow the [online tracking guide](docs/ONLINE_TRACKING.md) and run `python tracking/test.py` with the generated configuration. The online path requires a configured MLLM backend and user-supplied prompts.

### Installation

The commands below use Bash on Linux. Install a CUDA-compatible PyTorch and torchvision build first, and have a compatible CUDA toolkit, C++ compiler and Ninja available for PreciseRoIPooling. See the [installation guide](docs/INSTALL.md) for the tested package versions.

```bash
git clone https://github.com/FreeneyDong/QIRTrack.git
cd QIRTrack
python -m pip install -r requirements.txt
python -m pip install -e .
python tracking/preflight.py --offline
```

### Predictor fine-tuning and offline evaluation

Download the legacy S9 checkpoint from the [Model Zoo](docs/MODEL_ZOO.md) and save it as `checkpoints/s9_b256_seq_got_64k.pt`. Set the paths below to your labelled pair manifests and cached tracker observations; these inputs are not bundled. Run the input check first, then run fine-tuning or offline evaluation as needed.

```bash
python tracking/configure.py \
  --data-root /path/to/pair_data --output-root /path/to/predictor_outputs \
  --checkpoint checkpoints/s9_b256_seq_got_64k.pt

python tracking/predictor.py evaluate \
  --config configs/local.json --manifest /path/to/eval_pairs.jsonl \
  --run-name input_check --validate-only

CUDA_VISIBLE_DEVICES=0 python tracking/predictor.py train \
  --config configs/local.json --manifest /path/to/train_pairs.jsonl \
  --run-name s9_finetune --steps 4000 --batch-size 8

CUDA_VISIBLE_DEVICES=0 python tracking/predictor.py evaluate \
  --config configs/local.json --manifest /path/to/eval_pairs.jsonl \
  --run-name legacy_s9_eval --q 0.4
```

The training command fine-tunes the diagnostic checkpoint; it does not train the native tracker or reproduce the paper's training from scratch. The evaluation command above scores the checkpoint selected in `configs/local.json`. To evaluate the fine-tuned checkpoint, generate a separate config pointing to it. See [Quickstart](docs/QUICKSTART.md) for the pair schema.

### Track an image sequence

This diagnostic example uses ARTrackV2-Base256-seq-GOT. Supply its native checkpoint, the legacy S9 predictor, a local CLIP ViT-B/32 directory, a populated private prompt directory and a multimodal endpoint that supports image input and log probabilities. The original prompts are not supplied.

```bash
python tracking/configure_online.py \
  --tracker artrackv2 --domain lasot \
  --native-checkpoint /path/to/official_base256_got.pth.tar \
  --predictor checkpoints/s9_b256_seq_got_64k.pt \
  --clip /path/to/clip-vit-base-patch32 \
  --clip-python /path/to/clip-env/bin/python \
  --prompts /path/to/private-prompts \
  --mllm-url http://127.0.0.1:8001/v1/chat/completions \
  --mllm-model YOUR_SERVED_MODEL --q 0.4 \
  --out configs/local_artrackv2.json

CUDA_VISIBLE_DEVICES=0 python tracking/test.py \
  --config configs/local_artrackv2.json \
  --sequence /path/to/sequence/img --init-box 367 101 41 16 \
  --output outputs/demo_artrackv2 --max-frames 12
```

Replace the example box with your target's first-frame `x y width height` box. Omit `--max-frames 12` to process all ordered JPG/PNG frames; use a new output directory for every run. Results are written to `predictions.txt` in that directory. For OSTrack or SUTrack, change `--tracker` and supply both that tracker's native checkpoint and its matching predictor; the ARTrackV2 S9 is not interchangeable. See [Online Tracking](docs/ONLINE_TRACKING.md) for the remaining options.

## Weights and release scope

[Download legacy S9 (Google Drive, restricted access)](https://drive.google.com/file/d/1JjT9JF8ryTl1RcfIy_sV0NMC0oBHGSXI/view) · [Placement and checksum](docs/MODEL_ZOO.md)

This is trained with ARTrackV2-Base256-seq-GOT, which is **not the final benchmark checkpoint**. Native tracker weights are provided by their upstream projects.

See [reproducibility](docs/REPRODUCIBILITY.md) and [release scope](docs/RELEASE_SCOPE.md) for the qualified workflows and limitations. Please note that we made some changes to the code and retrained the model after submitting this repository, thus it is not the final version of the paper.

## Acknowledgements and licensing

The repository builds on [ARTrackV2](https://github.com/MIV-XJTU/ARTrack), [OSTrack](https://github.com/botaoye/OSTrack) and [SUTrack](https://github.com/chenxin-dlut/SUTrack). [Third-party notices](docs/THIRD_PARTY.md) are retained. A licence for the original QIRTrack code has not yet been selected. See [CITATION.cff](CITATION.cff) for citation status.
