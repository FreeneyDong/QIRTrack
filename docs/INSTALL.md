# Installation

Run commands from the repository root. Install a PyTorch/torchvision build compatible with your CUDA driver first, then `python -m pip install -r requirements.txt` and `python -m pip install -e .`. The dependency list is not a verified cross-platform lockfile; no new packages were installed on the research servers during packaging.

The required OpenAI CLIP implementation/tokenizer is included under `external/CLIP` with its upstream licence; the requirements file installs it locally. CLIP model weights are not included. PreciseRoIPooling requires a CUDA toolkit and compatible compiler for real GPU execution.

Run `python tracking/preflight.py --offline`, `python -m unittest discover -s tests -v`, and, when the legacy checkpoint is present, `python tracking/check_checkpoint.py`. The online preflight intentionally fails until every withheld prompt is supplied. Supplying a prompt directory with no files is not sufficient.

These checks do not constitute GPU end-to-end benchmark reproduction. See
[Reproducibility](REPRODUCIBILITY.md) for the completed short integration checks.

## CUDA extension check

S9 forward/backward builds PreciseRoIPooling on first use. Ensure `ninja --version`, `nvcc --version` and a compatible C++ compiler work in the active environment. The distributed source includes both `.h` and `.cuh` headers. An old environment with a broken Ninja Python entry point can put its existing `ninja/data/bin` directory first in PATH. C++ runtime ABI problems must be resolved in that environment, not by silently replacing RoI pooling with another operation. Set `TORCH_EXTENSIONS_DIR` to a writable per-project directory and `MAX_JOBS=1` when host memory is limited. These are environment settings, not model changes.

## Validated server runtime (not a clean-install guarantee)

The short integration checks used the following existing packages. Install a
compatible CUDA PyTorch build first; do not blindly upgrade the tracker runtime
to satisfy a newer MLLM/transformers environment. The CPU CLIP encoder can run
in a separate environment via `--clip-python`. The MLLM server is an external
service, not a dependency installed in the tracker process.

```json
{
  "torch": "1.11.0",
  "torchvision": "0.12.0",
  "timm": "0.5.4",
  "einops": "0.4.1",
  "easydict": "1.9",
  "numpy": "1.21.2",
  "scipy": "1.7.3",
  "Pillow": "9.0.1",
  "opencv-python": "4.5.5.64",
  "matplotlib": "3.5.1",
  "tensorboard": null,
  "tensorboardX": "2.5.1",
  "visdom": "0.1.8.9",
  "ftfy": null,
  "wcwidth": "0.2.5",
  "regex": "2022.3.15",
  "transformers": null
}
```

Entries shown as null were not installed as that distribution name; they are
not a suggested version. This snapshot does not replace a clean-machine test.

The integration checks used Python 3.9, Torch 1.11, torchvision 0.12, CUDA 11.3
and timm 0.5.4. The pip requirements cap NumPy below 1.24 for legacy upstream
aliases. The default transformers range targets this
older tracker environment; alternatively point `--clip-python` at a separate
modern CPU encoder environment, as in the actual checks. Dependency resolution
for a fresh installation has not been executed on the shared server.
