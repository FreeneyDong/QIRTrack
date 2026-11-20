"""Create a local online config with repository-relative paths where possible."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.config.paths import REPOSITORY, portable_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracker', choices=['artrackv2', 'ostrack', 'sutrack'], required=True)
    p.add_argument('--domain', default='lasot')
    p.add_argument('--native-checkpoint', required=True)
    p.add_argument('--predictor', required=True)
    p.add_argument('--clip', required=True, help='Local Hugging Face CLIP ViT-B/32 directory')
    p.add_argument('--clip-python', default=sys.executable)
    p.add_argument('--prompts', required=True)
    p.add_argument('--mllm-url', required=True, help='OpenAI-compatible chat/completions endpoint')
    p.add_argument('--mllm-model', required=True)
    p.add_argument('--q', required=True, type=float)
    p.add_argument('--p-high', default=.7, type=float)
    p.add_argument('--q-trusted', default=.8, type=float)
    p.add_argument('--out', default='configs/local_online.json')
    a = p.parse_args()
    if not 0 < a.q < a.q_trusted < 1 or not .5 < a.p_high < 1:
        p.error('Require 0 < q < q-trusted < 1 and .5 < p-high < 1')
    configs = {
        'artrackv2': 'artrackv2/artrackv2_seq/artrackv2_seq_256_got.yaml',
        'ostrack': 'ostrack/ostrack/vitb_384_mae_ce_32x4_ep300.yaml',
        'sutrack': 'sutrack/sutrack/sutrack_l384.yaml',
    }
    c = dict(tracker=a.tracker, domain=a.domain, q=a.q, p_high=a.p_high,
             mllm_url=a.mllm_url, mllm_model=a.mllm_model,
             native_config=(Path('experiments') / configs[a.tracker]).as_posix(),
             clip_python=portable_path(a.clip_python), path_base=REPOSITORY,
             framework={'q_trusted': a.q_trusted})
    for name in ('native_checkpoint', 'predictor', 'clip', 'prompts'):
        value = Path(getattr(a, name)).expanduser().resolve()
        if not value.exists():
            p.error(name + ' does not exist: ' + str(value))
        c[name] = portable_path(value)
    clip = Path(a.clip).expanduser()
    model_files = sorted(list(clip.glob('*.safetensors')) or list(clip.glob('pytorch_model*.bin')))
    if not model_files:
        p.error('No CLIP model weights found')
    h = hashlib.sha256()
    for file in model_files:
        with file.open('rb') as f:
            for block in iter(lambda: f.read(4194304), b''):
                h.update(block)
    c['clip_sha256'] = h.hexdigest()
    c['clip_weight_files'] = [x.name for x in model_files]
    if a.tracker == 'artrackv2':
        c['recent_policy'] = dict(score_threshold=.9159601032733917,
            entropy_threshold=.26971155405044556, minimum_age=8, maximum_age=128, trusted=False)
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(c, f, indent=2)
    print(target)


if __name__ == '__main__':
    main()
