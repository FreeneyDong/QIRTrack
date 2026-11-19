from pathlib import Path
import json
import os

DEFAULT_ROOT = Path(__file__).resolve().parents[4] / 'prompts'
ROOT = Path(os.environ.get('QIRTRACK_PROMPT_DIR', DEFAULT_ROOT))
EXPECTED_KEYS = tuple(json.loads((DEFAULT_ROOT / 'index.json').read_text(encoding='utf-8')))

def prompt_text(key, required=False):
    if key not in EXPECTED_KEYS:
        raise KeyError('Unknown prompt key: ' + str(key))
    p = ROOT / (key + '.txt')
    if not p.is_file() or not p.read_text(encoding='utf-8').strip():
        if required:
            raise RuntimeError('MLLM prompt withheld or missing: ' + key)
        return ''  # Permit offline imports; every HTTP entry is guarded below.
    return p.read_text(encoding='utf-8')

def require_prompts():
    missing = [key for key in EXPECTED_KEYS if not (ROOT / (key + '.txt')).is_file()
               or not (ROOT / (key + '.txt')).read_text(encoding='utf-8').strip()]
    if not EXPECTED_KEYS or missing:
        raise RuntimeError('MLLM prompts intentionally withheld; populate all required files before inference: ' + ', '.join(missing))

def guarded_urlopen(*args, **kwargs):
    require_prompts()
    from urllib.request import urlopen
    return urlopen(*args, **kwargs)
