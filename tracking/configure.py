"""Create a local predictor config with repository-relative paths where possible."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config.paths import REPOSITORY, portable_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='configs/local.json')
    args = parser.parse_args()
    if not Path(args.data_root).expanduser().is_dir():
        parser.error('data-root must exist')
    if not Path(args.checkpoint).expanduser().is_file():
        parser.error('checkpoint must exist')
    keys = ('data_root', 'output_root', 'checkpoint')
    config = {key: portable_path(getattr(args, key)) for key in keys}
    config.update(path_base=REPOSITORY, path_map={})
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(dest)


if __name__ == '__main__':
    main()
