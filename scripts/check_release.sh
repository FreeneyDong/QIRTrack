#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tracking/preflight.py --offline
python -m unittest discover -s tests -v
python tracking/check_checkpoint.py
