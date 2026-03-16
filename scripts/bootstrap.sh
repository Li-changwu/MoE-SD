#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

bash scripts/env_report.sh

echo "Bootstrap completed. Activate with: source .venv/bin/activate"
