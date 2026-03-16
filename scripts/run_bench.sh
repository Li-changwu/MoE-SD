#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config.yaml>"
  exit 1
fi

python3 tools/bench_runner.py --config "$1"
python3 tools/parse_bench_results.py --input-dir results/raw --output-dir results/parsed
