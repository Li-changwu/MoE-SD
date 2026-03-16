#!/usr/bin/env bash
set -euo pipefail

python3 tools/bench_runner.py --config configs/experiments/baseline_eagle3.yaml
python3 tools/parse_bench_results.py --input-dir results/raw --output-dir results/parsed
