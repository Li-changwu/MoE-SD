#!/usr/bin/env bash
set -euo pipefail

make dashboard-refresh
python3 tools/build_main_results.py \
  --registry-csv results/registry/experiment_registry.csv \
  --out-table results/main_table/main_comparison.csv \
  --out-fig-dir results/main_figures

echo "Main results reproduction done."
