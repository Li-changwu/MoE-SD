#!/usr/bin/env bash
set -euo pipefail

make dashboard-refresh
python3 tools/build_ablation_results.py \
  --registry-csv results/registry/experiment_registry.csv \
  --out-table results/ablation/ablation.csv \
  --out-fig-dir results/ablation_figures

echo "Ablation results reproduction done."
