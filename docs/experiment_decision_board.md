# Experiment Decision Board

This document defines the first implementation of an experiment decision board:

- 1 overview page: Optimization Dashboard
- 1 experiment registry table: experiment_registry.csv
- 1 regression/stability table: regression_stability.csv

## Standard experiment output layout

Each experiment should use the following structure:

results/experiments/EXP-YYYYMMDD-XXX/
  meta.json
  bench_raw.json
  summary.json
  trace_summary.json
  compare_to_baseline.json
  plots/
  notes.md

## Registry fields

The registry header is frozen in:

- results/registry/experiment_registry.csv

Important fields:

- change linkage: issue_id, branch, commit_hash, change_summary, hypothesis
- reproducibility: workload_profile, hardware_profile, seed
- baseline linkage: baseline_experiment_id, baseline_type, comparison_scope
- decision fields: result_label, primary_gain, primary_cost, final_conclusion
- auto fields: score_main, is_merge_candidate

## Label policy

Allowed labels:

- win
- partial_win
- neutral
- regression
- invalid

## score_main

The default score is:

score_main = 0.25 * (-delta_ttft_p95_pct)
           + 0.35 * (-delta_tpot_p95_pct)
           + 0.20 * (delta_throughput_pct)
           + 0.15 * (delta_goodput_pct)
           - 0.05 * (delta_gpu_mem_peak_pct)

You can override weights through CLI arguments.

## is_merge_candidate

Default decision policy in the utility script:

- win => true
- partial_win => true only when:
  - oom_count == 0
  - fallback_count == 0
  - delta_ttft_p95_pct <= 2
  - delta_gpu_mem_peak_pct <= 8
- otherwise => false

## CLI workflow

1) Initialize registry

python tools/experiment_registry.py init

2) Scaffold one experiment directory

python tools/experiment_registry.py scaffold --experiment-id EXP-20260316-001 --owner your_name --issue-id MOESD-001

3) Fill meta/summary/compare files for that experiment

4) Append or upsert this experiment into registry

python tools/experiment_registry.py append --exp-dir results/experiments/EXP-20260316-001

5) Build overview dashboard and regression table

python tools/build_decision_dashboard.py

Output files:

- results/figures/optimization_dashboard.html
- results/parsed/regression_stability.csv
