# Memory Breakdown

This document tracks memory usage decomposition across methods and K values.

## Decomposition Terms

- `target_mb`
- `draft_mb`
- `kv_mb`
- `temp_buffers_mb`
- `spec_metadata_mb`

## Output Artifacts

- `results/memory_breakdown/memory_breakdown_by_workload.csv`
- `results/memory_breakdown/memory_breakdown_by_method.csv`
- `results/memory_breakdown/memory_delta_no_sd_vs_eagle3.csv`
- `results/memory_breakdown/memory_breakdown_stacked.png`

## How To Run

```bash
python tools/memory_breakdown.py \
  --snapshots <memory_snapshots.jsonl> \
  --output-dir results/memory_breakdown
```

## Required Snapshot Fields

- `method` (`no_sd`, `eagle3`, ...)
- `k`
- `workload_profile`
- component columns listed above

## Comparison Goals

- compare no-SD vs EAGLE-3 memory footprint
- compare memory cost under different K
- produce a stable table for controller policy design
