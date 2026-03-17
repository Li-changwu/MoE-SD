# Main Results

Main comparison outputs are generated from registry entries.

## Artifacts

- `results/main_table/main_comparison.csv`
- `results/main_figures/main_ttft_p95.png`
- `results/main_figures/main_throughput.png`

## Reproduction

```bash
bash scripts/reproduce_main_results.sh
```

## Comparison Set (target)

- no-SD
- native EAGLE-3
- static governor v0
- phase-aware governor v1
- phase-aware + acceptance-aware prefetch
- full system

Current repository may contain a subset depending on available benchmark runs.
