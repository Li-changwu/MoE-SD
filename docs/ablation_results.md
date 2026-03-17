# Ablation Results

Ablation outputs are grouped by optimization module and method.

## Artifacts

- `results/ablation/ablation.csv`
- `results/ablation_figures/ablation_ttft_p95.png`
- `results/ablation_figures/ablation_throughput.png`

## Reproduction

```bash
bash scripts/reproduce_ablation.sh
```

## Required Ablations (target)

- remove phase-aware
- remove acceptance-aware prefetch
- remove dynamic memory partition
- fixed K
- fixed memory ratio

Current output reflects available rows in registry.
