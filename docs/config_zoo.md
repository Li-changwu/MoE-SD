# Config Zoo

## Experiment configs

- `configs/experiments/baseline_no_sd.yaml`
- `configs/experiments/baseline_eagle3.yaml`

## Workload configs

- `configs/workloads/online_short.yaml`
- `configs/workloads/online_medium.yaml`
- `configs/workloads/online_long.yaml`

## Policy configs

- `configs/policies/static_v0.yaml`
- `configs/policies/phase_aware_v1.yaml`
- `configs/policies/prefetch_v1.yaml`
- `configs/policies/memory_partition_v2.yaml`

## How to run config-driven bench

```bash
make run-bench CONFIG=configs/experiments/baseline_no_sd.yaml
```
