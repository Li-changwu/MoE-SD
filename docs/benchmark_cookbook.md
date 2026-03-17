# Benchmark Cookbook

## Baseline no-SD

```bash
make run-server-no-sd MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507
make run-baseline-no-sd
```

## Baseline EAGLE-3

```bash
make run-server-eagle3 MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507 SPEC_MODEL=models/eagle3_spec SPEC_METHOD=eagle3 SPEC_TOKENS=2
make run-baseline-eagle3
```

## Parse + Sync + Dashboard

```bash
make dashboard-refresh
```

## Main/Ablation outputs

```bash
make main-results
make ablation-results
```
