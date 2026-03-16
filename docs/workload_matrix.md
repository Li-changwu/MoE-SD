# Workload Matrix

This file freezes workload profiles used by benchmarks and paper plots.

## Naming Rule

Use `<mode>-<prompt_bucket>-<output_bucket>-<pressure>-<sampling>`.

Examples:
- `online-short-short-loose-deterministic`
- `online-medium-medium-medium-deterministic`
- `online-long-medium-tight-sampled`

## Frozen Profiles (v1)

| Profile | Modes | Prompt | Output | Rate | Memory Pressure | Sampling | K |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| online_short | serve, latency | 128 | 64 | 1 | loose | deterministic | 1/2/4/8 |
| online_medium | serve, latency, throughput | 512 | 128 | 2 | medium | deterministic | 1/2/4/8 |
| online_long | serve, latency | 2048 | 256 | 1 | tight | sampled | 1/2/4 |

Profiles are stored under `configs/workloads/`.
All experiment outputs must include `workload_profile` in metadata.
