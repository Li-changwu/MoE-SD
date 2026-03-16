# Benchmark Contract

All performance validation must use `vllm bench`.

Primary metrics:
- TTFT
- TPOT
- ITL
- throughput
- goodput

Every result must record:
- model
- benchmark mode
- workload profile
- request rate
- input length
- output length
- seed
- git commit

## Canonical Entry Points

- `make run-bench CONFIG=<config>`
- `make run-baseline-no-sd`
- `make run-baseline-eagle3`

Do not use ad-hoc one-off commands for reportable results.

## Required Output Layout

Raw output directory:

`results/raw/<method>/<workload_profile>/<mode>/<timestamp>_<config_hash>/`

Every run directory must include:

- raw benchmark json from `vllm bench`
- `metadata.json` with config and git context

Parsed summary:

- `results/parsed/summary.csv`

## Repetition Policy

- Each reported setting should run at least 3 times.
- Use the same seed set for all compared methods.
- Store each repetition as a distinct timestamped directory.
