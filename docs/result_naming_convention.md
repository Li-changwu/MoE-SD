# Result Naming Convention

All benchmark outputs must follow:

`results/raw/<method>/<workload_profile>/<mode>/<timestamp>_<config_hash>/`

## Required Metadata Fields

Every run must save `metadata.json` with:

- `model`
- `method`
- `workload_profile`
- `mode`
- `seed`
- `git_commit`
- `config_hash`
- `prompt_len`
- `output_len`
- `request_rate`
- `num_prompts`

## Directory Semantics

- `results/raw/`: raw benchmark JSON and metadata
- `results/parsed/`: normalized summary tables
- `results/figures/`: plots and dashboard artifacts

## Valid Example

`results/raw/eagle3/online_medium/serve/20260316T190500Z_8d2b1a7f2c/`

## Invalid Examples

- `results/raw/tmp-run/`
- `results/raw/serve.json`
- `results/raw/20260316/`

## Contract

- Parsing scripts must not guess metadata from filenames alone.
- Metadata is canonical; path acts as redundancy for fast triage.
- Each result must be traceable to config and git commit.
