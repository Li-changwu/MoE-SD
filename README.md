# vllm-moe-sd-scheduler


<!-- AUTO_DASHBOARD_START -->
## Optimization Dashboard Snapshot

Updated: 2026-03-16 20:02 UTC  
Dashboard HTML: [docs/dashboard/optimization_dashboard.html](docs/dashboard/optimization_dashboard.html)

### A. 当前最优结果卡片
- **Best TTFT**: AUTO-CD38E89FCC | model=Qwen3-30B-A3B-Instruct-2507 | ttft_p95_ms=15475.566118224524 | delta=n/a | method=no_sd | policy=baseline
- **Best TPOT**: AUTO-CD38E89FCC | model=Qwen3-30B-A3B-Instruct-2507 | tpot_p95_ms=788.2997422005117 | delta=n/a | method=no_sd | policy=baseline
- **Best Throughput**: AUTO-CD38E89FCC | model=Qwen3-30B-A3B-Instruct-2507 | throughput_tok_per_s=8.174784865868432 | delta=n/a | method=no_sd | policy=baseline
- **Best Goodput**: n/a

### C. 实验结论分布
- neutral: 2

### D. 模块贡献分布
- baseline/eagle3: 1
- baseline/no_sd: 1

### E. 当前推荐配置
- model: Qwen3-30B-A3B-Instruct-2507
- config: eagle3 + baseline
- workload_scope: qps1.0_n8
- risk: n/a
- score_main: 0.0
- is_merge_candidate: false

### F. 当前主要问题
1. Long prompt workload 下 TTFT p95 偏高

### 最近实验台账

| experiment_id | date | model | method | module | workload | ttft_p95_ms | tpot_p95_ms | throughput | goodput | result_label | score_main | merge_candidate |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| AUTO-CD38E89FCC | 2026-03-16 | Qwen3-30B-A3B-Instruct-2507 | no_sd | baseline/no_sd | qps1.0_n8 | 15475.566118224524 | 788.2997422005117 | 8.174784865868432 |  | neutral | 0.0 | false |
| AUTO-7DB5BD009B | 2026-03-16 | Qwen3-30B-A3B-Instruct-2507 | eagle3 | baseline/eagle3 | qps1.0_n8 | 18305.469982007053 | 992.7760795670902 | 6.042344974098638 |  | neutral | 0.0 | false |
<!-- AUTO_DASHBOARD_END -->

A plugin-style scheduling framework for memory-constrained MoE inference with speculative decoding on top of vLLM.

## What this project studies

This project targets the following problem:

> Under limited GPU memory, how should we jointly schedule phase behavior (prefill/decode), expert-related behavior, speculative decoding behavior, and runtime memory partitioning to improve throughput while reducing TTFT and TPOT?

Our initial target stack is:

- **Target model**: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Speculative decoding base**: vLLM native **EAGLE-3**
- **Benchmark tool**: `vllm bench`
- **Engineering goal**: plugin-style package that can be enabled on vLLM / vLLM-speculators with minimal intrusion

## Repository layout

- `adapters/`: version adapters for vLLM / speculators
- `controllers/`: governor, phase-aware control, memory partition, prefetch logic
- `collectors/`: acceptance, MoE trace, memory trace collectors
- `configs/`: models, workloads, policies, experiment configs
- `scripts/`: shell entrypoints
- `tools/`: parsing and plotting tools
- `results/`: raw / parsed / figure outputs
- `docs/`: design and reproducibility notes

## Quick start

### 1. Initialize repository layout

```bash
make init-layout
```

### 2. Install package

```bash
make install-dev
```

### 3. Bootstrap reproducible environment

```bash
make bootstrap-env
```

This will create `.venv`, install dependencies, and write `docs/env_report.txt`.

### 4. Start vLLM server without speculative decoding

```bash
make run-server-no-sd
```

### 5. Benchmark no-SD baseline

```bash
make run-baseline-no-sd
```

### 6. Start vLLM server with EAGLE-3

```bash
make run-server-eagle3
```

### 7. Benchmark EAGLE-3 baseline

```bash
make run-baseline-eagle3
```

### 8. Run benchmark from config

```bash
make run-bench CONFIG=configs/experiments/baseline_no_sd.yaml
```

### 9. Build experiment decision board

```bash
# initialize registry once
make init-registry

# create one experiment directory
make scaffold-exp EXP_ID=EXP-20260316-001 OWNER=sage ISSUE_ID=MOESD-001

# after filling summary/compare files, upsert into registry
make append-exp EXP_DIR=results/experiments/EXP-20260316-001

# generate overview dashboard + regression table
make dashboard-build
```

See docs/experiment_decision_board.md for full schema and policy.

### 10. Refresh dashboard on README

```bash
# local refresh: regenerate dashboard + update README snapshot block
make dashboard-readme

# one-command refresh: parse raw bench results, sync registry, rebuild dashboard
make dashboard-refresh
```

### 11. Run observability collectors

```bash
make collect-acceptance ACCEPT_TRACE=results/raw/trace/acceptance.jsonl
make collect-moe-trace MOE_TRACE=results/raw/trace/moe_trace.jsonl
make analyze-memory MEMORY_SNAPSHOTS=results/raw/trace/memory.jsonl
```

The repository also includes an auto-refresh workflow:

- .github/workflows/update-dashboard-readme.yml

It updates README on:

- push to experiment registry or experiment outputs
- manual workflow dispatch
- schedule (every 30 minutes)

## Reproducibility Docs

- `docs/version_matrix.md`: frozen environment matrix
- `docs/workload_matrix.md`: frozen workload profiles
- `docs/benchmark_contract.md`: benchmark metric/output contract
- `docs/make_targets.md`: standardized make entrypoints
- `docs/result_naming_convention.md`: result path + metadata contract
- `docs/issue_execution_plan.md`: issue execution waves and policy
- `docs/acceptance_report.md`: acceptance collector output contract
- `docs/moe_trace_report.md`: MoE trace collector output contract
- `docs/memory_breakdown.md`: memory decomposition output contract
- `docs/integration_boundary.md`: plugin/adapter/patch boundary and rollout rules
- `docs/config_protocol.md`: scheduler config and feature-flag contract
- `docs/controller_interface.md`: frozen scheduler controller interface
- `docs/state_schema.yaml`: request/phase/step state schema
- `docs/decision_trace_schema.yaml`: decision trace schema

## Package Runtime Entry

After `make install-dev`, you can inspect runtime mode quickly:

```bash
moe-sd-runtime --config-json '{"model":"Qwen/Qwen3-30B-A3B-Instruct-2507","workload_profile":"online_medium","feature_flags":{"enable_controller":false}}'
```

## Research roadmap

The project is organized around four milestones:

- **M0**: environment and native baselines
- **M1**: observability (acceptance, MoE expert trace, memory breakdown)
- **M2**: plugin-style integration and unified scheduler interface
- **M3**: core mechanisms
  - static governor
  - phase-aware governor
  - acceptance-aware expert prefetch
  - dynamic memory partition
- **M4**: main experiments, ablations, artifact packaging

## Design principles

1. **Benchmark-first**
   Every major change must be validated through `vllm bench`.

2. **Observability before control**
   We first measure acceptance dynamics, expert reuse, and memory pressure before implementing advanced scheduling logic.

3. **Minimal-intrusion integration**
   The project aims for a package-style integration with vLLM, using adapters and controlled hooks instead of large invasive patches whenever possible.

4. **Reproducibility**
   Every experiment should be tied to:
   - model
   - config
   - workload profile
   - git commit
   - result directory

## Immediate TODO

- [ ] Freeze environment versions
- [ ] Reproduce native no-SD baseline
- [ ] Reproduce native EAGLE-3 baseline
- [ ] Implement acceptance collector
- [ ] Implement Qwen3 MoE expert trace collector
- [ ] Implement memory breakdown analyzer
- [ ] Freeze scheduler interface

## Notes

This repository is under active development. Early versions prioritize:

- stable baselines
- unified benchmarking
- runtime observability

over aggressive optimization.
