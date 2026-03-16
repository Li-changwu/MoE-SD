SHELL := /bin/bash

PYTHON ?= python3
PIP ?= pip3

MODEL ?= Qwen/Qwen3-30B-A3B-Instruct-2507
HOST ?= 127.0.0.1
PORT ?= 8000
MAX_MODEL_LEN ?= 4096
GPU_MEMORY_UTILIZATION ?= 0.9
SERVER_EXTRA_ARGS ?=
ENABLE_OFFLOAD ?= 1
CPU_OFFLOAD_GB ?= 18
SWAP_SPACE_GB ?= 32

ifeq ($(ENABLE_OFFLOAD),1)
OFFLOAD_ARGS := --cpu-offload-gb $(CPU_OFFLOAD_GB) --swap-space $(SWAP_SPACE_GB)
else
OFFLOAD_ARGS :=
endif

VLLM_API_BASE ?= http://$(HOST):$(PORT)/v1
RESULTS_DIR ?= results
CONFIG_DIR ?= configs

# default benchmark config
PROMPT_LEN ?= 512
OUTPUT_LEN ?= 128
NUM_PROMPTS ?= 32
REQUEST_RATE ?= 2
SEED ?= 42

# spec decode config
SPEC_METHOD ?= eagle3
SPEC_TOKENS ?= 4
SPEC_MODEL ?= /home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3

SPEC_CONFIG := {"method":"$(SPEC_METHOD)","num_speculative_tokens":$(SPEC_TOKENS),"model":"$(SPEC_MODEL)"}

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  init-layout           Initialize repository layout"
	@echo "  install-dev           Install editable package"
	@echo "  bootstrap-env         Create venv, install dependencies, dump env report"
	@echo "  bootstrap-issues      Create GitHub labels/milestones/issues"
	@echo "  run-server-no-sd      Start vLLM server without speculative decoding"
	@echo "  run-server-eagle3     Start vLLM server with EAGLE-3"
	@echo "  run-baseline-no-sd    Run no-SD baseline chain"
	@echo "  run-baseline-eagle3   Run EAGLE-3 baseline chain"
	@echo "  run-bench             Run benchmark from CONFIG file"
	@echo "  bench-serve-no-sd     Run vllm bench serve for no-SD baseline"
	@echo "  bench-serve-eagle3    Run vllm bench serve for EAGLE-3 baseline"
	@echo "  bench-latency-no-sd   Run vllm bench latency for no-SD baseline"
	@echo "  bench-latency-eagle3  Run vllm bench latency for EAGLE-3 baseline"
	@echo "  bench-throughput-no-sd"
	@echo "  bench-throughput-eagle3"
	@echo "  parse-results         Parse raw benchmark results"
	@echo "  collect-acceptance    Build acceptance metrics from trace"
	@echo "  collect-moe-trace     Build MoE trace metrics from trace"
	@echo "  analyze-memory        Build memory breakdown tables and chart"
	@echo "  init-registry         Initialize experiment registry"
	@echo "  scaffold-exp          Create one standard experiment directory"
	@echo "  append-exp            Upsert one experiment into registry"
	@echo "  sync-registry         Sync registry from parsed summary.csv"
	@echo "  dashboard-build       Build optimization dashboard and regression table"
	@echo "  dashboard-readme      Refresh README dashboard snapshot"
	@echo "  dashboard-refresh     Parse raw results + sync registry + refresh dashboard"
	@echo "  clean-results         Remove generated results"

.PHONY: init-layout
init-layout:
	bash scripts/init_repo_layout.sh

.PHONY: install-dev
install-dev:
	$(PIP) install -e .

.PHONY: bootstrap-env
bootstrap-env:
	bash scripts/bootstrap.sh

.PHONY: bootstrap-issues
bootstrap-issues:
	bash scripts/bootstrap_github_issues.sh

.PHONY: run-server-no-sd
run-server-no-sd:
	vllm serve $(MODEL) \
		--host $(HOST) \
		--port $(PORT) \
		--max-model-len $(MAX_MODEL_LEN) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		$(OFFLOAD_ARGS) \
		$(SERVER_EXTRA_ARGS)

.PHONY: run-server-eagle3
run-server-eagle3:
	vllm serve $(MODEL) \
		--host $(HOST) \
		--port $(PORT) \
		--max-model-len $(MAX_MODEL_LEN) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		$(OFFLOAD_ARGS) \
		--speculative_config '$(SPEC_CONFIG)' \
		$(SERVER_EXTRA_ARGS)

.PHONY: bench-serve-no-sd
bench-serve-no-sd:
	mkdir -p $(RESULTS_DIR)/raw/no_sd
	vllm bench serve \
		--model $(MODEL) \
		--backend openai-chat \
		--base-url $(VLLM_API_BASE) \
		--endpoint /chat/completions \
		--num-prompts $(NUM_PROMPTS) \
		--random-input-len $(PROMPT_LEN) \
		--random-output-len $(OUTPUT_LEN) \
		--request-rate $(REQUEST_RATE) \
		--seed $(SEED) \
		--save-result \
		--result-dir $(RESULTS_DIR)/raw/no_sd

.PHONY: bench-serve-eagle3
bench-serve-eagle3:
	mkdir -p $(RESULTS_DIR)/raw/eagle3
	vllm bench serve \
		--model $(MODEL) \
		--backend openai-chat \
		--base-url $(VLLM_API_BASE) \
		--endpoint /chat/completions \
		--num-prompts $(NUM_PROMPTS) \
		--random-input-len $(PROMPT_LEN) \
		--random-output-len $(OUTPUT_LEN) \
		--request-rate $(REQUEST_RATE) \
		--seed $(SEED) \
		--save-result \
		--result-dir $(RESULTS_DIR)/raw/eagle3

.PHONY: bench-latency-no-sd
bench-latency-no-sd:
	mkdir -p $(RESULTS_DIR)/raw/no_sd_latency
	vllm bench latency \
		--model $(MODEL) \
		--input-len $(PROMPT_LEN) \
		--output-len $(OUTPUT_LEN) \
		--save-result \
		--result-dir $(RESULTS_DIR)/raw/no_sd_latency

.PHONY: bench-latency-eagle3
bench-latency-eagle3:
	mkdir -p $(RESULTS_DIR)/raw/eagle3_latency
	vllm bench latency \
		--model $(MODEL) \
		--input-len $(PROMPT_LEN) \
		--output-len $(OUTPUT_LEN) \
		--save-result \
		--result-dir $(RESULTS_DIR)/raw/eagle3_latency

.PHONY: bench-throughput-no-sd
bench-throughput-no-sd:
	mkdir -p $(RESULTS_DIR)/raw/no_sd_throughput
	vllm bench throughput \
		--model $(MODEL) \
		--random-input-len $(PROMPT_LEN) \
		--random-output-len $(OUTPUT_LEN) \
		--num-prompts $(NUM_PROMPTS) \
		--save-result \
		--result-dir $(RESULTS_DIR)/raw/no_sd_throughput

.PHONY: bench-throughput-eagle3
bench-throughput-eagle3:
	mkdir -p $(RESULTS_DIR)/raw/eagle3_throughput
	vllm bench throughput \
		--model $(MODEL) \
		--random-input-len $(PROMPT_LEN) \
		--random-output-len $(OUTPUT_LEN) \
		--num-prompts $(NUM_PROMPTS) \
		--save-result \
		--result-dir $(RESULTS_DIR)/raw/eagle3_throughput

CONFIG ?=

.PHONY: run-bench
run-bench:
	@if [ -z "$(CONFIG)" ]; then echo "CONFIG is required, e.g. make run-bench CONFIG=configs/experiments/baseline_no_sd.yaml"; exit 1; fi
	bash scripts/run_bench.sh "$(CONFIG)"

.PHONY: run-baseline-no-sd
run-baseline-no-sd:
	bash scripts/run_baseline_no_sd.sh

.PHONY: run-baseline-eagle3
run-baseline-eagle3:
	bash scripts/run_baseline_eagle3.sh

.PHONY: parse-results
parse-results:
	$(PYTHON) tools/parse_bench_results.py --input-dir $(RESULTS_DIR)/raw --output-dir $(RESULTS_DIR)/parsed

ACCEPT_TRACE ?=
MOE_TRACE ?=
MEMORY_SNAPSHOTS ?=
BENCH_SUMMARY ?=$(RESULTS_DIR)/parsed/summary.csv

.PHONY: collect-acceptance
collect-acceptance:
	@if [ -z "$(ACCEPT_TRACE)" ]; then echo "ACCEPT_TRACE is required, e.g. make collect-acceptance ACCEPT_TRACE=results/raw/trace/acceptance.jsonl"; exit 1; fi
	$(PYTHON) collectors/acceptance_collector.py --trace "$(ACCEPT_TRACE)" --bench-summary "$(BENCH_SUMMARY)" --output-dir $(RESULTS_DIR)/acceptance

.PHONY: collect-moe-trace
collect-moe-trace:
	@if [ -z "$(MOE_TRACE)" ]; then echo "MOE_TRACE is required, e.g. make collect-moe-trace MOE_TRACE=results/raw/trace/moe_trace.jsonl"; exit 1; fi
	$(PYTHON) collectors/moe_trace_collector.py --trace "$(MOE_TRACE)" --output-dir $(RESULTS_DIR)/moe_trace

.PHONY: analyze-memory
analyze-memory:
	@if [ -z "$(MEMORY_SNAPSHOTS)" ]; then echo "MEMORY_SNAPSHOTS is required, e.g. make analyze-memory MEMORY_SNAPSHOTS=results/raw/trace/memory.jsonl"; exit 1; fi
	$(PYTHON) tools/memory_breakdown.py --snapshots "$(MEMORY_SNAPSHOTS)" --output-dir $(RESULTS_DIR)/memory_breakdown

EXP_ID ?=
EXP_DIR ?=
OWNER ?=
ISSUE_ID ?=

.PHONY: init-registry
init-registry:
	$(PYTHON) tools/experiment_registry.py init --registry-csv $(RESULTS_DIR)/registry/experiment_registry.csv

.PHONY: scaffold-exp
scaffold-exp:
	@if [ -z "$(EXP_ID)" ]; then echo "EXP_ID is required, e.g. make scaffold-exp EXP_ID=EXP-20260316-001"; exit 1; fi
	$(PYTHON) tools/experiment_registry.py scaffold \
		--experiment-id $(EXP_ID) \
		--results-root $(RESULTS_DIR)/experiments \
		--owner "$(OWNER)" \
		--issue-id "$(ISSUE_ID)"

.PHONY: append-exp
append-exp:
	@if [ -z "$(EXP_DIR)" ]; then echo "EXP_DIR is required, e.g. make append-exp EXP_DIR=results/experiments/EXP-20260316-001"; exit 1; fi
	$(PYTHON) tools/experiment_registry.py append \
		--exp-dir $(EXP_DIR) \
		--registry-csv $(RESULTS_DIR)/registry/experiment_registry.csv

.PHONY: sync-registry
sync-registry:
	$(PYTHON) tools/sync_registry_from_summary.py \
		--summary-csv $(RESULTS_DIR)/parsed/summary.csv \
		--registry-csv $(RESULTS_DIR)/registry/experiment_registry.csv

.PHONY: dashboard-build
dashboard-build:
	$(PYTHON) tools/build_decision_dashboard.py \
		--registry-csv $(RESULTS_DIR)/registry/experiment_registry.csv \
		--output-html $(RESULTS_DIR)/figures/optimization_dashboard.html \
		--regression-csv $(RESULTS_DIR)/parsed/regression_stability.csv

.PHONY: build-dashboard
build-dashboard: dashboard-build

.PHONY: dashboard-readme
dashboard-readme:
	$(PYTHON) tools/build_decision_dashboard.py \
		--registry-csv $(RESULTS_DIR)/registry/experiment_registry.csv \
		--output-html docs/dashboard/optimization_dashboard.html \
		--regression-csv $(RESULTS_DIR)/parsed/regression_stability.csv
	$(PYTHON) tools/update_readme_dashboard.py \
		--registry-csv $(RESULTS_DIR)/registry/experiment_registry.csv \
		--readme README.md \
		--dashboard-rel-path docs/dashboard/optimization_dashboard.html

.PHONY: update-readme-dashboard
update-readme-dashboard: dashboard-readme

.PHONY: dashboard-refresh
dashboard-refresh: parse-results sync-registry dashboard-readme

.PHONY: clean-results
clean-results:
	rm -rf $(RESULTS_DIR)/raw/*
	rm -rf $(RESULTS_DIR)/parsed/*
	rm -rf $(RESULTS_DIR)/figures/*
