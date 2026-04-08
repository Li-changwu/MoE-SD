#!/usr/bin/env bash
# Cross-Architecture Benchmark: GPT-OSS-120B with BriskMoE
# Tests BriskMoE on a non-Qwen architecture to validate generalization.
#
# Configurations:
#   1. AR + ELMM (baseline): autoregressive with expert offloading
#   2. SD + ELMM: speculative decoding (Eagle3) with expert offloading
#   3. SD + ELMM + DIPP + dirty_flag: full BriskMoE
#
# Note: Vanilla vLLM (no ELMM) cannot run GPT-OSS-120B on A6000 due to
# Marlin MXFP4 conversion requiring ~40 GB GPU memory for model weights.

set -euo pipefail

MODEL="/root/models/gpt-oss-120b"
SPECULATOR="/root/models/gpt-oss-120b-speculator.eagle3"
DATASET="/root/MoE-SD/data/humaneval_bench.jsonl"
RUNNER="/root/MoE-SD/scripts/bench_humaneval_runner.py"
RESULTS_DIR="/root/MoE-SD/results/gptoss_cross_arch"
PYTHON="/opt/miniconda3/envs/moe-sd/bin/python"

NUM_PROMPTS=50
WARMUP=5
OUTPUT_LEN=128
MAX_MODEL_LEN=2048
CPU_OFFLOAD=58
GPU_UTIL=0.85

SPEC_CONFIG='{"method":"eagle3","model":"'"${SPECULATOR}"'","num_speculative_tokens":3}'

mkdir -p "${RESULTS_DIR}"

run_bench() {
    local config_name="$1"
    local output_json="${RESULTS_DIR}/${config_name}/result.json"
    mkdir -p "$(dirname "$output_json")"

    echo ""
    echo "============================================================"
    echo "  Config: ${config_name}"
    echo "  Output: ${output_json}"
    echo "============================================================"

    shift  # Remove config_name from args, rest are env vars and extra args
    local extra_args=()
    local env_prefix=""

    # Parse env vars and extra runner args from remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --spec) extra_args+=(--speculative-config "${SPEC_CONFIG}"); shift ;;
            ENV_*) env_prefix="${env_prefix} ${1#ENV_}"; shift ;;
            *) extra_args+=("$1"); shift ;;
        esac
    done

    # Build command
    local cmd="${PYTHON} ${RUNNER} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --output-len ${OUTPUT_LEN} \
        --num-prompts ${NUM_PROMPTS} \
        --warmup-prompts ${WARMUP} \
        --gpu-memory-utilization ${GPU_UTIL} \
        --cpu-offload-gb ${CPU_OFFLOAD} \
        --max-model-len ${MAX_MODEL_LEN} \
        --enforce-eager \
        --trust-remote-code \
        --dtype bfloat16 \
        --output-json ${output_json} \
        ${extra_args[*]:-}"

    # Run with environment variables
    echo "  Running: env ${env_prefix} ${cmd}"
    local log_file="${RESULTS_DIR}/${config_name}/bench.log"
    eval "env ${env_prefix} ${cmd}" 2>&1 | tee "${log_file}"
}

# Common ELMM env vars for all configs (GPT-OSS has no shared experts)
ELMM_BASE="ENV_VLLM_PLUGINS=elmm ENV_ELMM_CACHE_GB=20 ENV_ELMM_POOL_DIRECT=0 ENV_ELMM_SHARED_PARALLEL=0 ENV_ELMM_STALE_REMAP=0"

# --- Config 1: AR + ELMM (baseline) ---
run_bench "ar_elmm" \
    ${ELMM_BASE} \
    ENV_BRISKMOE_DIPP=0 \
    ENV_ELMM_ORACLE_PREFETCH=0

# --- Config 2: SD + ELMM ---
run_bench "sd_elmm" \
    ${ELMM_BASE} \
    ENV_BRISKMOE_DIPP=0 \
    ENV_ELMM_ORACLE_PREFETCH=1 \
    --spec

# --- Config 3: SD + ELMM + DIPP + dirty_flag (full BriskMoE) ---
run_bench "sd_dipp_dirty_flag" \
    ${ELMM_BASE} \
    ENV_BRISKMOE_DIPP=1 \
    ENV_ELMM_ORACLE_PREFETCH=1 \
    --spec

echo ""
echo "============================================================"
echo "  All benchmarks complete! Results in ${RESULTS_DIR}"
echo "============================================================"

# Summary
echo ""
echo "Quick comparison:"
for cfg in ar_elmm sd_elmm sd_dipp_dirty_flag; do
    result="${RESULTS_DIR}/${cfg}/result.json"
    if [[ -f "$result" ]]; then
        tps=$(python3 -c "import json; d=json.load(open('$result')); print(f'{d[\"tps_mean\"]:.2f}')")
        tps_med=$(python3 -c "import json; d=json.load(open('$result')); print(f'{d[\"tps_median\"]:.2f}')")
        echo "  ${cfg}: mean=${tps} tok/s, median=${tps_med} tok/s"
    fi
done
