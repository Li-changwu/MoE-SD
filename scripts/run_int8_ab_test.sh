#!/usr/bin/env bash
# =============================================================================
# INT8 W8A16 Pool Quantization A/B Test
# =============================================================================
# A: ELMM + TASER (BF16 pool baseline)
# B: ELMM + TASER + INT8 pool (W8A16 quantized)
#
# Runs both configs, measures tok/s, reports speedup.
# =============================================================================
set -euo pipefail

MODEL_DIR="/root/models/Qwen3-30B-A3B-Instruct-2507"
EAGLE_DIR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
CPU_OFFLOAD_GB=30
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}"
NUM_PROMPTS=20
MAX_TOKENS=200
WARMUP_PROMPTS=5
RESULTS_DIR="/root/MoE-SD/results/int8_ab_test"
LOG_DIR="/root/MoE-SD/logs/int8_ab_test"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ----- helpers -----
wait_for_server() {
    local max_wait=600; local waited=0
    echo "  Waiting for server..."
    while ! curl -sf "${BASE_URL}/v1/models" > /dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        [[ "$waited" -ge "$max_wait" ]] && { echo "  ERROR: server timeout"; return 1; }
    done
    echo "  Server ready (${waited}s)"
}

kill_server() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true; sleep 3
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true; sleep 2
}

run_bench() {
    local label="$1"
    echo "  Benchmarking: ${label}"
    conda run --no-capture-output -n moe-sd python /root/MoE-SD/scripts/elmm_bench_worker.py \
        --base-url "$BASE_URL" \
        --model "$MODEL_DIR" \
        --num-prompts "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS" \
        --warmup "$WARMUP_PROMPTS" \
        --output "${RESULTS_DIR}/${label}.jsonl" \
        --summary "${RESULTS_DIR}/${label}_summary.json" \
        2>&1 | tee "${LOG_DIR}/${label}_bench.log"
}

start_server() {
    local label="$1"; shift
    local extra_env="$@"
    echo "  Starting server: ${label}"
    kill_server
    
    local spec_config
    spec_config=$(python3 -c "import json; print(json.dumps({'model':'${EAGLE_DIR}','method':'eagle3','num_speculative_tokens':3,'draft_tensor_parallel_size':1}))")

    # Export env vars so conda run inherits them
    export VLLM_PLUGINS=adapters.vllm_elmm_plugin
    export ELMM_CACHE_GB=8
    export ELMM_STALE_REMAP=4
    export ELMM_STALE_REMAP_WARMUP=32
    export ELMM_STALE_REMAP_MAX_INTERVAL=128
    # Parse and export extra_env (e.g. "ELMM_INT8_POOL=1")
    for kv in $extra_env; do
        export "$kv"
    done

    conda run --no-capture-output -n moe-sd python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --speculative-config "$spec_config" \
        --cpu-offload-gb "$CPU_OFFLOAD_GB" \
        --tensor-parallel-size 1 \
        --port "$PORT" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        > "${LOG_DIR}/${label}_server.log" 2>&1 &
    
    wait_for_server
}

# =============================================================================
# A: BF16 Baseline
# =============================================================================
echo "================================================================"
echo "CONFIG A: ELMM + TASER (BF16 pool)"
echo "================================================================"
start_server "bf16_baseline" "ELMM_INT8_POOL=0"
run_bench "bf16_baseline"
kill_server

echo ""

# =============================================================================
# B: INT8 W8A16 Pool
# =============================================================================
echo "================================================================"
echo "CONFIG B: ELMM + TASER + INT8 Pool (W8A16)"
echo "================================================================"
start_server "int8_pool" "ELMM_INT8_POOL=1"
run_bench "int8_pool"
kill_server

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "RESULTS"
echo "================================================================"

python3 - <<'PYEOF'
import json, sys

def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None

a = load_result("/root/MoE-SD/results/int8_ab_test/bf16_baseline_summary.json")
b = load_result("/root/MoE-SD/results/int8_ab_test/int8_pool_summary.json")

if a and b:
    tps_a = a.get("avg_tps", 0)
    tps_b = b.get("avg_tps", 0)
    
    if tps_a > 0 and tps_b > 0:
        speedup = tps_b / tps_a
        print(f"  BF16 baseline:  {tps_a:.2f} tok/s")
        print(f"  INT8 pool:      {tps_b:.2f} tok/s")
        print(f"  Speedup:        {speedup:.3f}× ({(speedup - 1) * 100:+.1f}%)")
    else:
        print("  Could not parse tok/s from results")
        print(f"  BF16 keys: {list(a.keys())}")
        print(f"  INT8 keys: {list(b.keys())}")
else:
    print("  Missing result files")
PYEOF

echo "================================================================"
