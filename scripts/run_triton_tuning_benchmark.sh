#!/usr/bin/env bash
# =============================================================================
# Triton Config Tuning: E2E A/B Benchmark
# =============================================================================
# Compares ELMM with Oracle Prefetch WITH Triton A6000-tuned tile configs.
# Baseline: Prior Oracle Prefetch result (16.85 tok/s)
# =============================================================================
set -euo pipefail
cd /root/MoE-SD

MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULTS_DIR="results/triton_tuning"
PORT=8000

SPEC_CONFIG="{\"model\": \"${SPECULATOR}\", \"method\": \"eagle3\", \"num_speculative_tokens\": 3, \"draft_tensor_parallel_size\": 1}"

mkdir -p "$RESULTS_DIR"

VLLM_COMMON=(
    --model "$MODEL"
    --speculative-config "$SPEC_CONFIG"
    --tensor-parallel-size 1
    --cpu-offload-gb 30
    --max-model-len 4096
    --gpu-memory-utilization 0.85
    --enforce-eager
    --port "$PORT"
)

wait_server() {
    local timeout=300
    local start=$(date +%s)
    echo "[BENCH] Waiting for server on port $PORT..."
    while true; do
        if curl -s "http://127.0.0.1:${PORT}/health" | grep -q "ok\|healthy\|200"; then
            echo "[BENCH] Server ready!"
            return 0
        fi
        local now=$(date +%s)
        if (( now - start > timeout )); then
            echo "[BENCH] ERROR: Server did not become ready in ${timeout}s"
            return 1
        fi
        sleep 3
    done
}

kill_server() {
    echo "[BENCH] Stopping server..."
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
    sleep 3
}

run_bench() {
    local label="$1"
    local output_prefix="${RESULTS_DIR}/${label}"
    echo "[BENCH] Running benchmark: $label"
    conda run -n moe-sd python scripts/elmm_bench_worker.py \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "$MODEL" \
        --num-prompts 5 \
        --max-tokens 128 \
        --warmup 2 \
        --output "${output_prefix}_requests.jsonl" \
        --summary "${output_prefix}_summary.json"
    echo "[BENCH] Benchmark complete: $label"
    cat "${output_prefix}_summary.json"
}

# ============================================================================
# Triton Tuned: ELMM + Oracle Prefetch + A6000-tuned tile configs
# ============================================================================
echo ""
echo "============================================================"
echo "  Config: ELMM v2 + Oracle Prefetch + Triton Tuned Configs"
echo "============================================================"

kill_server

export VLLM_PLUGINS=elmm
export ELMM_CACHE_GB=4
export ELMM_LOG_INTERVAL=0
export ELMM_STALE_REMAP=16
export ELMM_ORACLE_PREFETCH=1
export ELMM_GPU_CACHE=0

conda run -n moe-sd python -m vllm.entrypoints.openai.api_server \
    "${VLLM_COMMON[@]}" \
    > "${RESULTS_DIR}/server_tuned.log" 2>&1 &
SERVER_PID=$!
echo "[BENCH] Server PID: $SERVER_PID"

if wait_server; then
    run_bench "triton_tuned"
else
    echo "[BENCH] FAILED to start server"
    tail -30 "${RESULTS_DIR}/server_tuned.log"
fi

kill_server

echo ""
echo "============================================================"
echo "  Triton Config Tuning benchmark complete!"
echo "  Results: $RESULTS_DIR"
echo "============================================================"
