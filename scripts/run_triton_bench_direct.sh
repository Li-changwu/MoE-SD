#!/usr/bin/env bash
# =============================================================================
# Triton Config Tuning: E2E Benchmark (direct python, no conda run)
# =============================================================================
set -euo pipefail
cd /root/MoE-SD

PYTHON=/opt/miniconda3/envs/moe-sd/bin/python
MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULTS_DIR="results/triton_tuning"
PORT=8000
SPEC_CONFIG="{\"model\": \"${SPECULATOR}\", \"method\": \"eagle3\", \"num_speculative_tokens\": 3, \"draft_tensor_parallel_size\": 1}"

mkdir -p "$RESULTS_DIR"

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
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
    sleep 3
}

echo ""
echo "============================================================"
echo "  Triton Tuned: ELMM v2 + Oracle + A6000-tuned tile configs"
echo "============================================================"

kill_server

VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=4 \
ELMM_LOG_INTERVAL=0 \
ELMM_STALE_REMAP=16 \
ELMM_ORACLE_PREFETCH=1 \
ELMM_GPU_CACHE=0 \
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --speculative-config "$SPEC_CONFIG" \
    --tensor-parallel-size 1 \
    --cpu-offload-gb 30 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port "$PORT" \
    > "${RESULTS_DIR}/server_tuned.log" 2>&1 &
SERVER_PID=$!
echo "[BENCH] Server PID: $SERVER_PID"

if wait_server; then
    echo "[BENCH] Running benchmark: triton_tuned"
    $PYTHON scripts/elmm_bench_worker.py \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "$MODEL" \
        --num-prompts 5 \
        --max-tokens 128 \
        --warmup 2 \
        --output "${RESULTS_DIR}/triton_tuned_requests.jsonl" \
        --summary "${RESULTS_DIR}/triton_tuned_summary.json"
    echo "[BENCH] Benchmark complete!"
    echo ""
    echo "=== RESULTS ==="
    cat "${RESULTS_DIR}/triton_tuned_summary.json"
else
    echo "[BENCH] FAILED to start server"
    tail -50 "${RESULTS_DIR}/server_tuned.log"
fi

kill_server

echo ""
echo "============================================================"
echo "  Triton Config Tuning benchmark complete!"
echo "============================================================"
