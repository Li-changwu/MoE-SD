#!/usr/bin/env bash
# =============================================================================
# ALPS Phase 1: CUDA Graph A/B Benchmark
# =============================================================================
# Compares ELMM with CUDA Graph (ALPS Phase 1) vs without.
#   A) Baseline:   ELMM v2 + TASER (ELMM_CUDA_GRAPH=0)
#   B) ALPS Ph.1:  ELMM v2 + TASER + CUDA Graph (ELMM_CUDA_GRAPH=1)
# =============================================================================
set -euo pipefail
cd /root/MoE-SD

MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULTS_DIR="results/alps_phase1"
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
    local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        echo "[BENCH] WARNING: $gpu_procs GPU process(es) still running, force killing..."
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        sleep 5
    fi
    echo "[BENCH] Server stopped."
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
# A) Baseline: ELMM + TASER (CUDA Graph disabled)
# ============================================================================
echo ""
echo "============================================================"
echo "  Config A: ELMM v2 + TASER (CUDA Graph OFF)"
echo "============================================================"

kill_server

VLLM_PLUGINS=elmm \
    ELMM_CACHE_GB=4 \
    ELMM_LOG_INTERVAL=0 \
    ELMM_STALE_REMAP=16 \
    ELMM_CUDA_GRAPH=0 \
    conda run -n moe-sd python -m vllm.entrypoints.openai.api_server \
        "${VLLM_COMMON[@]}" \
    > "${RESULTS_DIR}/server_baseline.log" 2>&1 &
SERVER_PID=$!
echo "[BENCH] Server PID: $SERVER_PID"

if wait_server; then
    run_bench "baseline_no_graph"
else
    echo "[BENCH] FAILED to start baseline server"
    tail -30 "${RESULTS_DIR}/server_baseline.log"
fi

kill_server

# ============================================================================
# B) ALPS Phase 1: ELMM + TASER + CUDA Graph
# ============================================================================
echo ""
echo "============================================================"
echo "  Config B: ELMM v2 + TASER + CUDA Graph ON"
echo "============================================================"

VLLM_PLUGINS=elmm \
    ELMM_CACHE_GB=4 \
    ELMM_LOG_INTERVAL=0 \
    ELMM_STALE_REMAP=16 \
    ELMM_CUDA_GRAPH=1 \
    conda run -n moe-sd python -m vllm.entrypoints.openai.api_server \
        "${VLLM_COMMON[@]}" \
    > "${RESULTS_DIR}/server_cudagraph.log" 2>&1 &
SERVER_PID=$!
echo "[BENCH] Server PID: $SERVER_PID"

if wait_server; then
    run_bench "alps_cuda_graph"
else
    echo "[BENCH] FAILED to start CUDA Graph server"
    tail -30 "${RESULTS_DIR}/server_cudagraph.log"
fi

kill_server

# ============================================================================
# Compare results
# ============================================================================
echo ""
echo "============================================================"
echo "  ALPS Phase 1: A/B Comparison"
echo "============================================================"

conda run -n moe-sd python3 -c "
import json, os, sys

results_dir = '${RESULTS_DIR}'
configs = {
    'baseline_no_graph': 'ELMM + TASER (no graph)',
    'alps_cuda_graph':   'ELMM + TASER + CUDA Graph',
}

print()
print(f'{\"Config\":<35} {\"avg tok/s\":>10} {\"total_tok\":>10} {\"time(s)\":>8}')
print('-' * 67)

tps_values = {}
for key, label in configs.items():
    fpath = os.path.join(results_dir, f'{key}_summary.json')
    if not os.path.exists(fpath):
        print(f'{label:<35} --- (no data)')
        continue
    with open(fpath) as f:
        s = json.load(f)
    tps = s.get('avg_tokens_per_sec', 0)
    tps_values[key] = tps
    print(f'{label:<35} {tps:>9.2f}  {s.get(\"total_output_tokens\",0):>10}  {s.get(\"total_time_sec\",0):>7.1f}')

if len(tps_values) == 2:
    base = tps_values.get('baseline_no_graph', 0)
    graph = tps_values.get('alps_cuda_graph', 0)
    if base > 0:
        improvement = (graph - base) / base * 100
        print()
        print(f'CUDA Graph improvement: {improvement:+.1f}%  ({base:.2f} -> {graph:.2f} tok/s)')
print()
"

echo "[BENCH] Done. Logs in ${RESULTS_DIR}/"
