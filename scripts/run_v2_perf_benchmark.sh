#!/usr/bin/env bash
# =============================================================================
# ELMM v1 vs v2 A/B Performance Benchmark
# =============================================================================
# Compares baseline ELMM (v1, Milestone 1) against new ELMM (v2, Milestone 2)
# with temporal locality collection, adaptive cache budget, and draft prefetch.
#
# Strategy:
#   A) ELMM v1 (baseline): disable new features
#   B) ELMM v2 (new):      enable all new features
#
# Each config:
#   1. Launch vLLM server with ELMM plugin
#   2. Wait for server ready
#   3. Send 5 benchmark requests, measure tok/s
#   4. Kill server, collect stats
# =============================================================================

set -euo pipefail
cd /root/MoE-SD

MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULTS_DIR="results/v2_perf_comparison"
PORT=8000

SPEC_CONFIG="{\"model\": \"${SPECULATOR}\", \"method\": \"eagle3\", \"num_speculative_tokens\": 3, \"draft_tensor_parallel_size\": 1}"

mkdir -p "$RESULTS_DIR"

# Common vLLM flags
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
    # Force kill if still alive
    pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
    sleep 3
    # Verify GPU is free
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
# A) ELMM v1 — Baseline (new features disabled)
# ============================================================================
echo ""
echo "============================================================"
echo "  Config A: ELMM v1 (baseline — new features disabled)"
echo "============================================================"

kill_server

echo "[BENCH] Launching ELMM v1 server..."
VLLM_PLUGINS=elmm \
    ELMM_CACHE_GB=4 \
    ELMM_LOG_INTERVAL=0 \
    ELMM_PREFETCH=0 \
    ELMM_LOCALITY=0 \
    ELMM_ADAPTIVE_BUDGET=0 \
    conda run -n moe-sd python -m vllm.entrypoints.openai.api_server \
        "${VLLM_COMMON[@]}" \
    > "${RESULTS_DIR}/server_v1.log" 2>&1 &
SERVER_PID=$!
echo "[BENCH] Server PID: $SERVER_PID"

if wait_server; then
    run_bench "v1_baseline"
else
    echo "[BENCH] FAILED to start v1 server"
    cat "${RESULTS_DIR}/server_v1.log" | tail -30
fi

kill_server

# ============================================================================
# B) ELMM v2 — All new features enabled
# ============================================================================
echo ""
echo "============================================================"
echo "  Config B: ELMM v2 (all new features enabled)"
echo "============================================================"

echo "[BENCH] Launching ELMM v2 server..."
VLLM_PLUGINS=elmm \
    ELMM_CACHE_GB=4 \
    ELMM_LOG_INTERVAL=0 \
    ELMM_PREFETCH=1 \
    ELMM_LOCALITY=1 \
    ELMM_LOCALITY_DIR="${RESULTS_DIR}/v2_locality" \
    ELMM_ADAPTIVE_BUDGET=1 \
    ELMM_REBALANCE_INTERVAL=200 \
    conda run -n moe-sd python -m vllm.entrypoints.openai.api_server \
        "${VLLM_COMMON[@]}" \
    > "${RESULTS_DIR}/server_v2.log" 2>&1 &
SERVER_PID=$!
echo "[BENCH] Server PID: $SERVER_PID"

if wait_server; then
    run_bench "v2_features"
else
    echo "[BENCH] FAILED to start v2 server"
    cat "${RESULTS_DIR}/server_v2.log" | tail -30
fi

kill_server

# ============================================================================
# Compare results
# ============================================================================
echo ""
echo "============================================================"
echo "  Performance Comparison"
echo "============================================================"

conda run -n moe-sd python3 -c "
import json, os

results_dir = '${RESULTS_DIR}'
configs = {
    'v1_baseline': 'ELMM v1 (baseline)',
    'v2_features': 'ELMM v2 (all features)',
}

print()
print(f'{\"Config\":<35} {\"avg tok/s\":>10} {\"total_tok\":>10} {\"time(s)\":>8} {\"requests\":>8}')
print('-' * 75)

summaries = {}
for key, label in configs.items():
    path = os.path.join(results_dir, f'{key}_summary.json')
    if os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        summaries[key] = s
        print(f'{label:<35} {s[\"avg_tps\"]:>10.2f} {s[\"total_tokens\"]:>10} {s[\"total_time_s\"]:>8.1f} {s[\"num_requests\"]:>8}')
    else:
        print(f'{label:<35} {'N/A':>10}')

print()
if 'v1_baseline' in summaries and 'v2_features' in summaries:
    v1 = summaries['v1_baseline']['avg_tps']
    v2 = summaries['v2_features']['avg_tps']
    if v1 > 0:
        ratio = v2 / v1
        delta_pct = (v2 - v1) / v1 * 100
        print(f'Performance change: v2/v1 = {ratio:.3f}x ({delta_pct:+.1f}%)')
        if ratio > 1.01:
            print('-> v2 is FASTER')
        elif ratio < 0.99:
            delta = (1 - ratio) * 100
            print(f'-> v2 has {delta:.1f}% overhead (expected: locality tracking + adaptive budget cost)')
        else:
            print('-> Performance is equivalent (within noise)')
else:
    print('Cannot compare — missing results')

# Show locality data if available
loc_dir = os.path.join(results_dir, 'v2_locality')
if os.path.exists(loc_dir):
    print()
    print('=== Locality Data (v2) ===')
    for fname in sorted(os.listdir(loc_dir)):
        fpath = os.path.join(loc_dir, fname)
        if fpath.endswith('.json'):
            with open(fpath) as f:
                data = json.load(f)
            print(f'  {fname}: {json.dumps(data, indent=2)[:500]}')
"

echo ""
echo "[BENCH] Done. Results in: ${RESULTS_DIR}/"
