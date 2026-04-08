#!/usr/bin/env bash
# =============================================================================
# PredCache v3 — Lazy decay + LRU-with-protection eviction
# =============================================================================
set -euo pipefail

PYTHON="/opt/miniconda3/envs/moe-sd/bin/python"
RUNNER="/root/MoE-SD/scripts/bench_humaneval_runner.py"
MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATASET="/root/MoE-SD/data/humaneval_bench.jsonl"
RESULT_DIR="/root/MoE-SD/results/predcache_v3"

OUTPUT_LEN=128
NUM_PROMPTS=50
WARMUP_PROMPTS=5
GPU_MEM_UTIL=0.90
CPU_OFFLOAD_ELMM=45
MAX_MODEL_LEN=4096

SPEC_CONFIG='{"method":"eagle3","model":"'$SPECULATOR'","num_speculative_tokens":3}'
ELMM_CACHE_GB=8
ELMM_STALE_REMAP=4

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_bench() {
    local label="$1"
    local offload_gb="$2"
    shift 2
    local extra_args=("$@")
    local outdir="$RESULT_DIR/$label"
    mkdir -p "$outdir"

    echo ""
    echo "================================================================"
    echo "[$(timestamp)] Running: $label"
    echo "================================================================"

    PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --warmup-prompts "$WARMUP_PROMPTS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --cpu-offload-gb "$offload_gb" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        --trust-remote-code \
        --dtype bfloat16 \
        --output-json "$outdir/result.json" \
        "${extra_args[@]}" \
        2>&1 | tee "$outdir/bench.log"

    echo "[$(timestamp)] Completed: $label"
}

mkdir -p "$RESULT_DIR"
pkill -f "vllm" 2>/dev/null || true
sleep 2

# 1. SD + PredCache v3 (LRU-with-protection, lazy decay)
echo ">>> [1/2] SD + PredCache v3"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 BRISKMOE_ELP=0 BRISKMOE_DIPP=0 BRISKMOE_PREDCACHE=1 \
    run_bench "sd_predcache_v3" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# 2. SD + PredCache v3 + DIPP
echo ">>> [2/2] SD + PredCache v3 + DIPP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 BRISKMOE_ELP=0 BRISKMOE_DIPP=1 BRISKMOE_PREDCACHE=1 \
    run_bench "sd_predcache_v3_dipp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# Summary
echo ""
echo "================================================================"
echo "[$(timestamp)] Full Comparison (all versions)"
echo "================================================================"

$PYTHON << 'PYEOF'
import json, os
result_dirs = {
    "sd_lru":              "/root/MoE-SD/results/predcache_eval/sd_lru",
    "sd_dipp":             "/root/MoE-SD/results/predcache_eval/sd_dipp",
    "sd_predcache_v1":     "/root/MoE-SD/results/predcache_eval/sd_predcache",
    "sd_predcache_v2":     "/root/MoE-SD/results/predcache_v2/sd_predcache_v2",
    "sd_predcache_v3":     "/root/MoE-SD/results/predcache_v3/sd_predcache_v3",
    "sd_predcache_v3+dipp":"/root/MoE-SD/results/predcache_v3/sd_predcache_v3_dipp",
}

results = {}
for label, dpath in result_dirs.items():
    path = os.path.join(dpath, "result.json")
    if os.path.exists(path):
        with open(path) as f:
            results[label] = json.load(f)

print(f"\n{'Config':<25} {'TPS':>8} {'Latency':>10} {'vs LRU':>8}")
print("-" * 55)

lru_tps = results.get("sd_lru", {}).get("tps_mean")
for label in result_dirs:
    if label not in results:
        continue
    d = results[label]
    tps = d["tps_mean"]
    lat = d["avg_latency"]
    ratio = f"{tps / lru_tps:.2f}x" if lru_tps else "—"
    print(f"{label:<25} {tps:>8.2f} {lat:>10.3f} {ratio:>8}")
PYEOF

echo ""
echo "[$(timestamp)] Done!"
