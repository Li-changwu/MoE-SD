#!/usr/bin/env bash
# =============================================================================
# PredCache Quick Benchmark — Compare PredCache vs LRU vs DIPP
# =============================================================================
set -euo pipefail

PYTHON="/opt/miniconda3/envs/moe-sd/bin/python"
RUNNER="/root/MoE-SD/scripts/bench_humaneval_runner.py"
MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATASET="/root/MoE-SD/data/humaneval_bench.jsonl"
RESULT_DIR="/root/MoE-SD/results/predcache_eval"

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
    echo "[$(timestamp)] Running: $label (cpu-offload-gb=$offload_gb)"
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

# 1. SD + LRU baseline
echo ">>> [1/4] SD + LRU (baseline)"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 BRISKMOE_ELP=0 BRISKMOE_DIPP=0 BRISKMOE_PREDCACHE=0 \
    run_bench "sd_lru" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# 2. SD + DIPP only (previous best: +7%)
echo ">>> [2/4] SD + DIPP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 BRISKMOE_ELP=0 BRISKMOE_DIPP=1 BRISKMOE_PREDCACHE=0 \
    run_bench "sd_dipp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# 3. SD + PredCache (forward-looking eviction)
echo ">>> [3/4] SD + PredCache"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 BRISKMOE_ELP=0 BRISKMOE_DIPP=0 BRISKMOE_PREDCACHE=1 \
    run_bench "sd_predcache" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# 4. SD + PredCache + DIPP (eviction + prefetch)
echo ">>> [4/4] SD + PredCache + DIPP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 BRISKMOE_ELP=0 BRISKMOE_DIPP=1 BRISKMOE_PREDCACHE=1 \
    run_bench "sd_predcache_dipp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# Summary
echo ""
echo "================================================================"
echo "[$(timestamp)] Results Summary"
echo "================================================================"

$PYTHON << 'PYEOF'
import json, os
result_dir = "/root/MoE-SD/results/predcache_eval"
labels = ["sd_lru", "sd_dipp", "sd_predcache", "sd_predcache_dipp"]
results = {}
for label in labels:
    path = os.path.join(result_dir, label, "result.json")
    if os.path.exists(path):
        with open(path) as f:
            results[label] = json.load(f)

if not results:
    print("No results found!")
    exit(1)

print(f"\n{'Config':<25} {'Latency(s)':>10} {'TPS':>8} {'TPOT(ms)':>10} {'vs LRU':>8}")
print("-" * 65)

lru_lat = results.get("sd_lru", {}).get("avg_latency")
for label in labels:
    if label not in results:
        continue
    d = results[label]
    lat = d["avg_latency"]
    tps = d["tps_mean"]
    tpot = d["tpot_mean_ms"]
    ratio = f"{lru_lat / lat:.2f}x" if lru_lat else "—"
    print(f"{label:<25} {lat:>10.3f} {tps:>8.2f} {tpot:>10.2f} {ratio:>8}")
PYEOF

echo ""
echo "[$(timestamp)] Done! Results in: $RESULT_DIR"
