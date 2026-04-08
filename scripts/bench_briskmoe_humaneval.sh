#!/usr/bin/env bash
# =============================================================================
# BriskMoE HumanEval Benchmark (BENCHMARK_DESIGN.md §3.1)
#
# Uses REAL HumanEval prompts (not random tokens) to produce realistic expert
# routing patterns with locality — essential for evaluating BriskMoE cache.
#
# Configs tested (same 7 as ablation):
#   1. ar_vanilla       — AR baseline (no SD, no ELMM)
#   2. sd_vanilla       — SD + ELMM (LRU eviction)
#   3. sd_sacr          — SD + ELMM + SACR eviction
#   4. sd_elp           — SD + ELMM + ELP partition
#   5. sd_dipp          — SD + ELMM + DIPP prefetch
#   6. sd_sacr_elp      — SD + ELMM + SACR + ELP
#   7. sd_briskmoe      — SD + ELMM + SACR + ELP + DIPP (full BriskMoE)
#
# Follows CCF-A consensus: batch=1, sequential, real dataset
# =============================================================================

set -euo pipefail

# ── Paths ──
PYTHON="/opt/miniconda3/envs/moe-sd/bin/python"
RUNNER="/root/MoE-SD/scripts/bench_humaneval_runner.py"
MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATASET="/root/MoE-SD/data/humaneval_bench.jsonl"
RESULT_DIR="/root/MoE-SD/results/briskmoe_humaneval"

# ── Benchmark Parameters ──
OUTPUT_LEN=128
NUM_PROMPTS=50
WARMUP_PROMPTS=5
GPU_MEM_UTIL=0.90
CPU_OFFLOAD=30         # AR baseline (no ELMM overhead)
CPU_OFFLOAD_ELMM=45    # ELMM configs need more offload
MAX_MODEL_LEN=4096

# ── Spec Decode Config ──
SPEC_CONFIG='{"method":"eagle3","model":"'$SPECULATOR'","num_speculative_tokens":3}'

# ── ELMM defaults ──
ELMM_CACHE_GB=8
ELMM_STALE_REMAP=4

# ── Helper ──
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

# ── Ensure result directory ──
mkdir -p "$RESULT_DIR"

# ── GPU check ──
echo "========================================"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv 2>/dev/null || true
echo "========================================"

# ── Kill any leftover vllm processes ──
pkill -f "vllm" 2>/dev/null || true
sleep 2

# =============================================================================
# 1. AR Baseline (no SD, no ELMM)
# =============================================================================
echo ""; echo ">>> [1/9] AR Baseline"
VLLM_PLUGINS="" \
    run_bench "ar_vanilla" "$CPU_OFFLOAD"

# =============================================================================
# 2. SD Vanilla (LRU eviction) — ELMM baseline
# =============================================================================
echo ""; echo ">>> [2/9] SD + ELMM (LRU baseline)"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
    run_bench "sd_vanilla" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 3. SD + SACR
# =============================================================================
echo ""; echo ">>> [3/9] SD + SACR"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=1 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
    run_bench "sd_sacr" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 4. SD + ELP
# =============================================================================
echo ""; echo ">>> [4/9] SD + ELP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=1 \
BRISKMOE_DIPP=0 \
    run_bench "sd_elp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 5. SD + DIPP
# =============================================================================
echo ""; echo ">>> [5/9] SD + DIPP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=1 \
    run_bench "sd_dipp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 6. SD + SACR + ELP
# =============================================================================
echo ""; echo ">>> [6/9] SD + SACR + ELP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=1 \
BRISKMOE_ELP=1 \
BRISKMOE_DIPP=0 \
    run_bench "sd_sacr_elp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 7. SD + BriskMoE Full (SACR + ELP + DIPP)
# =============================================================================
echo ""; echo ">>> [7/9] SD + BriskMoE (Full)"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=1 \
BRISKMOE_ELP=1 \
BRISKMOE_DIPP=1 \
BRISKMOE_PREDCACHE=0 \
    run_bench "sd_briskmoe" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 8. SD + PredCache (forward-looking eviction only)
# =============================================================================
echo ""; echo ">>> [8/9] SD + PredCache"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
BRISKMOE_PREDCACHE=1 \
    run_bench "sd_predcache" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 9. SD + PredCache + DIPP (forward-looking eviction + prefetch)
# =============================================================================
echo ""; echo ">>> [9/9] SD + PredCache + DIPP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=1 \
BRISKMOE_PREDCACHE=1 \
    run_bench "sd_predcache_dipp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# Summary: Parse and compare results
# =============================================================================
echo ""
echo "================================================================"
echo "[$(timestamp)] Parsing results..."
echo "================================================================"

$PYTHON << 'PYEOF'
import json
import os

result_dir = "/root/MoE-SD/results/briskmoe_humaneval"
labels = [
    "ar_vanilla",
    "sd_vanilla",
    "sd_sacr",
    "sd_elp",
    "sd_dipp",
    "sd_sacr_elp",
    "sd_briskmoe",
    "sd_predcache",
    "sd_predcache_dipp",
]

results = {}
for label in labels:
    path = os.path.join(result_dir, label, "result.json")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found")
        continue
    with open(path) as f:
        results[label] = json.load(f)

if not results:
    print("No results found!")
    exit(1)

print("\n" + "=" * 90)
print("BriskMoE HumanEval Benchmark Results")
print("=" * 90)
print("  Model:    Qwen3-30B-A3B + EAGLE-3 K=3")
print("  Hardware: A6000 48GB, PCIe Gen4, 8GB expert cache")
print("  Dataset:  HumanEval (real code prompts)")
print("=" * 90)

header = f"{'Config':<20} {'Latency(s)':>10} {'TPS':>8} {'TPOT(ms)':>10} {'p50(s)':>8} {'p99(s)':>8}"
print(header)
print("-" * 70)

ar_lat = results.get("ar_vanilla", {}).get("avg_latency", None)
sd_lat = results.get("sd_vanilla", {}).get("avg_latency", None)

for label in labels:
    if label not in results:
        continue
    d = results[label]
    lat = d["avg_latency"]
    tps = d["tps_mean"]
    tpot = d["tpot_mean_ms"]
    p50 = d["percentiles"]["50"]["latency"]
    p99 = d["percentiles"]["99"]["latency"]

    speedup_str = ""
    if ar_lat and label != "ar_vanilla":
        speedup_str += f"  {ar_lat / lat:.2f}x AR"
    if sd_lat and label not in ("ar_vanilla", "sd_vanilla"):
        speedup_str += f"  {sd_lat / lat:.2f}x LRU"

    print(f"{label:<20} {lat:>10.3f} {tps:>8.2f} {tpot:>10.2f} {p50:>8.3f} {p99:>8.3f}{speedup_str}")

if ar_lat and "sd_briskmoe" in results:
    bm_lat = results["sd_briskmoe"]["avg_latency"]
    print(f"\nBriskMoE speedup over AR: {ar_lat / bm_lat:.2f}x")

if sd_lat and "sd_briskmoe" in results:
    bm_lat = results["sd_briskmoe"]["avg_latency"]
    print(f"BriskMoE speedup over SD+LRU: {sd_lat / bm_lat:.2f}x")

PYEOF

echo ""
echo "[$(timestamp)] All benchmarks complete!"
echo "Results in: $RESULT_DIR"
