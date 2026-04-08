#!/usr/bin/env bash
# =============================================================================
# BriskMoE Ablation Benchmark (BENCHMARK_DESIGN.md §3.1 + §3.5 消融)
#
# Configs tested:
#   1. ar_vanilla       — AR baseline (no SD, no ELMM)
#   2. sd_vanilla       — SD + ELMM (LRU eviction, locality prefetch)
#   3. sd_sacr          — SD + ELMM + SACR eviction
#   4. sd_elp           — SD + ELMM + ELP partition
#   5. sd_dipp          — SD + ELMM + DIPP prefetch
#   6. sd_sacr_elp      — SD + ELMM + SACR + ELP
#   7. sd_briskmoe      — SD + ELMM + SACR + ELP + DIPP (full BriskMoE)
#
# Follows CCF-A consensus: batch=1, sequential, input=512, output=128
# =============================================================================

set -euo pipefail

# ── Paths ──
VLLM="/opt/miniconda3/envs/moe-sd/bin/vllm"
PYTHON="/opt/miniconda3/envs/moe-sd/bin/python"
MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULT_DIR="/root/MoE-SD/results/briskmoe_ablation"

# ── Benchmark Parameters (CCF-A consensus) ──
INPUT_LEN=512
OUTPUT_LEN=128
BATCH_SIZE=1
NUM_ITERS=10
WARMUP=3
GPU_MEM_UTIL=0.90
CPU_OFFLOAD=30         # AR baseline (no ELMM overhead)
CPU_OFFLOAD_ELMM=45    # ELMM configs need more offload: ELMM cache+scratch ~10GB
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
    local outdir="$RESULT_DIR/$label"
    mkdir -p "$outdir"

    echo ""
    echo "================================================================"
    echo "[$(timestamp)] Running: $label (cpu-offload-gb=$offload_gb)"
    echo "================================================================"
    echo "  Output: $outdir/latency.json"
    echo ""

    "$VLLM" bench latency \
        --model "$MODEL" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --batch-size "$BATCH_SIZE" \
        --num-iters "$NUM_ITERS" \
        --num-iters-warmup "$WARMUP" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --cpu-offload-gb "$offload_gb" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        --trust-remote-code \
        --dtype bfloat16 \
        --output-json "$outdir/latency.json" \
        "$@" \
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
echo ""; echo ">>> [1/7] AR Baseline"
VLLM_PLUGINS="" \
    run_bench "ar_vanilla" "$CPU_OFFLOAD"

# =============================================================================
# 2. SD Vanilla (LRU eviction, locality prefetch) — ELMM baseline
# =============================================================================
echo ""; echo ">>> [2/7] SD + ELMM (LRU baseline)"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
    run_bench "sd_vanilla" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 3. SD + SACR (SACR eviction only)
# =============================================================================
echo ""; echo ">>> [3/7] SD + SACR"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=1 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
    run_bench "sd_sacr" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 4. SD + ELP (ELP partition + LRU eviction)
# =============================================================================
echo ""; echo ">>> [4/7] SD + ELP"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=1 \
BRISKMOE_DIPP=0 \
    run_bench "sd_elp" "$CPU_OFFLOAD_ELMM" \
    --speculative-config "$SPEC_CONFIG"

# =============================================================================
# 5. SD + DIPP (DIPP prefetch only)
# =============================================================================
echo ""; echo ">>> [5/7] SD + DIPP"
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
echo ""; echo ">>> [6/7] SD + SACR + ELP"
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
echo ""; echo ">>> [7/7] SD + BriskMoE (Full)"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=1 \
BRISKMOE_ELP=1 \
BRISKMOE_DIPP=1 \
    run_bench "sd_briskmoe" "$CPU_OFFLOAD_ELMM" \
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

result_dir = "/root/MoE-SD/results/briskmoe_ablation"
labels = [
    "ar_vanilla",
    "sd_vanilla",
    "sd_sacr",
    "sd_elp",
    "sd_dipp",
    "sd_sacr_elp",
    "sd_briskmoe",
]

results = {}
for label in labels:
    path = os.path.join(result_dir, label, "latency.json")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found")
        continue
    with open(path) as f:
        results[label] = json.load(f)

if not results:
    print("No results found!")
    exit(1)

print("\n" + "=" * 90)
print("BriskMoE Ablation Results")
print("=" * 90)
print(f"  Model:    Qwen3-30B-A3B + EAGLE-3 K=3")
print(f"  Hardware: A6000 48GB, PCIe Gen4, 8GB expert cache")
print(f"  Config:   input={512}, output={128}, batch=1, iters=30")
print("=" * 90)

header = f"{'Config':<20} {'TPOT(ms)':>10} {'TPS':>8} {'TTFT(ms)':>10} {'E2EL(ms)':>10} {'Accept':>8}"
print(header)
print("-" * 70)

for label in labels:
    if label not in results:
        continue
    d = results[label]
    tpot = d.get("avg_latency", 0) * 1000 / max(d.get("output_len", 128), 1)
    tps = 1000.0 / tpot if tpot > 0 else 0
    ttft = d.get("avg_latency", 0) * 1000  # approximate
    e2el = d.get("avg_latency", 0) * 1000
    accept = d.get("acceptance_rate", "N/A")
    if isinstance(accept, float):
        accept = f"{accept:.3f}"

    print(f"{label:<20} {tpot:>10.2f} {tps:>8.2f} {ttft:>10.1f} {e2el:>10.1f} {accept:>8}")

# Quick speedup summary
if "ar_vanilla" in results and "sd_briskmoe" in results:
    ar_lat = results["ar_vanilla"].get("avg_latency", 1)
    bm_lat = results["sd_briskmoe"].get("avg_latency", 1)
    speedup = ar_lat / bm_lat if bm_lat > 0 else 0
    print(f"\nBriskMoE speedup over AR: {speedup:.2f}×")

if "sd_vanilla" in results and "sd_briskmoe" in results:
    sd_lat = results["sd_vanilla"].get("avg_latency", 1)
    bm_lat = results["sd_briskmoe"].get("avg_latency", 1)
    speedup = sd_lat / bm_lat if bm_lat > 0 else 0
    print(f"BriskMoE speedup over SD+LRU: {speedup:.2f}×")

PYEOF

echo ""
echo "[$(timestamp)] All benchmarks complete!"
echo "Results in: $RESULT_DIR"
