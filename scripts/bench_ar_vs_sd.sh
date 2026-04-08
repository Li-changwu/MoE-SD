#!/usr/bin/env bash
# =============================================================================
# BriskMoE Benchmark: AR vs SD (Qwen3-30B-A3B)
# 
# 按照 BENCHMARK_DESIGN.md §3.1 设计，测试两种模式：
#   1. AR baseline: Qwen3-30B-A3B 单独推理（无 SD）
#   2. SD (EAGLE-3): Qwen3-30B-A3B + Eagle3 speculator (K=3)
#
# 指标：TPOT, TTFT, E2EL, throughput, acceptance_rate (SD only)
# 数据集：使用 vllm bench latency 内置随机 prompt（input_len=512, output_len=128）
#         + ShareGPT 真实数据集 (10-20 条) 通过 vllm bench serve 测试
# =============================================================================

set -euo pipefail

# ── Disable MoE-SD plugins to get clean vanilla vLLM baseline ──
# Without this, both ELMM and MoE-Infinity plugins load and cause OOM.
# For BriskMoE plugin benchmarks, use VLLM_PLUGINS=elmm instead.
export VLLM_PLUGINS=""

# ── Paths ──
VLLM="/opt/miniconda3/envs/moe-sd/bin/vllm"
PYTHON="/opt/miniconda3/envs/moe-sd/bin/python"
MODEL="/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULT_DIR="/root/MoE-SD/results/ar_vs_sd"

# ── Benchmark Parameters (CCF-A consensus) ──
INPUT_LEN=512
OUTPUT_LEN=128
BATCH_SIZE=1
NUM_ITERS=10          # 小规模快速验证
WARMUP=3
GPU_MEM_UTIL=0.90
CPU_OFFLOAD=30
MAX_MODEL_LEN=4096

# ── Common engine flags ──
COMMON_FLAGS=(
    --model "$MODEL"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --batch-size "$BATCH_SIZE"
    --num-iters "$NUM_ITERS"
    --num-iters-warmup "$WARMUP"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --cpu-offload-gb "$CPU_OFFLOAD"
    --max-model-len "$MAX_MODEL_LEN"
    --enforce-eager
    --trust-remote-code
    --dtype bfloat16
)

# ── Helper ──
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_bench() {
    local label="$1"
    shift
    local outdir="$RESULT_DIR/$label"
    mkdir -p "$outdir"

    echo ""
    echo "================================================================"
    echo "[$(timestamp)] Running: $label"
    echo "================================================================"
    echo "  Output: $outdir/latency.json"
    echo ""

    # Run benchmark, tee to both terminal and log
    "$VLLM" bench latency \
        "${COMMON_FLAGS[@]}" \
        --output-json "$outdir/latency.json" \
        "$@" \
        2>&1 | tee "$outdir/bench.log"

    echo ""
    echo "[$(timestamp)] Completed: $label"
    echo ""
}

# ── Ensure result directory ──
mkdir -p "$RESULT_DIR"

# ── GPU check ──
echo "========================================"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv
echo "========================================"
echo ""

# ── Kill any leftover vllm processes ──
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# =============================================================================
# Experiment 1: AR Baseline (Qwen3-30B-A3B, no SD)
# =============================================================================
run_bench "ar_vanilla"

# =============================================================================
# Experiment 2: SD with EAGLE-3 (K=3)
# =============================================================================
SPEC_CONFIG='{"method":"eagle3","model":"'$SPECULATOR'","num_speculative_tokens":3}'
run_bench "sd_vanilla_k3" --speculative-config "$SPEC_CONFIG"

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
import sys

result_dir = "/root/MoE-SD/results/ar_vs_sd"
labels = ["ar_vanilla", "sd_vanilla_k3"]
results = {}

for label in labels:
    path = os.path.join(result_dir, label, "latency.json")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found, skipping")
        continue
    with open(path) as f:
        data = json.load(f)
    results[label] = data

if not results:
    print("No results found!")
    sys.exit(1)

# Print comparison table
print("\n" + "=" * 80)
print("BriskMoE Benchmark Results: AR vs SD")
print("=" * 80)
print(f"  Model:      Qwen3-30B-A3B-Instruct-2507")
print(f"  Hardware:   NVIDIA RTX A6000 48GB, PCIe Gen4")
print(f"  Input/Output: {512}/{128} tokens, batch=1")
print(f"  Iterations: 10 (warmup: 3)")
print("=" * 80)

# Format header
header = f"{'Metric':<35} {'AR Baseline':>15} {'SD (EAGLE3 K=3)':>18}"
print(header)
print("-" * 70)

def get_metric(data, key, default="N/A"):
    """Extract metric from vllm bench latency JSON output."""
    if key in data:
        val = data[key]
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)
    return default

# vllm bench latency outputs vary by version, try common keys
for label, data in results.items():
    if label == labels[0]:  # Print once
        # Show all available keys for debugging
        print(f"\n  Available metrics in output:")
        for k, v in sorted(data.items()):
            if not isinstance(v, (dict, list)):
                print(f"    {k}: {v}")
            elif isinstance(v, list) and len(v) < 10:
                print(f"    {k}: {v}")

# Key metrics comparison
metric_keys = [
    ("avg_latency", "Avg Latency (s)"),
    ("avg_latency_ms", "Avg Latency (ms)"),
    ("percentile_latencies", "Latency Percentiles"),
    ("elapsed_time", "Total Elapsed Time (s)"),
    ("throughput", "Throughput (req/s)"),
    ("output_throughput", "Output Throughput (tok/s)"),
    ("total_output_tokens", "Total Output Tokens"),
]

print(f"\n{'='*70}")
print(f"{'Metric':<35} {'AR Baseline':>15} {'SD (K=3)':>15}")
print(f"{'-'*70}")

for key, display_name in metric_keys:
    ar_val = get_metric(results.get("ar_vanilla", {}), key)
    sd_val = get_metric(results.get("sd_vanilla_k3", {}), key)
    print(f"  {display_name:<33} {ar_val:>15} {sd_val:>15}")

# Calculate speedup if possible
ar_data = results.get("ar_vanilla", {})
sd_data = results.get("sd_vanilla_k3", {})

ar_lat = ar_data.get("avg_latency") or ar_data.get("avg_latency_ms")
sd_lat = sd_data.get("avg_latency") or sd_data.get("avg_latency_ms")
if ar_lat and sd_lat and float(sd_lat) > 0:
    speedup = float(ar_lat) / float(sd_lat)
    print(f"\n  {'Latency Speedup (AR/SD)':<33} {speedup:>15.2f}x")

ar_tput = ar_data.get("output_throughput")
sd_tput = sd_data.get("output_throughput")
if ar_tput and sd_tput and float(ar_tput) > 0:
    tput_ratio = float(sd_tput) / float(ar_tput)
    print(f"  {'Throughput Speedup (SD/AR)':<33} {tput_ratio:>15.2f}x")

# Acceptance rate (SD only)
acc_rate = sd_data.get("acceptance_rate") or sd_data.get("mean_acceptance_rate")
if acc_rate:
    print(f"  {'SD Acceptance Rate':<33} {'N/A':>15} {float(acc_rate):>14.2f}%")

print(f"\n{'='*70}")
print("Done!")

# Save summary
summary_path = os.path.join(result_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "experiment": "ar_vs_sd",
        "model": "Qwen3-30B-A3B-Instruct-2507",
        "hardware": "NVIDIA RTX A6000 48GB",
        "input_len": 512,
        "output_len": 128,
        "batch_size": 1,
        "num_iters": 10,
        "results": results,
    }, f, indent=2)
print(f"\nSummary saved to: {summary_path}")
PYEOF

echo ""
echo "[$(timestamp)] All benchmarks completed."
