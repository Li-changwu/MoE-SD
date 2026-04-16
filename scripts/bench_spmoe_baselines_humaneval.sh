#!/usr/bin/env bash
# =============================================================================
# SP-MoE baseline reproduction on HumanEval (4 methods)
#   1) Mixtral-Offloading+SD  (LRU offloading, no prefetch)
#   2) MoE-Infinity+SD        (sparsity-aware trace matching via PredCache)
#   3) AdapMoE+SD             (adaptive gating prediction prefetch)
#   4) BriskMoE+SD (Full)     (SACR + ELP + DIPP)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_PYTHON_CANDIDATES=(
	"/home/cdb/miniforge3/envs/briskmoe/bin/python"
	"/opt/miniconda3/envs/moe-sd/bin/python"
)
if [[ -n "${PYTHON:-}" ]]; then
	:
else
	for candidate in "${DEFAULT_PYTHON_CANDIDATES[@]}"; do
		if [[ -x "$candidate" ]]; then
			PYTHON="$candidate"
			break
		fi
	done
	PYTHON="${PYTHON:-python3}"
fi

RUNNER="${RUNNER:-$REPO_ROOT/scripts/bench_humaneval_runner.py}"
MODEL="${MODEL:-$REPO_ROOT/model/Qwen3-30B-A3B-Instruct-2507}"
SPECULATOR="${SPECULATOR:-$REPO_ROOT/model/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3}"
DATASET="${DATASET:-$REPO_ROOT/data/humaneval_bench.jsonl}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/spmoe_baselines_humaneval}"

OUTPUT_LEN=${OUTPUT_LEN:-100}
NUM_PROMPTS=${NUM_PROMPTS:-50}
WARMUP_PROMPTS=${WARMUP_PROMPTS:-5}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
CPU_OFFLOAD_ELMM=${CPU_OFFLOAD_ELMM:-45}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}
SPEC_CONFIG='{"method":"eagle3","model":"'$SPECULATOR'","num_speculative_tokens":'"$NUM_SPEC_TOKENS"'}'

ELMM_CACHE_GB=${ELMM_CACHE_GB:-8}
ELMM_STALE_REMAP=${ELMM_STALE_REMAP:-4}

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

	PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
		--model "$MODEL" \
		--dataset "$DATASET" \
		--output-len "$OUTPUT_LEN" \
		--num-prompts "$NUM_PROMPTS" \
		--warmup-prompts "$WARMUP_PROMPTS" \
		--gpu-memory-utilization "$GPU_MEM_UTIL" \
		--cpu-offload-gb "$CPU_OFFLOAD_ELMM" \
		--max-model-len "$MAX_MODEL_LEN" \
		--enforce-eager \
		--trust-remote-code \
		--dtype bfloat16 \
		--speculative-config "$SPEC_CONFIG" \
		--output-json "$outdir/result.json" \
		"$@" \
		2>&1 | tee "$outdir/bench.log"

	echo "[$(timestamp)] Completed: $label"
}

mkdir -p "$RESULT_DIR"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
	best_gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
		| sort -t',' -k2 -nr \
		| head -n1 \
		| cut -d',' -f1 \
		| tr -d '[:space:]')"
	if [[ -n "$best_gpu" ]]; then
		export CUDA_VISIBLE_DEVICES="$best_gpu"
		echo "[$(timestamp)] Auto-selected GPU index: $CUDA_VISIBLE_DEVICES (max free memory)"
	fi
fi

echo "========================================"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv 2>/dev/null || true
echo "========================================"

pkill -f "vllm" 2>/dev/null || true
sleep 2

# 1) Mixtral-Offloading+SD (LRU only, no prefetch)
echo ""; echo ">>> [1/4] Mixtral-Offloading+SD"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_PREFETCH=0 \
ELMM_ORACLE_PREFETCH=0 \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
BRISKMOE_PREDCACHE=0 \
BRISKMOE_ADAPMOE=0 \
	run_bench "mixtral_offloading_sd"

# 2) MoE-Infinity+SD (sparsity-aware expert cache with request-level traces)
echo ""; echo ">>> [2/4] MoE-Infinity+SD"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_PREFETCH=1 \
ELMM_ORACLE_PREFETCH=0 \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
BRISKMOE_PREDCACHE=1 \
BRISKMOE_PREDCACHE_EAMC_CAP=120 \
BRISKMOE_PREDCACHE_MATCH_TOPK=3 \
BRISKMOE_PREDCACHE_MATCH_DIST=0.35 \
BRISKMOE_PREDCACHE_REQUEST_ITERS=64 \
BRISKMOE_ADAPMOE=0 \
	run_bench "moe_infinity_sd"

# 3) AdapMoE+SD (adaptive gating prediction prefetch)
echo ""; echo ">>> [3/4] AdapMoE+SD"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_PREFETCH=1 \
ELMM_ORACLE_PREFETCH=0 \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
ELMM_REBALANCE_INTERVAL=1000 \
BRISKMOE_SACR=0 \
BRISKMOE_ELP=0 \
BRISKMOE_DIPP=0 \
BRISKMOE_PREDCACHE=0 \
BRISKMOE_ADAPMOE=1 \
BRISKMOE_ADAPMOE_GATING=1 \
BRISKMOE_ADAPMOE_CACHE_ALLOC=1 \
BRISKMOE_ADAPMOE_MIN_K=1 \
BRISKMOE_ADAPMOE_MAX_K=3 \
BRISKMOE_ADAPMOE_HORIZON=2 \
BRISKMOE_ADAPMOE_SINGLE_RATIO=0.24 \
BRISKMOE_ADAPMOE_GATING_WARMUP=128 \
	run_bench "adapmoe_sd"

# 4) BriskMoE+SD (SACR + ELP + DIPP)
echo ""; echo ">>> [4/4] BriskMoE+SD (Full)"
VLLM_PLUGINS=elmm \
ELMM_CACHE_GB=$ELMM_CACHE_GB \
ELMM_PREFETCH=1 \
ELMM_ORACLE_PREFETCH=0 \
ELMM_STALE_REMAP=$ELMM_STALE_REMAP \
BRISKMOE_SACR=1 \
BRISKMOE_ELP=1 \
BRISKMOE_DIPP=1 \
BRISKMOE_PREDCACHE=0 \
BRISKMOE_ADAPMOE=0 \
	run_bench "briskmoe_full_sd"

echo ""
echo "================================================================"
echo "[$(timestamp)] Parsing results..."
echo "================================================================"

RESULT_DIR_ENV="$RESULT_DIR" "$PYTHON" << 'PYEOF'
import json
import os

result_dir = os.environ["RESULT_DIR_ENV"]
labels = ["mixtral_offloading_sd", "moe_infinity_sd", "adapmoe_sd", "briskmoe_full_sd"]

results = {}
for label in labels:
	path = os.path.join(result_dir, label, "result.json")
	if not os.path.exists(path):
		print(f"[WARN] Missing {path}")
		continue
	with open(path) as f:
		results[label] = json.load(f)

if len(results) != 4:
	print("[ERROR] Missing results, cannot summarize all 4 configs.")
	raise SystemExit(1)

base = results["mixtral_offloading_sd"]["avg_latency"]
print("\n" + "=" * 90)
print("SP-MoE + BriskMoE Full Reproduction on HumanEval")
print("=" * 90)
print(f"{'Config':<24} {'Latency(s)':>10} {'TPS':>8} {'TPOT(ms)':>10} {'p50(s)':>8} {'p99(s)':>8} {'Speedup':>10}")
print("-" * 90)
for label in labels:
	d = results[label]
	lat = d["avg_latency"]
	tps = d["tps_mean"]
	tpot = d["tpot_mean_ms"]
	p50 = d["percentiles"]["50"]["latency"]
	p99 = d["percentiles"]["99"]["latency"]
	spd = base / lat
	print(f"{label:<24} {lat:>10.3f} {tps:>8.2f} {tpot:>10.2f} {p50:>8.3f} {p99:>8.3f} {spd:>9.2f}x")

PYEOF

echo ""
echo "[$(timestamp)] Done. Results in: $RESULT_DIR"
