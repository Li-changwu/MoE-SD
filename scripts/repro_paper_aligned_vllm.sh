#!/usr/bin/env bash
# =============================================================================
# Paper-aligned vLLM Reproduction Template (one-click)
#
# Goal:
#   Reproduce method curves with a paper-like protocol:
#   - multi-dataset
#   - multi-method
#   - multi-run averaging
#   - automatic aggregation and plotting
#
# Notes:
#   1) This is a TEMPLATE aligned to current repository implementation.
#   2) Default datasets are those that already exist in this repo.
#   3) Missing datasets are skipped automatically.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_PYTHON_CANDIDATES=(
  "/home/cdb/miniforge3/envs/briskmoe/bin/python"
  "/opt/miniconda3/envs/moe-sd/bin/python"
)

if [[ -z "${PYTHON:-}" ]]; then
  for candidate in "${DEFAULT_PYTHON_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
      PYTHON="$candidate"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python3}"

RUNNER="${RUNNER:-$REPO_ROOT/scripts/bench_humaneval_runner.py}"
MODEL="${MODEL:-$REPO_ROOT/model/Qwen3-30B-A3B-Instruct-2507}"
SPECULATOR="${SPECULATOR:-$REPO_ROOT/model/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3}"

# ---- Paper-like protocol knobs ----
NUM_RUNS="${NUM_RUNS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
CPU_OFFLOAD_AR="${CPU_OFFLOAD_AR:-30}"
CPU_OFFLOAD_SD="${CPU_OFFLOAD_SD:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
ELMM_CACHE_GB="${ELMM_CACHE_GB:-8}"
ELMM_STALE_REMAP="${ELMM_STALE_REMAP:-4}"

# Datasets: comma-separated name=path
# You can add missing paper datasets here; missing files are skipped.
DATASETS="${DATASETS:-humaneval=$REPO_ROOT/data/humaneval_bench.jsonl,gsm8k=$REPO_ROOT/data/gsm8k_bench.jsonl}"

# Methods to run (comma-separated)
# Supported: ar_baseline, sd_lru, sd_moe_infinity, sd_adapmoe, sd_briskmoe
METHODS="${METHODS:-ar_baseline,sd_lru,sd_moe_infinity,sd_adapmoe,sd_briskmoe}"

RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/paper_aligned_repro_$(date +%Y%m%d_%H%M%S)}"
SPEC_CONFIG='{"method":"eagle3","model":"'"$SPECULATOR"'","num_speculative_tokens":3}'

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

pick_gpu() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local best_gpu
    best_gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t',' -k2 -nr \
      | head -n1 \
      | cut -d',' -f1 \
      | tr -d '[:space:]')"
    if [[ -n "$best_gpu" ]]; then
      export CUDA_VISIBLE_DEVICES="$best_gpu"
      echo "[$(timestamp)] Auto-selected GPU index: $CUDA_VISIBLE_DEVICES"
    fi
  fi
}

run_one() {
  local dataset_name="$1"
  local dataset_path="$2"
  local method="$3"
  local run_id="$4"

  local outdir="$RESULT_ROOT/$dataset_name/$method/run_$run_id"
  mkdir -p "$outdir"

  local offload="$CPU_OFFLOAD_SD"
  local plugin="elmm"
  local spec_args=(--speculative-config "$SPEC_CONFIG")

  # default method flags
  local -a envs=(
    "VLLM_PLUGINS=$plugin"
    "ELMM_CACHE_GB=$ELMM_CACHE_GB"
    "ELMM_STALE_REMAP=$ELMM_STALE_REMAP"
    "ELMM_PREFETCH=1"
    "ELMM_ORACLE_PREFETCH=0"
    "BRISKMOE_SACR=0"
    "BRISKMOE_ELP=0"
    "BRISKMOE_DIPP=0"
    "BRISKMOE_PREDCACHE=0"
    "BRISKMOE_ADAPMOE=0"
    "BRISKMOE_ADAPMOE_GATING=0"
    "BRISKMOE_ADAPMOE_CACHE_ALLOC=0"
  )

  case "$method" in
    ar_baseline)
      offload="$CPU_OFFLOAD_AR"
      envs=("VLLM_PLUGINS=")
      spec_args=()
      ;;

    sd_lru)
      # SD + ELMM baseline (LRU-like default cache behavior)
      envs+=(
        "ELMM_DRAFT_HOOK=0"
      )
      ;;

    sd_moe_infinity)
      # Native MoE-Infinity path on current vLLM plugin:
      # request-level historical sequence trace predictor.
      envs+=(
        "BRISKMOE_MOE_INFINITY=1"
        "BRISKMOE_MOE_INFINITY_HORIZON=2"
        "BRISKMOE_MOE_INFINITY_HISTORY=64"
        "BRISKMOE_MOE_INFINITY_MAX_K=4"
        "BRISKMOE_MOE_INFINITY_MIN_SIM=0.05"
        "BRISKMOE_ADAPMOE=0"
        "BRISKMOE_ADAPMOE_GATING=0"
        "BRISKMOE_ADAPMOE_CACHE_ALLOC=0"
      )
      ;;

    sd_adapmoe)
      # AdapMoE on vLLM (gating + DP cache alloc + adaptive prefetch)
      envs+=(
        "BRISKMOE_ADAPMOE=1"
        "BRISKMOE_ADAPMOE_GATING=1"
        "BRISKMOE_ADAPMOE_CACHE_ALLOC=1"
        "BRISKMOE_ADAPMOE_MIN_K=1"
        "BRISKMOE_ADAPMOE_MAX_K=3"
        "BRISKMOE_ADAPMOE_HORIZON=2"
        "BRISKMOE_ADAPMOE_SINGLE_RATIO=0.24"
        "BRISKMOE_ADAPMOE_GATING_WARMUP=128"
      )
      ;;

    sd_briskmoe)
      # Full BriskMoE stack
      envs+=(
        "BRISKMOE_SACR=1"
        "BRISKMOE_ELP=1"
        "BRISKMOE_DIPP=1"
      )
      ;;

    *)
      echo "[$(timestamp)] [ERROR] Unknown method: $method"
      return 1
      ;;
  esac

  echo "[$(timestamp)] Run dataset=$dataset_name method=$method repeat=$run_id"
  echo "[$(timestamp)] Output: $outdir"

  # Keep each run clean without killing this launcher itself.
  # The old pattern `pkill -f vllm` matched this script name.
  pkill -f "vllm\.entrypoints|api_server\.py|vllm serve" 2>/dev/null || true
  sleep 2

  env "${envs[@]}" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$RUNNER" \
    --model "$MODEL" \
    --dataset "$dataset_path" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --warmup-prompts "$WARMUP_PROMPTS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --cpu-offload-gb "$offload" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --output-json "$outdir/result.json" \
    "${spec_args[@]}" \
    2>&1 | tee "$outdir/bench.log"
}

mkdir -p "$RESULT_ROOT"
pick_gpu

echo "[$(timestamp)] Python: $PYTHON"
echo "[$(timestamp)] Result root: $RESULT_ROOT"
echo "[$(timestamp)] Model: $MODEL"
echo "[$(timestamp)] Speculator: $SPECULATOR"

IFS=',' read -r -a dataset_items <<< "$DATASETS"
IFS=',' read -r -a method_items <<< "$METHODS"

for ds in "${dataset_items[@]}"; do
  name="${ds%%=*}"
  path="${ds#*=}"

  if [[ ! -f "$path" ]]; then
    echo "[$(timestamp)] [WARN] Dataset missing, skip: $name -> $path"
    continue
  fi

  for method in "${method_items[@]}"; do
    for run_id in $(seq 1 "$NUM_RUNS"); do
      run_one "$name" "$path" "$method" "$run_id"
    done
  done
done

# -----------------------------------------------------------------------------
# Aggregate results and generate curve files.
# -----------------------------------------------------------------------------
RESULT_ROOT_ENV="$RESULT_ROOT" "$PYTHON" - << 'PYEOF'
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["RESULT_ROOT_ENV"])
rows = []

for result in root.glob("*/*/run_*/result.json"):
    dataset = result.parents[2].name
    method = result.parents[1].name
    run_name = result.parents[0].name
    run_id = int(run_name.split("_")[-1])
    d = json.loads(result.read_text())
    rows.append({
        "dataset": dataset,
        "method": method,
        "run_id": run_id,
        "avg_latency": float(d["avg_latency"]),
        "p99_latency": float(d["percentiles"]["99"]["latency"]),
        "tpot_ms": float(d["tpot_mean_ms"]),
        "tps_mean": float(d["tps_mean"]),
    })

if not rows:
    raise SystemExit("No result.json found; check logs.")

# per-run csv
with (root / "raw_runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda x: (x["dataset"], x["method"], x["run_id"])))

# aggregate mean/std + speedup vs ar_baseline
group = defaultdict(list)
for r in rows:
    group[(r["dataset"], r["method"])].append(r)

agg = []
for (dataset, method), vals in group.items():
    def mean(k):
        return sum(v[k] for v in vals) / len(vals)

    def std(k):
        m = mean(k)
        return math.sqrt(sum((v[k] - m) ** 2 for v in vals) / len(vals))

    agg.append({
        "dataset": dataset,
        "method": method,
        "runs": len(vals),
        "avg_latency_mean": mean("avg_latency"),
        "avg_latency_std": std("avg_latency"),
        "p99_latency_mean": mean("p99_latency"),
        "tpot_ms_mean": mean("tpot_ms"),
        "tps_mean": mean("tps_mean"),
        "tps_std": std("tps_mean"),
    })

# speedup field
ar_map = {
    a["dataset"]: a["tps_mean"]
    for a in agg
    if a["method"] == "ar_baseline"
}
for a in agg:
    base = ar_map.get(a["dataset"], 0.0)
    a["speedup_vs_ar"] = (a["tps_mean"] / base) if base > 0 else 0.0

agg = sorted(agg, key=lambda x: (x["dataset"], x["method"]))

with (root / "summary_agg.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
    w.writeheader()
    w.writerows(agg)

(root / "summary_agg.json").write_text(json.dumps(agg, indent=2))

# markdown quick table
lines = []
lines.append("# Paper-aligned Reproduction Summary")
lines.append("")
lines.append("| dataset | method | runs | tps_mean | speedup_vs_ar | avg_latency_mean(s) | p99_latency_mean(s) |")
lines.append("|---|---|---:|---:|---:|---:|---:|")
for a in agg:
    lines.append(
        f"| {a['dataset']} | {a['method']} | {a['runs']} | {a['tps_mean']:.3f} | {a['speedup_vs_ar']:.3f} | {a['avg_latency_mean']:.3f} | {a['p99_latency_mean']:.3f} |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n")

# optional plotting
try:
    import matplotlib.pyplot as plt

    by_dataset = defaultdict(list)
    for a in agg:
        by_dataset[a["dataset"]].append(a)

    for dataset, vals in by_dataset.items():
        vals = sorted(vals, key=lambda x: x["speedup_vs_ar"])
        names = [v["method"] for v in vals]
        speedups = [v["speedup_vs_ar"] for v in vals]

        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.bar(names, speedups, color=["#5B8FF9", "#5AD8A6", "#5D7092", "#F6BD16", "#E8684A"][: len(names)])
        ax.set_title(f"Speedup vs AR ({dataset})")
        ax.set_ylabel("Speedup")
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(speedups):
            ax.text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(root / f"curve_speedup_{dataset}.png", dpi=160)
        plt.close(fig)
except Exception as e:
    (root / "plot_error.log").write_text(f"Plot skipped: {e}\n")

print("Aggregation done:")
print(f"  - {root / 'raw_runs.csv'}")
print(f"  - {root / 'summary_agg.csv'}")
print(f"  - {root / 'summary_agg.json'}")
print(f"  - {root / 'summary.md'}")
PYEOF

echo "[$(timestamp)] Done. Results in: $RESULT_ROOT"
echo "[$(timestamp)] Next: open $RESULT_ROOT/summary.md"
