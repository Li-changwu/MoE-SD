#!/usr/bin/env bash
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
DATASET="${DATASET:-$REPO_ROOT/data/humaneval_bench.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/adapmoe_sweep_$(date +%Y%m%d_%H%M%S)}"

OUTPUT_LEN="${OUTPUT_LEN:-100}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CPU_OFFLOAD_ELMM="${CPU_OFFLOAD_ELMM:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
ELMM_CACHE_GB="${ELMM_CACHE_GB:-8}"
ELMM_STALE_REMAP="${ELMM_STALE_REMAP:-4}"

# stage-1 coarse screening
COARSE_NUM_PROMPTS="${COARSE_NUM_PROMPTS:-10}"
COARSE_WARMUP_PROMPTS="${COARSE_WARMUP_PROMPTS:-2}"
# stage-2 final validation
FINAL_NUM_PROMPTS="${FINAL_NUM_PROMPTS:-30}"
FINAL_WARMUP_PROMPTS="${FINAL_WARMUP_PROMPTS:-4}"

# stage-3 full confirmation for best final config
CONFIRM_NUM_PROMPTS="${CONFIRM_NUM_PROMPTS:-50}"
CONFIRM_WARMUP_PROMPTS="${CONFIRM_WARMUP_PROMPTS:-5}"

TOPK_FINAL="${TOPK_FINAL:-1}"

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
            echo "[$(timestamp)] Auto-selected GPU index: $CUDA_VISIBLE_DEVICES (max free memory)"
        fi
    fi
}

run_one() {
    local stage="$1"
    local label="$2"
    local single_ratio="$3"
    local max_k="$4"
    local horizon="$5"
    local warmup="$6"
    local rebalance="$7"
    local num_prompts="$8"
    local warmup_prompts="$9"

    local outdir="$RESULT_ROOT/$stage/$label"
    mkdir -p "$outdir"

    echo "[$(timestamp)] Stage=$stage label=$label sr=$single_ratio max_k=$max_k horizon=$horizon gating_warmup=$warmup rebalance=$rebalance"

    PYTHONUNBUFFERED=1 \
    VLLM_PLUGINS=elmm \
    ELMM_CACHE_GB="$ELMM_CACHE_GB" \
    ELMM_PREFETCH=1 \
    ELMM_ORACLE_PREFETCH=0 \
    ELMM_STALE_REMAP="$ELMM_STALE_REMAP" \
    ELMM_REBALANCE_INTERVAL="$rebalance" \
    BRISKMOE_SACR=0 \
    BRISKMOE_ELP=0 \
    BRISKMOE_DIPP=0 \
    BRISKMOE_PREDCACHE=0 \
    BRISKMOE_ADAPMOE=1 \
    BRISKMOE_ADAPMOE_GATING=1 \
    BRISKMOE_ADAPMOE_CACHE_ALLOC=1 \
    BRISKMOE_ADAPMOE_MIN_K=1 \
    BRISKMOE_ADAPMOE_MAX_K="$max_k" \
    BRISKMOE_ADAPMOE_HORIZON="$horizon" \
    BRISKMOE_ADAPMOE_SINGLE_RATIO="$single_ratio" \
    BRISKMOE_ADAPMOE_GATING_WARMUP="$warmup" \
    "$PYTHON" "$RUNNER" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output-len "$OUTPUT_LEN" \
        --num-prompts "$num_prompts" \
        --warmup-prompts "$warmup_prompts" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --cpu-offload-gb "$CPU_OFFLOAD_ELMM" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        --trust-remote-code \
        --dtype bfloat16 \
        --speculative-config "$SPEC_CONFIG" \
        --output-json "$outdir/result.json" \
        2>&1 | tee "$outdir/bench.log"
}

mkdir -p "$RESULT_ROOT"
pick_gpu

echo "[$(timestamp)] Result root: $RESULT_ROOT"
echo "[$(timestamp)] Python: $PYTHON"

# Keep environment clean before sweep.
pkill -f "vllm" 2>/dev/null || true
sleep 2

# Coarse grid: 6 points.
coarse_cfgs=(
  "sr014_k2_h2_w64_r256 0.14 2 2 64 256"
  "sr014_k3_h2_w64_r256 0.14 3 2 64 256"
  "sr020_k2_h2_w64_r256 0.20 2 2 64 256"
  "sr020_k3_h2_w64_r256 0.20 3 2 64 256"
  "sr026_k2_h2_w64_r256 0.26 2 2 64 256"
  "sr026_k3_h2_w64_r256 0.26 3 2 64 256"
)

for item in "${coarse_cfgs[@]}"; do
    read -r label sr mk hz gw rb <<< "$item"
    run_one "coarse" "$label" "$sr" "$mk" "$hz" "$gw" "$rb" "$COARSE_NUM_PROMPTS" "$COARSE_WARMUP_PROMPTS"
    pkill -f "vllm" 2>/dev/null || true
    sleep 2
done

# Select top-k from coarse by avg_latency.
TOP_JSON="$RESULT_ROOT/topk_from_coarse.json"
RESULT_ROOT_ENV="$RESULT_ROOT" TOPK_FINAL_ENV="$TOPK_FINAL" "$PYTHON" - << 'PYEOF'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT_ENV"])
topk = int(os.environ["TOPK_FINAL_ENV"])
rows = []
for result in (root / "coarse").glob("*/result.json"):
    d = json.loads(result.read_text())
    name = result.parent.name
    rows.append({
        "label": name,
        "avg_latency": float(d["avg_latency"]),
        "p99": float(d["percentiles"]["99"]["latency"]),
        "tps": float(d["tps_mean"]),
    })
rows.sort(key=lambda x: (x["avg_latency"], x["p99"], -x["tps"]))
out = rows[:topk]
(root / "topk_from_coarse.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PYEOF

mapfile -t finalists < <(RESULT_ROOT_ENV="$RESULT_ROOT" "$PYTHON" - << 'PYEOF'
import json
import os
from pathlib import Path
p = Path(os.environ["RESULT_ROOT_ENV"]) / "topk_from_coarse.json"
rows = json.loads(p.read_text())
for r in rows:
    print(r["label"])
PYEOF
)

# Expansion around finalist: compact local search.
for base in "${finalists[@]}"; do
    IFS='_' read -r srpart kpart hpart wpart rpart <<< "$base"
    sr="${srpart#sr}"; sr="${sr:0:1}.${sr:1}"
    k="${kpart#k}"

    final_cfgs=(
      "hz1_rb128_gw64 1 128 64"
      "hz2_rb128_gw64 2 128 64"
      "hz1_rb256_gw128 1 256 128"
      "hz2_rb256_gw128 2 256 128"
    )
    for cfg in "${final_cfgs[@]}"; do
        read -r suffix hz rb gw <<< "$cfg"
        label="${base}_${suffix}"
        run_one "final" "$label" "$sr" "$k" "$hz" "$gw" "$rb" "$FINAL_NUM_PROMPTS" "$FINAL_WARMUP_PROMPTS"
        pkill -f "vllm" 2>/dev/null || true
        sleep 2
    done
done

# Summarize final stage and emit best config env file.
RESULT_ROOT_ENV="$RESULT_ROOT" "$PYTHON" - << 'PYEOF'
import json
import os
from pathlib import Path
import re

root = Path(os.environ["RESULT_ROOT_ENV"])
rows = []
for result in (root / "final").glob("*/result.json"):
    d = json.loads(result.read_text())
    label = result.parent.name
    m = re.search(r"sr(\d{3})_k(\d+).*_hz(\d+)_rb(\d+)_gw(\d+)$", label)
    if not m:
        continue
    sr3, k, hz, rb, gw = m.groups()
    sr = float(f"{sr3[0]}.{sr3[1:]}")
    rows.append({
        "label": label,
        "single_ratio": sr,
        "max_k": int(k),
        "horizon": int(hz),
        "rebalance": int(rb),
        "gating_warmup": int(gw),
        "avg_latency": float(d["avg_latency"]),
        "p99": float(d["percentiles"]["99"]["latency"]),
        "tps": float(d["tps_mean"]),
        "tpot_ms": float(d["tpot_mean_ms"]),
    })

if not rows:
    raise SystemExit("no final results found")

rows.sort(key=lambda x: (x["avg_latency"], x["p99"], -x["tps"]))
(root / "sweep_summary_final.json").write_text(json.dumps(rows, indent=2))

best = rows[0]
env_path = root / "best_adapmoe.env"
env_path.write_text(
    "\n".join([
        "BRISKMOE_ADAPMOE=1",
        "BRISKMOE_ADAPMOE_GATING=1",
        "BRISKMOE_ADAPMOE_CACHE_ALLOC=1",
        f"BRISKMOE_ADAPMOE_SINGLE_RATIO={best['single_ratio']}",
        "BRISKMOE_ADAPMOE_MIN_K=1",
        f"BRISKMOE_ADAPMOE_MAX_K={best['max_k']}",
        f"BRISKMOE_ADAPMOE_HORIZON={best['horizon']}",
        f"BRISKMOE_ADAPMOE_GATING_WARMUP={best['gating_warmup']}",
        f"ELMM_REBALANCE_INTERVAL={best['rebalance']}",
    ]) + "\n"
)

print("=" * 96)
print("AdapMoE sweep final ranking (top 10)")
print("=" * 96)
print(f"{'rank':<5}{'label':<44}{'latency':>10}{'p99':>10}{'tps':>10}{'tpot(ms)':>12}")
for i, r in enumerate(rows[:10], 1):
    print(f"{i:<5}{r['label']:<44}{r['avg_latency']:>10.3f}{r['p99']:>10.3f}{r['tps']:>10.2f}{r['tpot_ms']:>12.2f}")
print("=" * 96)
print("Best config:")
print(json.dumps(best, indent=2))
print(f"Best env file: {env_path}")
PYEOF

# Full-prompt confirmation on the selected best final config.
BEST_LABEL="$(RESULT_ROOT_ENV="$RESULT_ROOT" "$PYTHON" - << 'PYEOF'
import json
import os
from pathlib import Path
rows = json.loads((Path(os.environ["RESULT_ROOT_ENV"]) / "sweep_summary_final.json").read_text())
print(rows[0]["label"])
PYEOF
)"

IFS='_' read -r srpart kpart _ <<< "$BEST_LABEL"
best_sr="${srpart#sr}"; best_sr="${best_sr:0:1}.${best_sr:1}"
best_k="${kpart#k}"
best_hz="$(echo "$BEST_LABEL" | sed -n 's/.*_hz\([0-9]\+\)_rb.*/\1/p')"
best_rb="$(echo "$BEST_LABEL" | sed -n 's/.*_rb\([0-9]\+\)_gw.*/\1/p')"
best_gw="$(echo "$BEST_LABEL" | sed -n 's/.*_gw\([0-9]\+\)$/\1/p')"

run_one "confirm" "best_confirm_50" "$best_sr" "$best_k" "$best_hz" "$best_gw" "$best_rb" "$CONFIRM_NUM_PROMPTS" "$CONFIRM_WARMUP_PROMPTS"
pkill -f "vllm" 2>/dev/null || true
sleep 2

RESULT_ROOT_ENV="$RESULT_ROOT" "$PYTHON" - << 'PYEOF'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT_ENV"])
final_rows = json.loads((root / "sweep_summary_final.json").read_text())
confirm = json.loads((root / "confirm" / "best_confirm_50" / "result.json").read_text())
report = {
    "best_final_candidate": final_rows[0],
    "confirm_50": {
        "avg_latency": float(confirm["avg_latency"]),
        "p99": float(confirm["percentiles"]["99"]["latency"]),
        "tps": float(confirm["tps_mean"]),
        "tpot_ms": float(confirm["tpot_mean_ms"]),
    },
}
(root / "best_report.json").write_text(json.dumps(report, indent=2))
print("=" * 96)
print("Best config confirmed on 50 prompts")
print("=" * 96)
print(json.dumps(report, indent=2))
PYEOF

echo "[$(timestamp)] Sweep done. Outputs in: $RESULT_ROOT"
