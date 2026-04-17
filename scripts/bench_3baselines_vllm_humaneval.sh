#!/usr/bin/env bash
set -euo pipefail

# Three-baseline comparison on HumanEval:
#   1) LRU baseline on ELMM plugin
#   2) MoE-Infinity official-like on pure vLLM plugin
#   3) AdapMoE official-like on pure vLLM plugin

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/cdb/miniforge3/envs/briskmoe/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi

MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/model}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-30B-A3B-Instruct-2507}"
SPEC_MODEL_PATH="${SPEC_MODEL_PATH:-$MODEL_ROOT/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3}"

DATASET="${DATASET:-data/humaneval_bench.jsonl}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-5}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

BASE_OUT="${BASE_OUT:-results}"
RUN_MOEINF_STRIDE16_20X128="${RUN_MOEINF_STRIDE16_20X128:-0}"

# Recorded config (2026-04-17): MoE-Infinity official-like, 20x128, stride=16.
run_moeinf_stride16_20x128() {
  local name="moeinf_official_vllm_stride16_20x128"
  local out_dir="$BASE_OUT/$name"
  mkdir -p "$out_dir"

  echo "============================================================"
  echo "Running $name"
  echo "Output: $out_dir"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=. \
  VLLM_PLUGINS=moeinf_official \
  ELMM_PREFETCH=0 \
  ADAPMOE_ENABLE=0 \
  MOE_INFINITY_ENABLE=0 \
  SPMOE_ENABLE=0 \
  MOE_INFINITY_OFFICIAL_VLLM=1 \
  MOE_INFINITY_PREFETCH_TOPK=4 \
  MOE_INFINITY_PREFETCH_PER_LAYER=1 \
  MOE_INFINITY_PREFETCH_MAX_LAYERS=2 \
  MOE_INFINITY_PREFETCH_SCORE_THRESHOLD=1e-3 \
  MOE_INFINITY_LOG_INTERVAL=0 \
  MOE_INFINITY_TRACE_UPDATE_STRIDE=16 \
  MOE_INFINITY_PREDICT_UPDATE_STRIDE=16 \
  MOE_INFINITY_SHADOW_POLICY_STRIDE=16 \
  "$PYTHON_BIN" scripts/bench_humaneval_runner.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-len 128 \
    --num-prompts 20 \
    --warmup-prompts 3 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --trust-remote-code \
    --dtype bfloat16 \
    --output-json "$out_dir/result.json" \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$SPEC_MODEL_PATH\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
    > "$out_dir/bench.log" 2>&1
}

run_case() {
  local name="$1"
  shift
  local out_dir="$BASE_OUT/$name"
  mkdir -p "$out_dir"

  echo "============================================================"
  echo "Running $name"
  echo "Output: $out_dir"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=. \
  VLLM_PLUGINS=elmm \
  ELMM_CACHE_GB=8 \
  ELMM_RWAWE=0 \
  ELMM_STALE_REMAP=0 \
  ELMM_ORACLE_PREFETCH=0 \
  ELMM_STACKED_GATING=0 \
  ELMM_HFDE=0 \
  ELMM_SHARED_PARALLEL=0 \
  ELMM_DRAFT_UTILITY=0 \
  ELMM_ENTROPY_BUDGET=0 \
  ELMM_ADAPTIVE_BUDGET=0 \
  ELMM_PREFETCH=0 \
  ELMM_LOCALITY=0 \
  ELMM_OVERFLOW_CTRL=0 \
  ELMM_UNIFIED=0 \
  ELMM_DIRECT_DISPATCH=0 \
  BRISKMOE_SACR=0 \
  BRISKMOE_ELP=0 \
  BRISKMOE_DIPP=0 \
  BRISKMOE_PREDCACHE=0 \
  SPMOE_ENABLE=0 \
  ADAPMOE_ENABLE=0 \
  MOE_INFINITY_ENABLE=0 \
  "$@" \
  "$PYTHON_BIN" scripts/bench_humaneval_runner.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --warmup-prompts "$WARMUP_PROMPTS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --trust-remote-code \
    --dtype bfloat16 \
    --output-json "$out_dir/result.json" \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$SPEC_MODEL_PATH\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
    2>&1 | tee "$out_dir/bench.log"
}

# Baseline 1: Pure LRU (ELMM plugin)
run_case "baseline_lru_offload45"

# Baseline 2: MoE-Infinity official-like on pure vLLM
run_case "moeinf_official_vllm" \
  env \
  VLLM_PLUGINS=moeinf_official \
  MOE_INFINITY_OFFICIAL_VLLM=1 \
  MOE_INFINITY_PREFETCH_TOPK=4 \
  MOE_INFINITY_PREFETCH_PER_LAYER=1 \
  MOE_INFINITY_PREFETCH_MAX_LAYERS=2 \
  MOE_INFINITY_PREFETCH_SCORE_THRESHOLD=1e-3

# Baseline 3: AdapMoE official-like on pure vLLM
# NOTE: "adopmoe" in request is treated as "adapmoe".
run_case "adapmoe_official_vllm" \
  env \
  VLLM_PLUGINS=adapmoe_official \
  ADAPMOE_OFFICIAL_VLLM=1 \
  ADAPMOE_DP_ENABLE=1 \
  ADAPMOE_ADAPTGATE=1 \
  ADAPMOE_PREFETCH_HORIZON=1 \
  ADAPMOE_PREFETCH_TOPK=2 \
  ADAPMOE_THRESHOLD_BASE=0.005

if [[ "$RUN_MOEINF_STRIDE16_20X128" == "1" ]]; then
  run_moeinf_stride16_20x128
fi

"$PYTHON_BIN" - << 'PY'
import json
import os
from pathlib import Path

base = Path(os.environ.get("BASE_OUT", "results"))
names = [
    "baseline_lru_offload45",
    "moeinf_official_vllm",
    "adapmoe_official_vllm",
]
rows = []
for name in names:
    p = base / name / "result.json"
    if not p.exists():
        continue
    obj = json.loads(p.read_text())
    rows.append((
        name,
        obj.get("avg_latency", 0.0),
        obj.get("percentiles", {}).get("99", {}).get("latency", 0.0),
        obj.get("tps_mean", 0.0),
    ))

print("\n==================== 3-Baseline Summary ====================")
print(f"{'Config':<28} {'Avg Lat(s)':>12} {'P99(s)':>10} {'TPS':>10}")
for name, avg, p99, tps in rows:
    print(f"{name:<28} {avg:>12.3f} {p99:>10.3f} {tps:>10.2f}")
print("============================================================")
PY
