#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/cdb/miniforge3/envs/briskmoe/bin/python" ]]; then
    PYTHON_BIN="/home/cdb/miniforge3/envs/briskmoe/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
OUT_DIR="${OUT_DIR:-results/moeinf_official_vllm_smoke}"
MODEL="${MODEL:-$ROOT_DIR/model/Qwen3-30B-A3B-Instruct-2507}"
DATASET="${DATASET:-data/humaneval_bench.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$OUT_DIR"

echo "[bench_moeinf_official_vllm] OUT_DIR=$OUT_DIR"
echo "[bench_moeinf_official_vllm] MODEL=$MODEL"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
PYTHONUNBUFFERED=1 \
PYTHONPATH=. \
VLLM_PLUGINS=moeinf_official \
MOE_INFINITY_OFFICIAL_VLLM=1 \
"$PYTHON_BIN" scripts/bench_humaneval_runner.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --warmup-prompts "$WARMUP_PROMPTS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enforce-eager \
  --trust-remote-code \
  --dtype bfloat16 \
  --output-json "$OUT_DIR/result.json" \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"$ROOT_DIR/model/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3\",\"num_speculative_tokens\":3}" \
  2>&1 | tee "$OUT_DIR/bench.log"

echo "[bench_moeinf_official_vllm] done: $OUT_DIR/result.json"
