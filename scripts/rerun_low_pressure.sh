#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR"

MODEL="/home/sage3/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"

CPU_OFFLOAD_GB=30
MAX_MODEL_LEN=2048
SWAP_SPACE=4
SEED=42
BATCH_SIZE=1
INPUT_LEN=128
OUTPUT_LEN=128
NUM_ITERS=3
NUM_ITERS_WARMUP=1
GPU_MEM=0.90
TIMEOUT=600

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/rerun_low_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR" logs

log()  { echo -e "\033[0;32m[INFO]\033[0m $1"; }
err()  { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
hdr()  { echo -e "\n\033[1;36m══ $1 ══\033[0m\n"; }

EXPERIMENTS=("0" "4" "8")
TOTAL=${#EXPERIMENTS[@]}
IDX=0
FAILED=0

hdr "Rerun LOW pressure (gpu_mem=$GPU_MEM) — $(date)"
log "Results -> $RESULTS_DIR"
log "Timeout: ${TIMEOUT}s per experiment"

for K in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    LABEL="low_k${K}"
    hdr "[$IDX/$TOTAL] $LABEL  (gpu_mem=$GPU_MEM, K=$K)"

    OUT_JSON="$RESULTS_DIR/${LABEL}.json"
    LOG_FILE="$RESULTS_DIR/${LABEL}.log"

    CMD=(
        vllm bench latency
        --model "$MODEL"
        --batch-size "$BATCH_SIZE"
        --input-len "$INPUT_LEN"
        --output-len "$OUTPUT_LEN"
        --num-iters "$NUM_ITERS"
        --num-iters-warmup "$NUM_ITERS_WARMUP"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM"
        --cpu-offload-gb "$CPU_OFFLOAD_GB"
        --swap-space "$SWAP_SPACE"
        --seed "$SEED"
        --output-json "$OUT_JSON"
        --enforce-eager
    )

    if [[ "$K" -gt 0 ]]; then
        SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$SPECULATOR\", \"num_speculative_tokens\": $K}"
        CMD+=(--speculative-config "$SPEC_CONFIG")
    fi

    log "Command: ${CMD[*]}"

    START_TIME=$(date +%s)
    set +e
    timeout "$TIMEOUT" bash -c '"$@" 2>&1 | tee "$0"' "$LOG_FILE" "${CMD[@]}"
    EXIT_CODE=$?
    set -e
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [[ $EXIT_CODE -eq 124 ]]; then
        err "❌ $LABEL TIMEOUT after ${TIMEOUT}s"
        FAILED=$((FAILED + 1))
    elif [[ $EXIT_CODE -eq 0 && -f "$OUT_JSON" ]]; then
        log "✅ $LABEL completed in ${ELAPSED}s"
        python3 -c "
import json
with open('$OUT_JSON', 'r') as f:
    data = json.load(f)
data['_mvp_meta'] = {
    'label': '$LABEL',
    'gpu_memory_utilization': $GPU_MEM,
    'K': $K,
    'pressure': 'low',
    'input_len': $INPUT_LEN,
    'output_len': $OUTPUT_LEN,
    'batch_size': $BATCH_SIZE,
    'cpu_offload_gb': $CPU_OFFLOAD_GB,
    'elapsed_seconds': $ELAPSED,
}
with open('$OUT_JSON', 'w') as f:
    json.dump(data, f, indent=2)
"
    else
        err "❌ $LABEL FAILED (exit=$EXIT_CODE, ${ELAPSED}s) — see $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    log "Waiting 10s for GPU cleanup..."
    sleep 10
    echo ""
done

hdr "Summary"
echo ""
log "Results: $RESULTS_DIR"
for K in "${EXPERIMENTS[@]}"; do
    LABEL="low_k${K}"
    JSON="$RESULTS_DIR/${LABEL}.json"
    if [[ -f "$JSON" ]]; then
        LAT=$(python3 -c "import json; print(f'{json.load(open(\"$JSON\"))[\"avg_latency\"]:.2f}s')" 2>/dev/null || echo "parse error")
        log "  ✅ $LABEL  avg_latency=$LAT"
    else
        err "  ❌ $LABEL  NO RESULT"
    fi
done

if [[ $FAILED -eq 0 ]]; then
    log "All $TOTAL experiments succeeded"
else
    err "$FAILED/$TOTAL experiments failed"
fi
