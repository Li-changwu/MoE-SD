#!/bin/bash
###############################################################################
# MVP Pilot Experiment: Non-monotonic SD under Memory Pressure
# Issue #28 - Minimal viable experiment
#
# Design:
#   - 2 memory pressure levels: Low (0.90) / High (0.70)
#   - 3 K values: 0 (no SD) / 4 / 8
#   - 1 workload: decode-heavy (short prompt, long generation)
#   - Mode: latency (offline, single batch)
#
# Total: 6 experiments
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR"

# ── Configuration ────────────────────────────────────────────────────────────
MODEL="/home/sage3/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR="/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"

# Fixed across all experiments
CPU_OFFLOAD_GB=30
MAX_MODEL_LEN=2048
SWAP_SPACE=4
SEED=42

# Optimized: batch=1, short output for CPU-offload speed
INPUT_LEN=128
BATCH_SIZE=1
OUTPUT_LEN=128

# Latency benchmark (reduced for speed)
NUM_ITERS=3
NUM_ITERS_WARMUP=1

# Results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/mvp_pilot_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Logging ──────────────────────────────────────────────────────────────────
log_header() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${BLUE}  $1${NC}"; echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════${NC}\n"; }
log_info()   { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()  { echo -e "${RED}[ERROR]${NC} $1"; }
log_exp()    { echo -e "${CYAN}[EXP]${NC} $1"; }

# ── Experiment Matrix ────────────────────────────────────────────────────────
# Format: "label gpu_mem_util K"
EXPERIMENTS=(
    "low_k0   0.90  0"
    "low_k4   0.90  4"
    "low_k8   0.90  8"
    "high_k0  0.70  0"
    "high_k4  0.70  4"
    "high_k8  0.70  8"
)

# ── Pre-flight checks ───────────────────────────────────────────────────────
log_header "MVP Pilot Experiment — Pre-flight"

log_info "Model: $MODEL"
log_info "Speculator: $SPECULATOR"
log_info "GPU memory utilization: Low=0.90, High=0.70"
log_info "K values: 0, 4, 8"
log_info "Workload: decode-heavy (input=$INPUT_LEN, output=$OUTPUT_LEN)"
log_info "cpu_offload_gb=$CPU_OFFLOAD_GB, max_model_len=$MAX_MODEL_LEN"
log_info "Results: $RESULTS_DIR"
echo ""

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log_info "GPU: $GPU_NAME ($GPU_MEM)"
else
    log_warn "nvidia-smi not found"
fi

# Check vllm
if ! command -v vllm &>/dev/null; then
    log_error "vllm not found. Please activate conda environment: conda activate vllm_moe_sd"
    exit 1
fi
log_info "vLLM version: $(vllm --version 2>/dev/null || echo 'unknown')"

# ── Save experiment config ───────────────────────────────────────────────────
cat > "$RESULTS_DIR/experiment_config.json" << EOFCFG
{
    "experiment": "mvp_pilot",
    "timestamp": "$TIMESTAMP",
    "model": "$MODEL",
    "speculator": "$SPECULATOR",
    "cpu_offload_gb": $CPU_OFFLOAD_GB,
    "max_model_len": $MAX_MODEL_LEN,
    "swap_space": $SWAP_SPACE,
    "seed": $SEED,
    "workload": {
        "type": "decode-heavy",
        "input_len": $INPUT_LEN,
        "output_len": $OUTPUT_LEN
    },
    "latency_config": {
        "num_iters": $NUM_ITERS,
        "num_iters_warmup": $NUM_ITERS_WARMUP
    },
    "pressure_levels": {
        "low": 0.90,
        "high": 0.70
    },
    "k_values": [0, 4, 8]
}
EOFCFG
log_info "Config saved to $RESULTS_DIR/experiment_config.json"

# ── Run experiments ──────────────────────────────────────────────────────────
TOTAL=${#EXPERIMENTS[@]}
CURRENT=0
FAILED=0

log_header "Running $TOTAL Experiments"

for exp in "${EXPERIMENTS[@]}"; do
    read -r LABEL GPU_MEM_UTIL K <<< "$exp"
    CURRENT=$((CURRENT + 1))

    log_header "[$CURRENT/$TOTAL] $LABEL  (gpu_mem=$GPU_MEM_UTIL, K=$K)"

    OUT_JSON="$RESULTS_DIR/${LABEL}.json"
    LOG_FILE="$RESULTS_DIR/${LABEL}.log"

    # Build command
    CMD=(
        vllm bench latency
        --model "$MODEL"
        --batch-size "$BATCH_SIZE"
        --input-len "$INPUT_LEN"
        --output-len "$OUTPUT_LEN"
        --num-iters "$NUM_ITERS"
        --num-iters-warmup "$NUM_ITERS_WARMUP"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --cpu-offload-gb "$CPU_OFFLOAD_GB"
        --swap-space "$SWAP_SPACE"
        --seed "$SEED"
        --output-json "$OUT_JSON"
        --enforce-eager
    )

    # Add speculative config if K > 0
    if [[ "$K" -gt 0 ]]; then
        SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$SPECULATOR\", \"num_speculative_tokens\": $K}"
        CMD+=(--speculative-config "$SPEC_CONFIG")
    fi

    log_exp "Command: ${CMD[*]}"
    echo ""

    START_TIME=$(date +%s)

    # Run experiment, allow failure without aborting entire script
    set +e
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [[ $EXIT_CODE -eq 0 ]]; then
        log_info "✅ $LABEL completed in ${ELAPSED}s"

        # Inject metadata into result JSON
        if [[ -f "$OUT_JSON" ]]; then
            python3 << EOFPY
import json
with open('$OUT_JSON', 'r') as f:
    data = json.load(f)
data['_mvp_meta'] = {
    'label': '$LABEL',
    'gpu_memory_utilization': $GPU_MEM_UTIL,
    'K': $K,
    'pressure': 'low' if float('$GPU_MEM_UTIL') > 0.8 else 'high',
    'input_len': $INPUT_LEN,
    'output_len': $OUTPUT_LEN,
    'cpu_offload_gb': $CPU_OFFLOAD_GB,
    'elapsed_seconds': $ELAPSED,
}
with open('$OUT_JSON', 'w') as f:
    json.dump(data, f, indent=2)
print('Metadata injected into $OUT_JSON')
EOFPY
        fi
    else
        log_error "❌ $LABEL FAILED (exit=$EXIT_CODE, ${ELAPSED}s) — see $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
log_header "Experiment Complete"

if [[ $FAILED -eq 0 ]]; then
    log_info "✅ All $TOTAL experiments succeeded"
else
    log_error "❌ $FAILED/$TOTAL experiments failed"
fi

log_info "Results directory: $RESULTS_DIR"
echo ""
log_info "Next step: analyze results:"
log_info "  python3 scripts/analyze_mvp.py --results-dir $RESULTS_DIR"
