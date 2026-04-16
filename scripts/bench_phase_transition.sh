#!/usr/bin/env bash
# ===========================================================================
# BriskMoE Phase Transition Experiment
# ---------------------------------------------------------------------------
# Sweeps cpu_offload_gb to show that Speculative Decoding (SD) effectiveness
# degrades — and eventually inverts — as MoE model offloading increases.
#
# X-axis: cpu_offload_gb  (higher → more weight on CPU → more PCIe traffic)
# Y-axis: Throughput (tokens/s)
#
# For each offload level we run:
#   1) AR  – autoregressive baseline (VLLM_PLUGINS="" → no ELMM cache)
#   2) SD  – speculative decoding with EAGLE-3 (VLLM_PLUGINS="" → no ELMM)
#
# PCIe throughput is captured via nvidia-smi dmon in background.
# Results go to: results/phase_transition/{ar,sd}_offload_<GB>/
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
PROJECT_ROOT="$(pwd)"

# ======================== Configuration ========================
PYTHON_BIN="${PYTHON_BIN:-/home/cdb/miniforge3/envs/briskmoe/bin/python}"
MODEL="${MODEL:-${PROJECT_ROOT}/model/Qwen3-30B-A3B-Instruct-2507}"
DRAFT_MODEL="${DRAFT_MODEL:-${PROJECT_ROOT}/model/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3}"
DATASET="${DATASET:-${PROJECT_ROOT}/data/humaneval_bench.jsonl}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results/phase_transition}"
GPU_ID="${GPU_ID:-1}"

# Benchmark parameters
OUTPUT_LEN=128
NUM_PROMPTS=10
WARMUP=3
GPU_MEM=0.90              # KEEP CONSTANT — not a sweep variable!
MAX_MODEL_LEN=4096
DTYPE="bfloat16"
SPEC_DEPTH=3              # EAGLE-3 speculative tokens (K)

# ---------------------------------------------------------------------------
# Experiment design note:
#   Independent variable  = cpu_offload_gb  (precise control of offloading)
#   Control variable      = gpu_memory_utilization (kept FIXED at 0.90)
#
#   Why NOT sweep gpu_memory_utilization?
#     Because it also controls KV-cache budget — a confounding variable.
#     cpu_offload_gb only moves model weights to CPU; KV-cache is unaffected.
#
#   A100 80 GB: model (57 GB) fits entirely → sweep 0..45 (full range)
#   A6000 48 GB: must offload ≥ 15 GB → sweep 15..45
# ---------------------------------------------------------------------------
# Sweep: cpu_offload_gb values (ascending = more offloading)
# On A100 80 GB, start from 0 (everything on GPU = ideal baseline)
OFFLOAD_VALUES=(0 5 10 15 20 25 30 35 40 45)

# Safety
TIMEOUT=3600             # max seconds per single benchmark run

# ======================== Helpers ========================
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] PYTHON_BIN not found or not executable: $PYTHON_BIN"
    exit 1
fi
if [[ ! -d "$MODEL" ]]; then
    echo "[ERROR] MODEL directory not found: $MODEL"
    exit 1
fi
if [[ ! -d "$DRAFT_MODEL" ]]; then
    echo "[ERROR] DRAFT_MODEL directory not found: $DRAFT_MODEL"
    exit 1
fi
if [[ ! -f "$DATASET" ]]; then
    echo "[ERROR] DATASET file not found: $DATASET"
    exit 1
fi

export RESULT_DIR
mkdir -p "$RESULT_DIR"

SPEC_JSON="{\"method\":\"eagle3\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${SPEC_DEPTH}}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

TOTAL_RUNS=$(( ${#OFFLOAD_VALUES[@]} * 2 ))
COMPLETED_RUNS=0
SUCCESS_RUNS=0
SKIPPED_RUNS=0
FAILED_RUNS=0
RUN_STATUS="unknown"

render_progress() {
    local width=32
    local pct=0
    if (( TOTAL_RUNS > 0 )); then
        pct=$(( COMPLETED_RUNS * 100 / TOTAL_RUNS ))
    fi
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    local bar_full bar_empty
    bar_full=$(printf "%${filled}s" "" | tr ' ' '=')
    bar_empty=$(printf "%${empty}s" "" | tr ' ' '.')
    echo "[${bar_full}${bar_empty}] ${pct}% (${COMPLETED_RUNS}/${TOTAL_RUNS}) ok=${SUCCESS_RUNS} skip=${SKIPPED_RUNS} fail=${FAILED_RUNS}"
}

update_progress() {
    local status=$1 tag=$2 offload=$3
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
    case "$status" in
        success) SUCCESS_RUNS=$((SUCCESS_RUNS + 1)) ;;
        skip) SKIPPED_RUNS=$((SKIPPED_RUNS + 1)) ;;
        fail) FAILED_RUNS=$((FAILED_RUNS + 1)) ;;
        *) ;;
    esac
    log "PROG  ${tag} offload=${offload}GB status=${status}"
    render_progress
}

# run_bench <tag> <offload_gb> <spec_json_or_empty>
run_bench() {
    local tag=$1 offload=$2 spec_arg="${3:-}"
    local outdir="${RESULT_DIR}/${tag}_offload_${offload}"
    mkdir -p "$outdir"

    # ---- Resume: skip if result already exists ----
    if [[ -f "$outdir/result.json" ]]; then
        RUN_STATUS="skip"
        log "SKIP  ${tag}  offload=${offload}GB  (result.json exists)"
        return 0
    fi

    log "START ${tag}  offload=${offload}GB ..."

    # ---- Build command ----
    local cmd=(
        "$PYTHON_BIN" scripts/bench_humaneval_runner.py
        --model "$MODEL"
        --dataset "$DATASET"
        --output-len "$OUTPUT_LEN"
        --num-prompts "$NUM_PROMPTS"
        --warmup-prompts "$WARMUP"
        --gpu-memory-utilization "$GPU_MEM"
        --cpu-offload-gb "$offload"
        --max-model-len "$MAX_MODEL_LEN"
        --enforce-eager
        --trust-remote-code
        --dtype "$DTYPE"
        --output-json "$outdir/result.json"
    )
    if [[ -n "$spec_arg" ]]; then
        cmd+=(--speculative-config "$spec_arg")
    fi

    # ---- Start PCIe bandwidth monitor (background) ----
    local pcie_log="$outdir/pcie_dmon.csv"
    # Clean previous sampling and pin telemetry to the selected GPU only.
    : > "$pcie_log"
    nvidia-smi dmon -s t -d 1 -i "$GPU_ID" -f "$pcie_log" &
    local dmon_pid=$!

    # ---- Run benchmark (vanilla vLLM: no ELMM) ----
    local rc=0
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    VLLM_PLUGINS="" \
    PYTHONUNBUFFERED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    timeout "$TIMEOUT" "${cmd[@]}" 2>&1 | tee "$outdir/bench.log" || rc=${PIPESTATUS[0]}

    # ---- Stop PCIe monitor ----
    kill "$dmon_pid" 2>/dev/null || true
    wait "$dmon_pid" 2>/dev/null || true

    if [[ $rc -ne 0 ]]; then
        RUN_STATUS="fail"
        log "FAIL  ${tag}  offload=${offload}GB  (exit=$rc)"
        # Write a marker so we know it was attempted
        echo "{\"error\": \"exit_code_${rc}\"}" > "$outdir/result_failed.json"
        return 1
    fi

    RUN_STATUS="success"
    log "DONE  ${tag}  offload=${offload}GB"
    return 0
}

# ======================== Main Sweep ========================
log "============================================================"
log "  BriskMoE Phase Transition Experiment"
log "  Model : $(basename "$MODEL")  (~57 GB BF16)"
log "  Draft : $(basename "$DRAFT_MODEL")  (~1 GB)"
log "  GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "  CUDA  : CUDA_VISIBLE_DEVICES=${GPU_ID}"
log "  Sweep : cpu_offload_gb = ${OFFLOAD_VALUES[*]}"
log "  Runs  : ${#OFFLOAD_VALUES[@]} offload × 2 (AR+SD) = $(( ${#OFFLOAD_VALUES[@]} * 2 )) total"
log "============================================================"
render_progress

for offload in "${OFFLOAD_VALUES[@]}"; do
    echo ""
    log ">>> cpu_offload_gb = ${offload} <<<"

    # AR first (less memory pressure → more likely to succeed)
    log "STEP  $((COMPLETED_RUNS + 1))/${TOTAL_RUNS}  ar  offload=${offload}GB"
    run_bench "ar" "$offload" "" || true
    update_progress "$RUN_STATUS" "ar" "$offload"

    # SD (needs extra ~1.5 GB for draft model)
    log "STEP  $((COMPLETED_RUNS + 1))/${TOTAL_RUNS}  sd  offload=${offload}GB"
    run_bench "sd" "$offload" "$SPEC_JSON" || true
    update_progress "$RUN_STATUS" "sd" "$offload"
done

# ======================== Summary Table ========================
echo ""
log "============================================================"
log "  Summary"
log "============================================================"

"$PYTHON_BIN" << 'PYEOF'
import json, os

result_dir = os.environ.get("RESULT_DIR", "results/phase_transition")
offloads = [0, 10, 20, 30, 35, 40, 45]
model_size_gb = 57.0  # approximate BF16 size

header = f"{'Offload':>8} {'GPU%':>6} {'AR TPS':>8} {'SD TPS':>8} {'Speedup':>8} {'AR PCIe':>10} {'SD PCIe':>10}"
print(header)
print("-" * len(header))

for off in offloads:
    gpu_pct = (model_size_gb - off) / model_size_gb * 100.0

    ar_tps = sd_tps = None
    ar_pcie = sd_pcie = None

    for tag in ("ar", "sd"):
        rpath = f"{result_dir}/{tag}_offload_{off}/result.json"
        ppath = f"{result_dir}/{tag}_offload_{off}/pcie_dmon.csv"

        tps_val = None
        if os.path.exists(rpath):
            try:
                d = json.load(open(rpath))
                tps_val = d.get("tps_mean")
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse PCIe dmon: average rxpci during benchmark
        pcie_avg = None
        if os.path.exists(ppath):
            rxvals = []
            with open(ppath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # Format: gpu rxpci txpci
                    if len(parts) >= 2:
                        try:
                            rxvals.append(int(parts[1]))
                        except ValueError:
                            pass
            if rxvals:
                pcie_avg = sum(rxvals) / len(rxvals)

        if tag == "ar":
            ar_tps = tps_val
            ar_pcie = pcie_avg
        else:
            sd_tps = tps_val
            sd_pcie = pcie_avg

    ar_s = f"{ar_tps:.2f}" if ar_tps else "N/A"
    sd_s = f"{sd_tps:.2f}" if sd_tps else "N/A"
    sp_s = f"{sd_tps/ar_tps:.2f}x" if (ar_tps and sd_tps) else "N/A"
    ap_s = f"{ar_pcie:.0f} MB/s" if ar_pcie else "N/A"
    sp_p = f"{sd_pcie:.0f} MB/s" if sd_pcie else "N/A"

    print(f"{off:>5} GB {gpu_pct:>5.0f}% {ar_s:>8} {sd_s:>8} {sp_s:>8} {ap_s:>10} {sp_p:>10}")

print()
print("GPU% = approximate fraction of model weights remaining on GPU")
print("PCIe = average rx bandwidth from nvidia-smi dmon during benchmark")
PYEOF

log "Results saved to: $RESULT_DIR/"
log "Run  scripts/plot_phase_transition.py  to generate figures."
