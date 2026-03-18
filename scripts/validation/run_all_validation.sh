#!/bin/bash
# ==============================================================================
# SpecMoE 先行验证实验 — 一键运行
# ==============================================================================
#
# 用法:
#   cd /root/MoE-SD
#   bash scripts/validation/run_all_validation.sh [--skip-trace] [--K 3]
#
# 实验流水线:
#   Exp0: Expert Trace 采集     (需要 GPU, ~15 min)
#   Exp1: MAF 实证测量           (纯 CPU 分析, ~1 min)
#   Exp2: Dedup 收益估算         (纯 CPU 分析 + 可选 GPU microbench, ~2 min)
#   Exp3: SDD 可行性验证         (纯 CPU 分析, ~3 min)
#   Exp4: Temporal Locality      (纯 CPU 分析, ~1 min)
#
# 前置条件:
#   - Python 3.10+, torch (GPU), transformers, numpy, pandas
#   - Qwen3-30B 模型在 /home/sage3/models/Qwen3-30B-A3B-Instruct-2507
#   - combined_sharegpt.json 在 data/
#
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
MODEL_PATH="/home/sage3/models/Qwen3-30B-A3B-Instruct-2507"
DATASET="data/combined_sharegpt.json"
OUTPUT_DIR="results/validation"
NUM_PROMPTS=50
MAX_TOKENS=128
K=3
SKIP_TRACE=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-trace)   SKIP_TRACE=true; shift ;;
        --model)        MODEL_PATH="$2"; shift 2 ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --num-prompts)  NUM_PROMPTS="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --K)            K="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TRACE_FILE="$OUTPUT_DIR/expert_trace.jsonl"

echo "========================================"
echo " SpecMoE Validation Experiments"
echo "========================================"
echo " Model:       $MODEL_PATH"
echo " Dataset:     $DATASET"
echo " Prompts:     $NUM_PROMPTS"
echo " Max tokens:  $MAX_TOKENS"
echo " K:           $K"
echo " Output:      $OUTPUT_DIR"
echo " Skip trace:  $SKIP_TRACE"
echo "========================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# ── Exp0: Expert Trace Collection ──
if [ "$SKIP_TRACE" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " [Exp0] Expert Trace Collection"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/validation/exp0_collect_expert_trace.py \
        --model "$MODEL_PATH" \
        --dataset "$DATASET" \
        --num-prompts "$NUM_PROMPTS" \
        --max-tokens "$MAX_TOKENS" \
        --output "$TRACE_FILE"
    echo ""
else
    echo "[Exp0] Skipped (--skip-trace). Using existing trace: $TRACE_FILE"
    if [ ! -f "$TRACE_FILE" ]; then
        echo "ERROR: Trace file not found: $TRACE_FILE"
        echo "Run without --skip-trace first."
        exit 1
    fi
    echo ""
fi

# Verify trace exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "ERROR: Trace collection failed. $TRACE_FILE not found."
    exit 1
fi
TRACE_LINES=$(wc -l < "$TRACE_FILE")
echo "Trace file: $TRACE_FILE ($TRACE_LINES events)"
echo ""

# ── Exp1: MAF Analysis ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [Exp1] MAF Measurement & Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/validation/exp1_maf_analysis.py \
    --trace "$TRACE_FILE" \
    --output-dir "$OUTPUT_DIR"
echo ""

# ── Exp2: Dedup Savings ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [Exp2] Dedup Savings Estimation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/validation/exp2_dedup_savings.py \
    --trace "$TRACE_FILE" \
    --output-dir "$OUTPUT_DIR"
echo ""

# ── Exp3: SDD Feasibility ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [Exp3] SDD Feasibility"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/validation/exp3_sdd_feasibility.py \
    --trace "$TRACE_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --K "$K"
echo ""

# ── Exp4: Temporal Locality ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [Exp4] Expert Temporal Locality"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/validation/exp4_temporal_locality.py \
    --trace "$TRACE_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --K "$K"
echo ""

# ── Summary ──
echo "========================================"
echo " All Validation Experiments Complete"
echo "========================================"
echo ""
echo "Reports generated:"
echo "  $OUTPUT_DIR/trace_collection_summary.json  (Exp0)"
echo "  $OUTPUT_DIR/exp1_maf_report.json           (Exp1: MAF)"
echo "  $OUTPUT_DIR/exp2_dedup_report.json         (Exp2: Dedup)"
echo "  $OUTPUT_DIR/exp3_sdd_report.json           (Exp3: SDD)"
echo "  $OUTPUT_DIR/exp4_locality_report.json      (Exp4: Locality)"
echo ""
echo "CSV data files:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (none yet)"
echo ""

# ── Aggregate Go/No-Go Decision ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Go/No-Go Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json, sys
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
total_pass = 0
total_fail = 0

for report_file in sorted(output_dir.glob('exp*_report.json')):
    with open(report_file) as f:
        report = json.load(f)
    tests = report.get('hypothesis_tests', {})
    for name, test in tests.items():
        status = '✅' if test.get('pass') else '❌'
        print(f'  {status} {name}: {test.get(\"interpretation\", \"\")}')
        if test.get('pass'):
            total_pass += 1
        else:
            total_fail += 1

print()
print(f'  Total: {total_pass} passed, {total_fail} failed')
if total_fail == 0:
    print('  🟢 GO — 所有假设验证通过，SpecMoE 技术路线可行')
elif total_fail <= 2:
    print('  🟡 CONDITIONAL GO — 部分假设未通过，需调整技术路线')
else:
    print('  🔴 NO-GO — 多个核心假设未通过，需重新评估方向')
"
echo "========================================"
