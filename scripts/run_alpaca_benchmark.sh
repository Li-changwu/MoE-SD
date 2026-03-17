#!/bin/bash
# Alpaca Dataset Benchmark Runner
# Tests both no_sd and eagle3 methods on Alpaca subset (100 samples)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "Alpaca Dataset Benchmark"
echo "=========================================="
echo ""

# Check if data file exists
if [[ ! -f "data/alpaca_subset.jsonl" ]]; then
    echo "[ERROR] Missing Alpaca dataset: data/alpaca_subset.jsonl"
    echo "Please download it first:"
    echo "  python tools/dataset_downloader.py --dataset alpaca --max-samples 100"
    exit 1
fi

echo "[INFO] Dataset found: data/alpaca_subset.jsonl"
echo ""

# Create logs directory
mkdir -p logs

# Run no_sd baseline
echo "=========================================="
echo "Test 1/2: Alpaca + No Speculative Decoding"
echo "=========================================="
python tools/bench_runner.py --config configs/experiments/alpaca_no_sd.yaml 2>&1 | tee logs/alpaca_no_sd.log
echo ""
echo "[INFO] Completed: alpaca_no_sd"
echo ""

# Run eagle3 baseline
echo "=========================================="
echo "Test 2/2: Alpaca + EAGLE-3 Speculative Decoding"
echo "=========================================="
python tools/bench_runner.py --config configs/experiments/alpaca_eagle3.yaml 2>&1 | tee logs/alpaca_eagle3.log
echo ""
echo "[INFO] Completed: alpaca_eagle3"
echo ""

echo "=========================================="
echo "All Alpaca Benchmarks Completed!"
echo "=========================================="
echo ""
echo "Results location: results/"
echo "  - results/alpaca_no_sd/"
echo "  - results/alpaca_eagle3/"
echo ""
echo "Log files: logs/"
echo "  - logs/alpaca_no_sd.log"
echo "  - logs/alpaca_eagle3.log"
echo ""
echo "To compare results, run:"
echo "  python tools/parse_bench_results.py results/alpaca_no_sd/ results/alpaca_eagle3/"
