#!/usr/bin/env bash
set -euo pipefail

# Quick start script for downloading public datasets and running benchmarks
# Usage: ./scripts/download_and_bench.sh <dataset> [num_samples]

DATASET="${1:-sharegpt}"
NUM_SAMPLES="${2:-100}"

echo "=========================================="
echo "Public Dataset Benchmark Setup"
echo "=========================================="
echo ""

# Step 1: Download dataset
echo "[Step 1] Downloading ${DATASET} dataset (${NUM_SAMPLES} samples)..."
python3 tools/dataset_downloader.py \
    --dataset "${DATASET}" \
    --max-samples "${NUM_SAMPLES}" \
    --output "data/${DATASET}_subset.jsonl"

# Verify download
if [[ ! -f "data/${DATASET}_subset.jsonl" ]]; then
    echo "ERROR: Dataset download failed!"
    exit 1
fi

echo ""
echo "[Step 2] Dataset downloaded successfully"
wc -l "data/${DATASET}_subset.jsonl"

# Step 3: Show sample prompts
echo ""
echo "[Step 3] Sample prompts from dataset:"
head -3 "data/${DATASET}_subset.jsonl" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || head -3 "data/${DATASET}_subset.jsonl"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Start your vLLM server:"
echo "   python3 -m vllm.entrypoints.openai.api_server \\"
echo "     --model <your-model-path> \\"
echo "     --max-model-len 2048 \\"
echo "     --gpu-memory-utilization 0.9"
echo ""
echo "2. Run benchmark with dataset:"
echo "   python3 tools/bench_runner.py --config configs/experiments/baseline_${DATASET}.yaml"
echo ""
echo "3. Or customize with:"
echo "   python3 tools/bench_runner.py --config configs/experiments/baseline_${DATASET}.yaml"
echo ""
echo "Results will be saved to: results/raw/"
echo ""
