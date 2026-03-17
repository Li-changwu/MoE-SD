#!/bin/bash
# Automated Benchmark Runner for MoE-SD Project
# Runs benchmarks on public datasets (Alpaca, Dolly) with no_sd and eagle3 methods

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "vllm_moe_sd" ]]; then
    log_error "Please activate vllm_moe_sd conda environment first"
    echo "Run: source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm_moe_sd"
    exit 1
fi

# Check if data files exist
for dataset in alpaca dolly; do
    if [[ ! -f "data/${dataset}_subset.jsonl" ]]; then
        log_error "Missing dataset: data/${dataset}_subset.jsonl"
        exit 1
    fi
done

log_info "All datasets found. Starting benchmarks..."

# Run benchmarks
CONFIGS=("alpaca_no_sd" "alpaca_eagle3" "dolly_no_sd" "dolly_eagle3")

for config in "${CONFIGS[@]}"; do
    log_info "=========================================="
    log_info "Running: $config"
    log_info "=========================================="
    
    mkdir -p logs
    
    # Run benchmark (bench_runner will start vLLM internally)
    python tools/bench_runner.py --config configs/experiments/${config}.yaml 2>&1 | tee logs/${config}.log
    
    log_info "Completed: $config"
done

log_info "=========================================="
log_info "All benchmarks completed!"
log_info "Results: results/"
log_info "Logs: logs/"
