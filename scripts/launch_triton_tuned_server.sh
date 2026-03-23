#!/usr/bin/env bash
# Server launcher for Triton tuning benchmark
# Sets env vars via 'export' then launches server directly
set -euo pipefail
cd /root/MoE-SD

export VLLM_PLUGINS=elmm
export ELMM_CACHE_GB=4
export ELMM_LOG_INTERVAL=0
export ELMM_STALE_REMAP=16
export ELMM_ORACLE_PREFETCH=1
export ELMM_GPU_CACHE=0

exec /opt/miniconda3/envs/moe-sd/bin/python -m vllm.entrypoints.openai.api_server \
    --model "/root/models/Qwen3-30B-A3B-Instruct-2507" \
    --speculative-config '{"model": "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3", "method": "eagle3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1}' \
    --tensor-parallel-size 1 \
    --cpu-offload-gb 30 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
