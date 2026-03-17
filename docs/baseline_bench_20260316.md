# Baseline Benchmark Snapshot (2026-03-16)

This note captures a comparable no-SD vs EAGLE-3 run on `vllm==0.17.1`.

## Setup

- Model: `/home/sage3/models/Qwen3-30B-A3B-Instruct-2507`
- Spec model: `/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3`
- Prompt/output: 128 / 64
- Num prompts: 8
- Request rate: 1.0
- Max model len: 256
- GPU memory utilization: 0.9
- CPU offload: 30 GiB
- Swap space: 64 GiB

## Commands

```bash
# no-SD serve
vllm bench serve --model /home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
  --backend openai --base-url http://127.0.0.1:8000/v1 --endpoint /completions \
  --num-prompts 8 --random-input-len 128 --random-output-len 64 --request-rate 1 \
  --seed 42 --save-result --result-dir results/raw/no_sd_serve

# EAGLE-3 serve (server launched with --speculative_config)
vllm bench serve --model /home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
  --backend openai --base-url http://127.0.0.1:8000/v1 --endpoint /completions \
  --num-prompts 8 --random-input-len 128 --random-output-len 64 --request-rate 1 \
  --seed 42 --save-result --result-dir results/raw/eagle3_serve

# no-SD latency/throughput
make bench-latency-no-sd MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
  PROMPT_LEN=128 OUTPUT_LEN=64 MAX_MODEL_LEN=256 GPU_MEMORY_UTILIZATION=0.9 \
  CPU_OFFLOAD_GB=30 SWAP_SPACE_GB=64

make bench-throughput-no-sd MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
  NUM_PROMPTS=8 PROMPT_LEN=128 OUTPUT_LEN=64 MAX_MODEL_LEN=256 \
  GPU_MEMORY_UTILIZATION=0.9 CPU_OFFLOAD_GB=30 SWAP_SPACE_GB=64

# EAGLE-3 latency/throughput
make bench-latency-eagle3 MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
  SPEC_MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3 \
  SPEC_METHOD=eagle3 SPEC_TOKENS=2 PROMPT_LEN=128 OUTPUT_LEN=64 MAX_MODEL_LEN=256 \
  GPU_MEMORY_UTILIZATION=0.9 CPU_OFFLOAD_GB=30 SWAP_SPACE_GB=64

make bench-throughput-eagle3 MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
  SPEC_MODEL=/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3 \
  SPEC_METHOD=eagle3 SPEC_TOKENS=2 NUM_PROMPTS=8 PROMPT_LEN=128 OUTPUT_LEN=64 \
  MAX_MODEL_LEN=256 GPU_MEMORY_UTILIZATION=0.9 CPU_OFFLOAD_GB=30 SWAP_SPACE_GB=64
```

## Results (from `results/parsed/summary.csv`)

| Mode | Metric | no-SD | EAGLE-3 |
| --- | --- | ---: | ---: |
| serve | mean TTFT (ms) | 8983.77 | 9081.55 |
| serve | p95 TTFT (ms) | 11636.37 | 11731.42 |
| serve | output throughput (tok/s) | 7.4378 | 6.5184 |
| latency | avg latency (ms) | 66241.40 | 72965.91 |
| throughput | total tokens/s | 23.5624 | 19.5327 |
| throughput | requests/s | 0.1227 | 0.1017 |

## Raw Artifacts

- `results/raw/no_sd_serve/openai-1.0qps-Qwen3-30B-A3B-Instruct-2507-20260316-211033.json`
- `results/raw/eagle3_serve/openai-1.0qps-Qwen3-30B-A3B-Instruct-2507-20260316-213535.json`
- `results/raw/no_sd_latency/latency.json`
- `results/raw/eagle3_latency/latency.json`
- `results/raw/no_sd_throughput/throughput.json`
- `results/raw/eagle3_throughput/throughput.json`

## Notes

- On vLLM 0.17, `latency` and `throughput` use `--output-json` instead of `--save-result/--result-dir`.
- `bench serve` can return `Bad Request` when `random_input_len + random_output_len` exceeds server `max_model_len`.