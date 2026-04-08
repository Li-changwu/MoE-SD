# BriskMoE

**Restoring Speculative Speedups for Offloaded Mixture-of-Experts Inference**

BriskMoE is a lightweight vLLM plugin that makes speculative decoding (SD) an effective latency accelerator for offloaded MoE serving on a single GPU. Without BriskMoE, enabling SD on offloaded MoE models yields negligible speedup (1.05×) due to cache-path overhead and SD-induced transient overflow. BriskMoE eliminates these bottlenecks, achieving up to **4.80×** speedup on Qwen3-30B-A3B and **5.00×** on GPT-OSS-120B.

## Key Results

| Model | AR Baseline | SD + Cache (naive) | SD + BriskMoE | Speedup |
|-------|:-----------:|:------------------:|:-------------:|:-------:|
| Qwen3-30B-A3B (BF16, top-8) | 2.01 tok/s | 7.67 tok/s | 9.98 tok/s | **4.80×** |
| GPT-OSS-120B (MXFP4, top-4) | 2.40 tok/s | 11.52 tok/s | 12.01 tok/s | **5.00×** |

Hardware: NVIDIA RTX A6000 (48 GB), PCIe Gen4, 503 GB host RAM.

## Architecture

BriskMoE addresses two bottlenecks unique to SD + expert caching:

1. **Unblock** — eliminates cache-hit path overhead via Pool-Direct execution, synchronization-free cache routing (TASER), cross-layer prefetch, and fused-MoE kernel tuning.
2. **Protect** — prevents SD-induced cache overflow via DIPP (Draft-Informed Prioritized Preloading), which exploits the draft-generation interval to pre-stage experts.

BriskMoE is implemented as a non-intrusive vLLM general plugin (`adapters/vllm_elmm_plugin.py`). It monkey-patches the model worker after loading and requires **zero modification** to vLLM internals.

## Repository Layout

```
adapters/               # Core BriskMoE plugin
  elmm_plugin.py        #   ELMM: Expert-Level Memory Management (main module)
  vllm_elmm_plugin.py   #   vLLM plugin entry point (registered via pyproject.toml)
  draft_prefetch_hook.py #   Draft-guided expert prefetch hook for EAGLE-3
  accept_reject_tracker.py # Per-expert accept/reject attribution tracker
  sacr.py               #   SACR: Speculation-Aware Cache Replacement
  elp.py                #   ELP: Expert Lifecycle Partitioning
  dipp.py               #   DIPP: Draft-Informed Prioritized Preloading
  pred_cache.py          #   PredCache: Predictive expert cache manager
  briskmoe_cache.py      #   Unified cache facade (SACR + ELP + DIPP)
scripts/                # Benchmark and experiment scripts
tests/                  # Unit tests for each module
data/                   # Benchmark datasets (HumanEval, GSM8K)
docs/paper/             # ATC / SC paper sources
```

## Setup

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU with ≥ 48 GB VRAM (tested on RTX A6000)
- CUDA 12.x
- Model weights downloaded to a local path

### Installation

```bash
# Create conda environment
conda create -n briskmoe python=3.10 -y
conda activate briskmoe

# Install dependencies
pip install -r requirements.txt

# Install BriskMoE as editable package (registers the vLLM plugin)
pip install -e .
```

### Verify Installation

```bash
python -c "from adapters.vllm_elmm_plugin import register; print('BriskMoE plugin ready')"
```

## Benchmarks

All benchmarks use real HumanEval prompts (not random tokens) to produce realistic expert routing patterns. Results are written to `results/<experiment>/` as JSON files.

### Environment Variables

BriskMoE is controlled entirely via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PLUGINS` | `""` | Set to `elmm` to activate BriskMoE |
| `ELMM_CACHE_GB` | `8` | GPU expert cache size in GB |
| `ELMM_POOL_DIRECT` | `1` | Enable Pool-Direct execution (skip scratchpad copy) |
| `ELMM_STALE_REMAP` | `4` | TASER stale-remap threshold |
| `ELMM_DIRECT_DISPATCH` | `0` | Enable direct dispatch optimization |
| `BRISKMOE_DIPP` | `0` | Enable DIPP (Draft-Informed Prioritized Preloading) |
| `BRISKMOE_SACR` | `0` | Enable SACR (Speculation-Aware Cache Replacement) |
| `BRISKMOE_ELP` | `0` | Enable ELP (Expert Lifecycle Partitioning) |

### 1. Single Run with `bench_humaneval_runner.py`

The core benchmark runner feeds HumanEval prompts through `vllm.LLM.generate()` and measures per-prompt latency.

```bash
# AR baseline (no SD, no BriskMoE)
VLLM_PLUGINS="" \
python scripts/bench_humaneval_runner.py \
    --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
    --dataset data/humaneval_bench.jsonl \
    --output-len 128 \
    --num-prompts 50 \
    --warmup-prompts 5 \
    --gpu-memory-utilization 0.90 \
    --cpu-offload-gb 30 \
    --max-model-len 4096 \
    --enforce-eager \
    --trust-remote-code \
    --dtype bfloat16 \
    --output-json results/ar_vanilla/result.json
```

**Parameters explained:**
- `--model`: Path to the HuggingFace model directory.
- `--dataset`: JSONL file where each line contains `{"prompt": "..."}`.
- `--output-len 128`: Generate 128 tokens per prompt.
- `--num-prompts 50`: Number of timed prompts (after warmup).
- `--warmup-prompts 5`: Prompts used for JIT warmup (not timed).
- `--gpu-memory-utilization 0.90`: Fraction of GPU memory vLLM may use.
- `--cpu-offload-gb 30`: Amount of model weights offloaded to CPU (GB).
- `--max-model-len 4096`: Maximum sequence length (KV cache budget).
- `--enforce-eager`: Disable CUDA graphs for deterministic offloading.
- `--trust-remote-code`: Required for Qwen3 model code.
- `--dtype bfloat16`: Model precision.
- `--output-json`: Path to save result JSON (TPS mean/median, latency stats).

**To enable speculative decoding**, add:
```bash
    --speculative-config '{"method":"eagle3","model":"/path/to/speculator.eagle3","num_speculative_tokens":3}'
```
This tells vLLM to use EAGLE-3 with K=3 draft tokens per step.

### 2. Full Ablation: `bench_briskmoe_humaneval.sh`

Runs 7 configurations sequentially to produce the ablation table:

| # | Config | ELMM | SACR | ELP | DIPP | Description |
|---|--------|:----:|:----:|:---:|:----:|-------------|
| 1 | `ar_vanilla` | — | — | — | — | AR baseline (no SD) |
| 2 | `sd_vanilla` | ✓ | — | — | — | SD + LRU cache only |
| 3 | `sd_sacr` | ✓ | ✓ | — | — | + Speculation-Aware Replacement |
| 4 | `sd_elp` | ✓ | — | ✓ | — | + Expert Lifecycle Partitioning |
| 5 | `sd_dipp` | ✓ | — | — | ✓ | + Draft-Informed Preloading |
| 6 | `sd_sacr_elp` | ✓ | ✓ | ✓ | — | SACR + ELP combined |
| 7 | `sd_briskmoe` | ✓ | ✓ | ✓ | ✓ | Full BriskMoE |

```bash
# Edit paths in the script header, then:
bash scripts/bench_briskmoe_humaneval.sh
```

Results are saved to `results/briskmoe_humaneval/<config>/result.json`.

### 3. Cross-Architecture: `bench_gptoss_cross_arch.sh`

Validates BriskMoE on GPT-OSS-120B (MXFP4 quantization, 128 experts, top-4 routing):

```bash
bash scripts/bench_gptoss_cross_arch.sh
```

Runs 3 configurations: `ar_elmm`, `sd_elmm`, `sd_briskmoe`. Results in `results/gptoss_cross_arch/`.

### 4. Cache Strategy Simulation: `bench_elp_dipp_ablation.py`

Offline trace-based simulation of cache replacement policies (LRU, SACR, ELP, DIPP) across multiple cache sizes. Does **not** require GPU — operates on pre-recorded routing traces.

```bash
python scripts/bench_elp_dipp_ablation.py
```

### 5. Motivation Figures: `collect_and_plot_real_motivation.py`

Generates the paper's motivation figure data (working set expansion, throughput comparison):

```bash
python scripts/collect_and_plot_real_motivation.py
```

## Tests

```bash
# Run all BriskMoE unit tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_dipp.py -v
```

## Citation

Paper under submission to USENIX ATC 2026.
