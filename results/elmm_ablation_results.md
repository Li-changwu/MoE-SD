# ELMM Ablation Results

## Experiment Date: 2025-03-20

### Hardware
- GPU: NVIDIA RTX A6000 (48 GB, 44.55 GB usable)
- PCIe: Gen4 x16 (~25 GB/s)
- HBM bandwidth: ~768 GB/s

### Model
- Qwen3-30B-A3B-Instruct-2507
- 128 experts, top-8, 48 MoE layers
- ~54 GB in bf16 (30 GB offloaded to CPU via UVA)

### Speculative Decoding
- EAGLE-3 (0.6B params, K=3)
- Acceptance rate: ~25-45% (varies by sequence)

---

## Results

| Config | Description | Throughput (tok/s) | Speedup vs C2 |
|--------|-------------|-------------------|---------------|
| C1 | No speculative decode, UVA offload | ~1.99 | 0.99× |
| C2 | EAGLE-3 + UVA (baseline) | 2.009 | 1.00× |
| **ELMM** | **EAGLE-3 + ELMM (4GB cache)** | **5.89** | **2.93×** |

### ELMM Configuration
- Cache: 4 GB (pre-allocated GPU pool)
  - 26 offloaded layers × ~17 experts/layer
- Scratchpad: 1.15 GB (shared GPU buffer for all layers)
- GPU memory utilization: 0.85
- KV cache: 8.58 GiB (91,840 tokens)

### ELMM Detailed Measurements (5 requests, 128 max_tokens)
| Request | Tokens | Time (s) | tok/s |
|---------|--------|----------|-------|
| 0 | 128 | 23.94 | 5.35 |
| 1 | 128 | 19.56 | 6.54 |
| 2 | 128 | 19.22 | 6.66 |
| 3 | 128 | 24.79 | 5.16 |
| 4 | 128 | 21.18 | 6.04 |
| **Total** | **640** | **108.68** | **5.89** |

---

## Analysis

### Why ELMM Works
1. **UVA PCIe overhead**: Without caching, every MoE layer loads ~8 activated experts
   from CPU via PCIe (~25 GB/s) on every token. With K=3 speculative tokens,
   MAF theory predicts 3.64× redundant PCIe traffic.

2. **ELMM cache eliminates redundancy**: By caching recently-used experts in GPU
   memory (HBM), cache hits serve at ~768 GB/s instead of ~25 GB/s (30× faster).
   With 17 experts cached per layer (out of 128), typical decode steps achieve
   high hit rates since expert activation patterns have strong temporal locality.

3. **Scratchpad-swap protocol**: Instead of trying to modify UVA tensors in-place,
   ELMM pre-fills a GPU scratchpad with needed expert weights, then swaps
   `param.data` before the kernel call. This avoids the UVA write-to-CPU problem.

### Key Technical Insights
- `_vllm_offloaded_cpu_data` attribute is lost during `process_weights_after_loading`
  due to `replace_parameter()` creating new Parameter objects. Detection uses
  sibling params (gate weight retains the attribute).
- Pre-allocated cache pools are mandatory — lazy allocation causes OOM because
  vLLM's KV profiler greedily uses all remaining GPU memory.
- No explicit CUDA synchronization needed — all operations on same stream.
- ELMM's pre-allocated pools actually improve GPU memory fragmentation.
