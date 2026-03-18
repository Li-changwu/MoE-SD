# Section 1: Introduction

<!-- SpecMoE: Operator-Level Co-Design of Speculative Decoding and Mixture-of-Experts Inference -->
<!-- Target: OSDI'26 / ATC'26 (CCF-A) -->

## 1 Introduction

Large Language Models (LLMs) based on Mixture-of-Experts (MoE) architectures have emerged as a dominant paradigm for scaling model capacity without proportional compute costs. Models such as Mixtral-8×22B, DeepSeek-V3, and Qwen3-30B-A3B achieve state-of-the-art performance by sparsely activating a subset of *experts* per token—for instance, Qwen3-30B routes each token to 8 of 128 experts per layer across 48 MoE layers, yielding only 3B active parameters from a 30B total. This sparse activation, however, demands that the full expert parameter set (approximately 57.6 GB in bfloat16 for Qwen3-30B) reside in accessible memory, often necessitating CPU offloading on commodity GPUs (e.g., NVIDIA RTX A6000 with 48 GB).

Speculative Decoding (SD) has become the standard latency-reduction technique for autoregressive LLM inference. By employing a lightweight *draft model* to propose $K$ candidate tokens that the *target model* verifies in a single parallel forward pass, SD amortizes the high per-token cost of large models. On dense models, SD routinely achieves 2–3× speedups.

**However, a critical yet underexplored problem arises when SD meets MoE: speculative decoding can *degrade* rather than improve performance.** Our measurements on Qwen3-30B with EAGLE-3 (a state-of-the-art SD method) show a **17.1% throughput reduction** compared to standard autoregressive decoding.  This counterintuitive result stems from what we term the **MoE Amplification Factor (MAF)**: during the verify phase, $K+1$ tokens are processed simultaneously through each MoE layer. Because different tokens typically route to different experts, the number of unique experts that must be loaded grows sublinearly but substantially with $K$. Under CPU offloading, where expert weight transfer over PCIe is the primary bottleneck, this amplified expert loading cost can overwhelm the latency savings from token parallelism.

We formalize this tradeoff with the **MAF-aware Speedup formula**:

$$S = \frac{\bar{\alpha}(K+1)}{1 + \gamma + \beta(\text{MAF}(K) - 1)}$$

where $\bar{\alpha}$ is the mean acceptance rate, $\gamma$ captures the draft model overhead, and $\beta$ is the fraction of per-token latency attributable to expert weight loading (the *offload ratio*). When $\beta$ is large—as in CPU-offloaded MoE—the denominator grows faster than the numerator, and $S < 1$ (slowdown).

Existing approaches to the MoE+SD problem operate at the **scheduling level**: adjusting $K$ dynamically (Cascade), budgeting expert activation counts (MoE-Spec), or swapping experts proactively (MoE-SpAc). These methods treat the MoE kernel as a black box and cannot reduce the fundamental per-expert loading cost.

In this paper, we propose **SpecMoE**, the first system to address the MoE amplification problem at the **operator level** through kernel–scheduler co-design. SpecMoE introduces three synergistic techniques:

1. **SpecFusedMoE Kernel** — A Triton-based fused MoE operator that performs cross-token expert deduplication during the verify phase. Instead of loading each expert independently for every token, SpecFusedMoE builds a unified dispatch table that loads each unique expert *once* and scatters outputs to all requesting tokens. This reduces the effective MAF from the naive $K+1$ toward the theoretical minimum $\text{MAF}(K) = N \cdot (1 - (1-k/N)^{K+1})/k$.

2. **Speculation Divergence Detector (SDD)** — A layer-wise early termination mechanism that monitors router logit divergence across MoE layers. When a draft token's routing pattern diverges significantly from expected behavior (indicating likely rejection), SDD freezes that token's MoE computation in subsequent layers. This further reduces the effective expert count without waiting for the final verification result.

3. **Expert Temporal Cache** — A GPU/CPU tiered caching system that exploits the inter-round expert locality inherent in speculative decoding. By prefetching experts predicted by the draft model's routing during the draft phase, the cache hides a significant fraction of PCIe transfer latency.

SpecMoE is implemented as a lightweight, non-intrusive extension to vLLM 0.17+ through runtime monkey-patching of the `fused_moe` operator. We evaluate SpecMoE on Qwen3-30B-A3B with EAGLE-3 on an NVIDIA RTX A6000 with 30 GB CPU offloading.

**Contributions:**

- We formalize the **MoE Amplification Factor (MAF)** and derive the closed-form **MAF-aware speedup formula**, revealing why SD degrades MoE inference under CPU offloading and establishing the breakeven condition $\bar{\alpha}_{\min} = (1 + \gamma + \beta(\text{MAF}(K)-1)) / (K+1)$.

- We design and implement **SpecFusedMoE**, the first SD-aware MoE kernel with cross-token expert deduplication, achieving up to 75% expert load reduction when tokens share routing patterns.

- We propose **SDD**, a novel layer-wise early termination technique based on router logit divergence, reducing effective MAF by an estimated 21.5%.

- We build an **expert temporal cache** with speculative prefetch that leverages draft-model routing predictions to pre-stage expert weights on GPU.

- We integrate all components into **SpecMoE**, a production-ready system with 67 unit tests, and demonstrate its effectiveness on real MoE+SD workloads.
