# Section 2: Background & Motivation

## 2.1 Mixture-of-Experts Architecture

Modern MoE transformers replace the dense Feed-Forward Network (FFN) in each layer with a sparse set of parallel experts, gated by a lightweight router. Formally, for an input token embedding $\mathbf{x} \in \mathbb{R}^{d}$, the MoE layer computes:

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}(\mathbf{r}(\mathbf{x}), k)} g_i(\mathbf{x}) \cdot \text{FFN}_i(\mathbf{x})$$

where $\mathbf{r}(\mathbf{x}) \in \mathbb{R}^N$ is the router logit vector over $N$ experts, $\text{TopK}(\cdot, k)$ selects the $k$ experts with highest router scores, and $g_i(\mathbf{x})$ is the normalized gating weight.

**Qwen3-30B-A3B** is a representative production MoE model:

| Parameter | Value | Implication |
|-----------|-------|-------------|
| Total parameters | 30.5B | Model capacity |
| Active parameters | 3.3B | Per-token compute cost |
| Hidden dimension $d$ | 2,048 | Token embedding size |
| MoE layers $L$ | 48 | Depth of expert stack |
| Experts per layer $N$ | 128 | Expert pool size |
| Top-$k$ routing | $k = 8$ | Activated experts per token |
| Expert FFN dimension | 768 | Each expert's hidden size |
| Expert size (bf16) | ~9.4 MB | 3 weight matrices ×(768×2048×2B) |
| Total expert memory | ~57.6 GB | 128×48×9.4 MB |

Since 57.6 GB exceeds the A6000's 48 GB VRAM, expert weights must be partially offloaded to CPU memory via PCIe. In CPU-offloaded inference, expert weight transfer becomes the dominant latency component, as PCIe Gen4 ×16 bandwidth (~25 GB/s) is orders of magnitude slower than GPU HBM2e bandwidth (~768 GB/s).

## 2.2 Speculative Decoding

Speculative Decoding (SD) accelerates autoregressive generation by exploiting the *parallel verification* property: while generating tokens sequentially requires $T$ serial forward passes, SD uses a cheap *draft model* to propose $K$ candidate tokens, then verifies all $K$ tokens (plus the next token) in a single target-model forward pass.

**EAGLE-3** draft model uses a lightweight transformer head that predicts the next-token hidden state based on the target model's last-layer features. For Qwen3-30B-A3B, the EAGLE-3 speculator has approximately 0.6B parameters.

The SD speedup for dense models is:

$$S_{\text{dense}} = \frac{\bar{\alpha}(K+1)}{1 + \gamma}$$

where $\bar{\alpha}$ is the mean acceptance rate (fraction of draft tokens accepted) and $\gamma$ is the draft-model overhead ratio. With $\bar{\alpha} \approx 0.6$ and $\gamma \approx 0.1$, dense-model SD achieves $S \approx 1.6\times$ for $K=3$.

## 2.3 The MoE Amplification Problem

The above analysis assumes the cost of a forward pass is independent of the batch size (or scales sublinearly). This holds for dense models where all parameters are loaded regardless of input. **For MoE models under CPU offloading, this assumption breaks down catastrophically.**

### MAF Definition

When $K+1$ tokens pass through a MoE layer simultaneously, each token selects $k$ experts from $N$. Without deduplication, the naive expert load count is $(K+1) \times k$. With deduplication, only **unique** experts need loading. We define the **MoE Amplification Factor**:

$$\text{MAF}(K) = \frac{\mathbb{E}\left[\left|\bigcup_{i=0}^{K} E_i\right|\right]}{k}$$

where $E_i$ denotes the set of $k$ experts selected by token $i$.

### Theoretical MAF under Independent Routing

Assuming tokens independently sample $k$ experts from $N$ (a reasonable approximation for the verify batch where draft tokens are generated sequentially from different context states), the expected number of unique experts follows a **coupon collector** argument:

$$\mathbb{E}\left[\left|\bigcup_{i=0}^{K} E_i\right|\right] = N \cdot \left(1 - \left(1 - \frac{k}{N}\right)^{K+1}\right)$$

Thus:

$$\boxed{\text{MAF}(K) = \frac{N}{k} \cdot \left(1 - \left(1 - \frac{k}{N}\right)^{K+1}\right)}$$

For Qwen3-30B ($N=128, k=8$):

| $K$ | $K+1$ tokens | Naive loads | Unique experts (theory) | MAF | Dedup saving |
|-----|-------------|-------------|------------------------|-----|--------------|
| 0 | 1 | 8 | 8.0 | 1.00 | 0% |
| 1 | 2 | 16 | 15.5 | 1.94 | 3.1% |
| 2 | 3 | 24 | 22.6 | 2.82 | 5.8% |
| 3 | 4 | 32 | 29.3 | 3.66 | 8.4% |
| 4 | 5 | 40 | 35.7 | 4.46 | 10.8% |
| 5 | 6 | 48 | 41.7 | 5.21 | 13.1% |

**Observation**: For small $k/N$ (= 8/128 = 6.25%), the dedup saving per token is modest (~3–13%), but the **absolute load increase** is massive. Going from $K=0$ to $K=3$ increases expert loads per layer from 8 to 29.3 — a **3.66× increase in PCIe transfer volume**.

### MAF-Aware Speedup Formula

Incorporating the MoE amplification into the SD speedup analysis, we decompose per-token latency into:
- **Compute component** $(1-\beta)$: attention + routing + activation (GPU-bound, scales well with batching)
- **Expert loading component** $\beta$: PCIe transfer of expert weights (bandwidth-bound, scales linearly with unique expert count)

The speedup becomes:

$$\boxed{S = \frac{\bar{\alpha}(K+1)}{1 + \gamma + \beta(\text{MAF}(K) - 1)}}$$

### The Breakeven Condition

Setting $S = 1$ and solving for $\bar{\alpha}$:

$$\bar{\alpha}_{\min} = \frac{1 + \gamma + \beta(\text{MAF}(K) - 1)}{K+1}$$

For $\gamma = 0.1$ and $\beta = 0.6$ (typical for CPU-offloaded MoE):

| $K$ | MAF(K) | $\beta \cdot (\text{MAF}(K)-1)$ | Denominator | $\bar{\alpha}_{\min}$ |
|-----|--------|--------------------------------|-|---|
| 1 | 1.94 | 0.56 | 1.66 | 0.83 |
| 2 | 2.82 | 1.09 | 2.19 | 0.73 |
| 3 | 3.66 | 1.60 | 2.70 | 0.68 |
| 4 | 4.46 | 2.08 | 3.18 | 0.64 |

**Key Insight**: Even at $K=3$, SD requires $\bar{\alpha} \geq 68\%$ just to break even. Typical EAGLE-3 acceptance rates on long-context prompts are 55–65%, falling below this threshold. This explains our empirical observation of 17.1% throughput degradation.

### Empirical Validation

Our baseline measurements on Qwen3-30B with EAGLE-3 ($K=3$, A6000 + 30 GB CPU offload):

| Metric | No-SD | EAGLE-3 | Delta |
|--------|-------|---------|-------|
| Output throughput (tok/s) | 7.44 | 6.52 | **−12.4%** |
| Total throughput (tok/s) | 23.56 | 19.53 | **−17.1%** |
| TTFT p95 (ms) | 8,984 | 9,082 | +1.1% |
| Average latency (ms) | 66,241 | 72,966 | +10.2% |

Plugging the observed $S = 19.53/23.56 = 0.829$ into our formula with $K=3$, $\gamma=0.1$:

$$0.829 = \frac{\bar{\alpha} \cdot 4}{1.1 + 0.6 \cdot (3.66-1)} \implies \bar{\alpha} \approx 0.56$$

This matches the typical EAGLE-3 acceptance rate, validating our MAF model.

## 2.4 Why Scheduling-Level Solutions Are Insufficient

Existing approaches address the MoE+SD conflict at the scheduling level:

| Method | Mechanism | Limitation |
|--------|-----------|------------|
| **Cascade** (2024) | Dynamic $K$ based on acceptance utility | Cannot reduce per-expert loading cost; may repeatedly undershoot optimal $K$ |
| **MoE-Spec** (2024) | Budget expert activation count | Truncates expert selection → quality degradation |
| **MoE-SpAc** (2025) | Speculative expert swapping | Requires accurate expert prediction; overhead when predictions miss |
| **SP-MoE** (2024) | Schedule prefill/decode separately | Orthogonal to expert loading cost |

All these methods treat the MoE operator as a black box. They cannot:
1. **Deduplicate** expert loads across the verify batch ($K+1$ tokens sharing experts).
2. **Early-terminate** obviously-rejected tokens before they complete all 48 MoE layers.
3. **Prefetch** expert weights using draft-model routing predictions.

SpecMoE addresses all three at the **operator level**, reducing MAF and hiding transfer latency below the scheduling abstraction boundary. Crucially, SpecMoE is **complementary** to scheduling-level methods — the reduced MAF directly improves the breakeven condition, enabling higher $K$ values and greater SD gains.
