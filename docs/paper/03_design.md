# Section 3: SpecMoE System Design

## 3.1 System Overview

SpecMoE is an operator-level co-design framework that eliminates the MoE Amplification Factor (MAF) penalty in speculative decoding. Rather than modifying the scheduler or the draft model, SpecMoE intercepts the MoE operator itself within the vLLM serving engine and applies three coordinated optimizations:

1. **SpecFusedMoE** — A fused MoE kernel with cross-token expert deduplication that reduces expert weight loads from $O((K+1) \cdot k)$ to $O(|\text{unique experts}|)$.
2. **Speculation Divergence Detection (SDD)** — Layer-wise early termination that freezes tokens predicted to be rejected, skipping their computation in deeper MoE layers.
3. **Acceptance-Aware Expert Prefetch** — Proactive GPU cache management that uses draft-model routing predictions to overlap expert transfers with computation.

Figure 1 shows the SpecMoE architecture integrated into vLLM's speculative decoding pipeline:

```
┌──────────────────────────────────────────────────────────┐
│                   vLLM Serving Engine                     │
│  ┌────────────┐    ┌────────────┐    ┌────────────────┐  │
│  │ Draft Model │    │ Verify     │    │ Accept/Reject  │  │
│  │ (EAGLE-3)  │──▶│ Forward    │──▶│ Decision       │  │
│  └──────┬─────┘    └──────┬─────┘    └───────┬────────┘  │
│         │                 │                   │           │
│  ┌──────▼─────────────────▼───────────────────▼────────┐ │
│  │              SpecMoE Operator Layer                  │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │ │
│  │  │SpecFused │  │  SDD     │  │ Expert Cache +   │  │ │
│  │  │MoE Kernel│  │  Early   │  │ Prefetch Engine  │  │ │
│  │  │(Dedup)   │  │  Termn.  │  │ (Acceptance-Aware│  │ │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Phase-Aware Governor & Controllers         │ │
│  │  K adaptation │ Memory partition │ Prefetch policy   │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

The key architectural principle is **transparency**: SpecMoE intercepts vLLM's `fused_moe` operator via monkey-patching, requiring zero modifications to the model definition, scheduler, or inference loop. This enables drop-in deployment with any vLLM-supported MoE model.

## 3.2 SpecFusedMoE: Cross-Token Expert Deduplication

### 3.2.1 Motivation

In standard MoE inference, each token's top-$k$ expert dispatch is computed independently. When $K+1$ tokens are processed in the verify batch, the same expert may be loaded from CPU memory multiple times — once per token that selects it. SpecFusedMoE eliminates this redundancy by building an expert-grouped dispatch table that loads each unique expert exactly once.

### 3.2.2 Algorithm

Given a verify batch of $T = K+1$ tokens with router assignments $\{(e_{t,1}, w_{t,1}), \ldots, (e_{t,k}, w_{t,k})\}_{t=1}^{T}$, SpecFusedMoE operates in three phases:

**Phase 1: Dispatch Table Construction.** Build an expert-to-tokens mapping:
$$D = \{e \mapsto \{(t, w_{t,j}) \mid e_{t,j} = e, \; 1 \leq t \leq T, \; 1 \leq j \leq k\}\}$$

The number of entries in $D$ is $U = |\{e_{t,j}\}|$, the number of unique experts. This is exactly $k \cdot \text{MAF}(K)$.

**Phase 2: Expert-Grouped MatMul.** For each unique expert $e \in D$:
1. Load expert weights $W_e^{\text{gate}}, W_e^{\text{up}}, W_e^{\text{down}}$ from GPU cache (or CPU via PCIe).
2. Gather hidden states $\mathbf{h}_t$ for all tokens $t \in D[e]$.
3. Compute fused SiLU-gated FFN:
$$\mathbf{g} = \text{SiLU}(\mathbf{H} \cdot W_e^{\text{gate}\top}), \quad \mathbf{u} = \mathbf{H} \cdot W_e^{\text{up}\top}, \quad \mathbf{o} = (\mathbf{g} \odot \mathbf{u}) \cdot W_e^{\text{down}\top}$$

4. Scatter weighted outputs $w_{t,j} \cdot \mathbf{o}_t$ back to per-token accumulators.

**Phase 3: Active Mask Application.** When SDD marks token $t$ as frozen (via the `active_mask` boolean tensor), all its expert computations are skipped — the token's dispatch entries are removed from $D$ before Phase 2.

### 3.2.3 Implementation

We provide two backends:

- **Triton JIT kernel** (`_spec_fused_moe_kernel`): Parallelizes over (expert, output_tile) pairs with per-block reductions. Requires Triton ≥ 3.0. The kernel fuses gate/up projection, SiLU activation, element-wise multiply, and down projection into a single launch, minimizing GPU memory traffic.

- **PyTorch reference** (`SpecFusedMoEFunction`): For debugging and portability. Uses explicit loops over unique experts with `torch.index_select` for gather/scatter.

Weight layout follows vLLM convention: `w1` has shape $[E, 2N, D]$ (interleaved gate and up projections) and `w2` has shape $[E, D, N]$ (down projection), where $E = 128$, $N = 768$, $D = 2048$.

### 3.2.4 MAF Reduction Analysis

For $K+1$ tokens in the verify batch, the theoretical MAF is:
$$\text{MAF}(K) = \frac{N}{k}\left(1 - \left(1-\frac{k}{N}\right)^{K+1}\right)$$

SpecFusedMoE achieves exactly this deduplication, reducing the number of expert loads from $(K+1)\cdot k$ (naive) to $k \cdot \text{MAF}(K)$ (deduplicated). The **dedup ratio** is:

$$\text{Dedup} = 1 - \frac{\text{MAF}(K)}{K+1}$$

For $K=3$: $\text{Dedup} = 1 - 3.66/4 = 8.4\%$. While modest per-token, this translates to saving $\sim$26 expert loads per layer (from 32 to $\sim$29.3), compounding across 48 layers to save $\sim$125 expert loads per verify step.

## 3.3 Speculation Divergence Detection (SDD)

### 3.3.1 Motivation

In standard speculative decoding, all $K$ draft tokens pass through the full target model (all 48 MoE layers) before acceptance is decided. However, tokens that will ultimately be rejected often exhibit detectable routing divergence early in the network. SDD exploits this by monitoring divergence signals layer-by-layer and "freezing" tokens whose router outputs strongly diverge from the draft model's predictions.

### 3.3.2 Divergence Metrics

SDD supports three divergence metrics, each capturing a different aspect of routing disagreement:

**Entropy-based:** For router logits $\mathbf{r} \in \mathbb{R}^N$ at layer $l$:
$$d_{\text{entropy}} = 1 - \frac{H(\text{softmax}(\mathbf{r}))}{\log N}$$
where $H(\cdot)$ is Shannon entropy. Low entropy indicates high confidence in a small subset of experts — divergent from the diffuse routing expected for "easy" tokens.

**Top-$k$ overlap:** Between draft and target router selections at layer $l$:
$$d_{\text{overlap}} = 1 - \frac{|E_{\text{draft}}^l \cap E_{\text{target}}^l|}{|E_{\text{draft}}^l \cup E_{\text{target}}^l|}$$
where $E_{\text{draft}}^l, E_{\text{target}}^l$ are the top-$k$ expert sets. This directly measures whether the draft model predicted the right experts.

**KL divergence:** Between draft and target router probability distributions:
$$d_{\text{KL}} = D_{\text{KL}}(\text{softmax}(\mathbf{r}_{\text{target}}^l) \| \text{softmax}(\mathbf{r}_{\text{draft}}^l))$$

**Combined mode** (default) averages the normalized scores of all available metrics.

### 3.3.3 Freeze Decision

For each draft token $t$ in the verify batch, SDD maintains a `TokenDivergenceState`:

```
consecutive_divergent: int = 0
frozen: bool = False
frozen_at_layer: int = -1
```

At each MoE layer $l \geq l_{\min}$ (default $l_{\min} = 8$, skipping early layers whose routing is less informative):

1. Compute divergence score $d_t^l$ using the selected metric.
2. If $d_t^l$ exceeds the threshold: increment `consecutive_divergent`.
3. If `consecutive_divergent` $\geq \tau_c$ (default $\tau_c = 3$): set `frozen = True`, record `frozen_at_layer = l`.

A frozen token is excluded from all subsequent MoE layers via the `active_mask` passed to SpecFusedMoE. This saves:
$$\text{Saved loads}(t) = (L - l_{\text{freeze}}) \times k$$
expert loads per frozen token. For a token frozen at layer 20 (of 48), this saves $28 \times 8 = 224$ expert loads.

### 3.3.4 Configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_check_layer` | 8 | Start checking after layer 8 |
| `kl_threshold` | 2.0 | KL divergence threshold |
| `entropy_threshold` | 1.5 | Normalized entropy threshold |
| `topk_overlap_threshold` | 0.25 | Minimum Jaccard overlap |
| `consecutive_threshold` | 3 | Consecutive divergences to freeze |
| `method` | `"combined"` | Metric: kl / overlap / entropy / combined |

### 3.3.5 SDD–SpecFusedMoE Interaction

SDD's `active_mask` integrates directly with SpecFusedMoE's Phase 3. When SpecFusedMoE receives `active_mask[t] = False`, token $t$'s entries are pruned from the dispatch table — its experts are not loaded (unless needed by other active tokens). This creates a compound effect: SDD reduces both the **number of tokens** processed and the **effective MAF** (frozen tokens' experts no longer inflate the unique expert count).

## 3.4 Acceptance-Aware Expert Prefetch

### 3.4.1 Motivation

Expert cache misses dominate latency in CPU-offloaded MoE inference. Each miss incurs a synchronous PCIe transfer of ~9.4 MB (one expert in bf16), taking ~0.38 ms at PCIe Gen4 ×16 bandwidth. With 29.3 unique experts per layer and 48 layers, cold-start verify cost is $29.3 \times 48 \times 0.38 \approx 534$ ms — dwarfing the ~15 ms compute cost.

SpecMoE's expert cache maintains a tiered GPU/CPU architecture with proactive prefetching from draft-model routing predictions.

### 3.4.2 Cache Architecture

```
┌─────────────────────────────────────────┐
│              GPU HBM (8 GB)             │
│  ┌──────────────────────────────────┐   │
│  │   LRU Cache                      │   │
│  │   (layer, expert) → {w1, w2}    │   │
│  │   ~850 expert slots              │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │   Prefetch Staging Buffer        │   │
│  │   (async cudaMemcpyAsync)        │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    ▲
                    │ PCIe Gen4 ×16 (~25 GB/s)
                    ▼
┌─────────────────────────────────────────┐
│        CPU Pinned Memory (~57.6 GB)     │
│  Full expert weight store (read-only)   │
│  (layer, expert) → {w1, w2}            │
└─────────────────────────────────────────┘
```

With 8 GB GPU cache budget and ~9.4 MB per expert, the cache holds ~850 expert slots. Given 48 layers × 128 experts = 6,144 total experts, this is a 13.8% cache ratio. Effective hit rates are much higher due to temporal locality in autoregressive generation.

### 3.4.3 Acceptance-Aware Prefetch Policy

During the draft phase, the draft model's router selects experts at each layer. These selections are excellent predictors of which experts the target model will need during verify (since the draft model is trained to mimic the target model's behavior). SpecMoE's prefetch policy scores each candidate expert:

$$\text{score}(e, l) = p_{\text{need}} \cdot p_{\text{accept}} \cdot b_{\text{benefit}} - c_{\text{cost}} + \text{bonuses} - \text{penalties}$$

where:
- $p_{\text{need}}$ — probability the expert is actually needed (from draft router softmax)
- $p_{\text{accept}}$ — current acceptance rate (from AcceptanceTracker EMA)
- $b_{\text{benefit}}$ — transfer latency saved if prefetch hits
- $c_{\text{cost}}$ — opportunity cost of evicting a cached expert

**Bonuses and penalties:**
- **Shared expert bonus** (+0.20): experts selected by multiple tokens in the batch
- **Reuse bonus** (+0.15): experts accessed in previous iterations
- **Depth penalty** (−0.12 per layer ratio): deep-layer experts are penalized (more likely to be skipped by SDD)
- **Isolation penalty** (−0.18): experts not shared across tokens and deep in the network
- **Waste penalty** (−1.0 per waste ratio): penalizes prefetching experts that were previously wasted

Based on the final score, each candidate is classified:

| Score range | Action | Semantics |
|-------------|--------|-----------|
| $\geq 0.60$ | **Hard prefetch** | Immediate async transfer to GPU |
| $[0.35, 0.60)$ | **Soft prefetch** | Queued, initiated if bandwidth permits |
| $< 0.35$ | **Defer** | Not prefetched; loaded on demand if needed |

### 3.4.4 Cache Performance

The prefetch depth (default: 16 layers ahead) determines how far into the future the cache looks. Combined with LRU eviction (or frequency-based eviction), the cache reports:
- **Hit rate**: fraction of `get_expert` calls served from GPU cache
- **Prefetch accuracy**: fraction of prefetched experts actually used during verify
- **Average transfer time**: mean PCIe transfer latency for cache misses

## 3.5 Phase-Aware Control Plane

### 3.5.1 Controller Architecture

SpecMoE's control plane adapts key parameters in real time based on system state. The **RuntimeState** captures:

```python
RuntimeState:
  phase: PREFILL | DECODE
  gpu_mem_used_mb / gpu_mem_total_mb
  kv_cache_mb
  acceptance_rate (EMA)
  step_id
```

The **Phase-Aware Governor** makes three coordinated decisions per step:

**1. Speculation depth ($K$):** Phase-dependent bounds with pressure-aware clamping.
- PREFILL: $K \leq 1$ (conservative, prioritize TTFT)
- DECODE: $K \leq 4$ (aggressive, maximize throughput)
- GPU pressure $\geq 90\%$: $K \leq 1$
- Acceptance rate $< 35\%$: $K \leq 1$

**2. Memory partition:** Dynamic GPU budget allocation among three consumers.

| Pressure | Expert cache | Speculative buffer | KV cache |
|----------|-------------|-------------------|----------|
| Low ($<80\%$) | 28% | 18% | 54% |
| Medium ($80\text{–}90\%$) | 26% | 14% | 60% |
| High ($\geq 90\%$) | 22% | 10% | 68% |

Exponential smoothing ($\alpha = 0.5$) prevents oscillation between allocation regimes.

**3. Prefetch aggressiveness:** Adjusts hard/soft thresholds based on cache hit rate and remaining GPU budget.

### 3.5.2 Acceptance Tracking

The **AcceptanceTracker** maintains three complementary rate estimates:

| Metric | Window | Update rule | Use case |
|--------|--------|-------------|----------|
| EMA rate | Infinite | $\rho \leftarrow (1-\alpha)\rho + \alpha \cdot \rho_{\text{inst}}$ | Smooth trend for $K$ adaptation |
| Window rate | Last 128 verifications | $\sum a_i / \sum p_i$ | Recent performance for threshold checks |
| Global rate | All time | $A_{\text{total}} / P_{\text{total}}$ | Reporting and debugging |

A warmup guard (default: 8 rounds) prevents premature $K$ adaptation before statistics stabilize.

## 3.6 vLLM Integration

### 3.6.1 Zero-Modification Hook

SpecMoE integrates into vLLM via a `FusedMoEHook` that monkey-patches the `vllm.model_executor.layers.fused_moe.fused_moe` function at runtime:

```python
# Installation
original_fn = vllm.fused_moe.fused_moe
vllm.fused_moe.fused_moe = wrapped_fn

# Wrapped function (simplified)
def wrapped_fn(hidden_states, w1, w2, topk_weights, topk_ids, **kw):
    if verify_mode_enabled:
        return spec_moe_dispatch(hidden_states, w1, w2,
                                 topk_weights, topk_ids, active_mask)
    else:
        return original_fn(hidden_states, w1, w2,
                           topk_weights, topk_ids, **kw)
```

The hook operates in two modes:
- **Passthrough** (default): During normal inference and draft model execution, calls the original vLLM fused_moe.
- **SpecMoE**: During verify phase only, routes through SpecFusedMoE with SDD integration and expert caching.

This design is non-intrusive — model definitions, attention layers, and the scheduler remain unmodified.

### 3.6.2 Engine Lifecycle

The `SpecMoEEngine` orchestrator manages the full lifecycle:

| Phase | Operation | Components involved |
|-------|-----------|-------------------|
| **Initialization** | `initialize(model)` — Create dispatcher, SDD, cache; register expert weights; install hook | All |
| **Draft step** | `on_draft_step(routing, probs)` — Forward draft routing to prefetch engine | PrefetchPolicy, ExpertCache |
| **Verify begin** | `begin_verify(K, mask)` — Activate SpecMoE mode in hook | FusedMoEHook |
| **Layer forward** | (Automatic via hook) — Dedup dispatch + SDD check + cache lookup per layer | SpecFusedMoE, SDD, ExpertCache |
| **Verify end** | `end_verify(accepted, proposed)` — Update acceptance tracker, log statistics | AcceptanceTracker, Governor |
| **Shutdown** | `shutdown()` — Uninstall hooks, flush traces, persist statistics | All |

### 3.6.3 Trace Collection

When enabled, the `MoETraceCollector` records per-verify-round events:
- Expert IDs selected per token per layer
- Cache hit/miss per expert
- SDD freeze decisions
- Acceptance outcomes

These traces enable offline analysis of expert locality patterns, SDD precision/recall, and cache efficiency — critical for tuning SpecMoE parameters for new models.
