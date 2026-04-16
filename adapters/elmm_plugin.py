"""
Expert-Level Memory Management (ELMM) Plugin for vLLM
======================================================
Replaces vLLM's UVA-based CPU offloading with explicit GPU expert cache
+ async H2D transfers, making Speculative Decoding viable on offloaded MoE.

Key insight: vLLM's UVA offloading creates CUDA views of CPU pinned memory
via ``get_accelerator_view_from_cpu_tensor()``. The Triton FusedMoE kernel
accesses only activated expert rows (``b_ptr + off_experts * stride_be``),
causing PCIe transfers for every touched expert on every step with NO
cross-step caching. MAF theory shows this causes ~3.6x redundant PCIe
traffic across K+1 speculative decode tokens.

ELMM architecture:
  1. Shared GPU scratchpad: One full-size [E, ...] GPU buffer shared
     across all offloaded layers (layers execute sequentially).
  2. Per-layer LRU expert cache: Persistent across decode steps.
     GPU budget is distributed across layers.
  3. Scratchpad-swap protocol: Before each kernel call, write needed
     experts from cache (fast HBM copy) or CPU (PCIe miss) into
     scratchpad, swap param.data, run kernel, restore param.data.

This module is activated from the vLLM server launch script by calling
``activate_elmm()`` after model loading in the GPU worker process.
"""

import contextlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ELMMConfig:
    """Configuration for expert-level memory management."""
    # GPU budget for expert cache (bytes). Default 8 GB.
    gpu_cache_budget_bytes: int = 8 * 1024**3
    # Use a dedicated CUDA stream for async prefetch.
    use_prefetch_stream: bool = True
    # Enable draft-guided prefetching.
    enable_prefetch: bool = True
    # Log per-step cache stats every N intercepts (0=disabled).
    log_interval: int = 0
    # --- Temporal Locality Collection ---
    # Enable structured locality data collection (overlap, reuse distance).
    enable_locality_collection: bool = True
    # Export locality data to this directory (empty = disabled).
    locality_export_dir: str = ""
    # Export every N verify rounds (0 = only on shutdown).
    locality_export_interval: int = 0
    # --- Adaptive Cache Budget ---
    # Enable adaptive per-layer cache budget rebalancing.
    enable_adaptive_budget: bool = True
    # Rebalance every N intercepts (per-layer). Higher = less overhead.
    rebalance_interval: int = 5000
    # Minimum fraction of total slots any layer can hold (prevents starvation).
    min_slot_fraction: float = 0.02
    # EMA alpha for smoothing per-layer hit rates.
    hit_rate_ema_alpha: float = 0.05
    # --- Pool-Direct Mode ---
    # Skip scratchpad copies for cache hits by pointing kernel directly
    # at the per-layer cache pool with remapped topk_ids.
    enable_pool_direct: bool = True
    # --- Direct Triton Dispatch ---
    # Bypass quant_method.apply() call chain and invoke Triton kernels
    # directly, eliminating ~8 levels of Python indirection per layer.
    # Only active when pool-direct is also enabled and batch is small.
    enable_direct_dispatch: bool = True
    # --- GPU-Side Cache Lookup ---
    # Keep cache mapping on GPU tensors to avoid GPU→CPU sync in the
    # hot path. Falls back to CPU-side eviction only on cache miss.
    enable_gpu_cache: bool = False  # E1 experiment: -18.5% regression, disabled
    # --- Stale-Remap Fast Path ---
    # After warmup, reuse per-layer remap tables and skip Phase 3
    # (unique + tolist + dict lookup + scatter) for N-1 out of N steps.
    # On every Nth step, do full validation. Trades ~0.1% correctness
    # for ~8ms/step savings (eliminates GPU→CPU sync on 93% of calls).
    stale_remap_interval: int = 0  # 0 = disabled, N > 0 = validate every N steps
    stale_remap_warmup: int = 32   # full Phase 3 for first N calls per layer (reduced: cache stabilises in ~2 steps)
    stale_remap_max_interval: int = 128  # adaptive: max interval before forced validation
    # --- Phase Profiling ---
    # Record CUDA event timing for each forward phase. Prints summary
    # after profile_warmup + profile_steps intercepts, then disables.
    enable_phase_profiling: bool = False
    profile_warmup: int = 200      # skip initial intercepts
    profile_steps: int = 100       # measure this many intercepts
    # --- CUDA Graph (ALPS Phase 1) ---
    # Capture TASER stale path as CUDA Graph to eliminate kernel launch
    # overhead (~7.5μs × 5 kernels × 26 layers = ~1ms/step savings).
    # Only active when TASER stale-remap and direct dispatch are enabled.
    # Disable with ELMM_CUDA_GRAPH=0 or --enforce-eager.
    # NOTE: Disabled by default — per-layer graph requires 3 placeholder
    # copy_() operations per layer per step, adding 78 extra kernel launches
    # that largely cancel the 26 saved graph replays. Net ~2%, often offset
    # by graph pool overhead. Keep for future work (whole-model graph capture).
    enable_cuda_graph: bool = False
    # --- SharedExpert Parallelization (ALPS Phase 3) ---
    # Launch SharedExpert on a separate CUDA stream concurrently with the
    # MoE kernel.  SharedExpert is compute-light (~0.035 ms) and completes
    # well before the HBM-bound MoE kernel (~0.35 ms), so the overlap is
    # nearly free.  Saves ~0.64 ms/step (= 0.035 × 0.7 × 26 layers).
    enable_shared_parallel: bool = True
    # --- Oracle Cross-Layer Prefetch (ALPS Phase 2) ---
    # During Layer N's MoE kernel (HBM-bound), prefetch experts for Layer
    # N+1 via PCIe on a separate stream.  Uses the frozen TASER remap table
    # to predict N+1's needed experts (deterministic in stale path).
    # Mainly benefits warmup / prefill (cold cache); during stale path the
    # cache is already populated so prefetch is a no-op.
    enable_oracle_prefetch: bool = True
    # --- HFDE (Hit-First Disaggregated Execution) ---
    # When cache misses occur, split MoE execution into two passes:
    #   Pass 1 (hit): compute cached experts while miss experts load async
    #   Pass 2 (miss): compute miss experts after loading completes
    # Overlaps PCIe miss-loading with hit-expert GPU compute.
    # Requires pool_direct + direct_dispatch.  Only activates when there
    # are both hits AND misses (common during TASER validation steps).
    enable_hfde: bool = True
    # --- Stacked Gating Predictor (P1) ---
    # Build inter-layer co-activation frequency tables during warmup,
    # then use Layer N's actual routing to predict Layer N+1's experts.
    # Replaces the naive cache-overlap heuristic in oracle prefetch.
    enable_stacked_gating: bool = True
    # Top-K co-activated experts to predict per source expert.
    stacked_gating_top_k: int = 4
    # --- RWAWE Eviction (P2) ---
    # Replace pure LRU with Recency-Weighted Access-Weighted Eviction.
    # Score(e) = alpha * exp(-lambda * age) + (1-alpha) * ema_freq(e)
    # Evicts lowest-score expert instead of least-recently-used.
    enable_rwawe: bool = True
    rwawe_alpha: float = 0.6   # weight toward recency (0-1)
    rwawe_lambda: float = 0.1  # recency exponential decay rate
    rwawe_beta: float = 0.05   # EMA frequency update coefficient
    # --- Draft-Utility Signal (P3) ---
    # Weight RWAWE frequency updates by per-expert utility: experts activated
    # by more verify tokens get higher frequency boost (proxy for accept likelihood).
    enable_draft_utility: bool = True
    # --- Entropy-Aware Budget (P4) ---
    # Weight rebalance allocation by per-layer routing entropy (unique experts EMA).
    # High-entropy layers get proportionally more cache slots.
    enable_entropy_budget: bool = True
    entropy_ema_gamma: float = 0.05  # EMA coefficient for unique experts tracking
    entropy_kappa: float = 1.0       # entropy influence exponent
    # --- TASER v2: Dual-Rail Hot Expert Routing ---
    # Tolerate partial misses in cruise mode by splitting execution into
    # hit (pool-direct) and miss (HFDE async) passes.
    taser_miss_budget_ratio: float = 0.15   # max miss ratio per step in cruise
    taser_converge_threshold: float = 0.90  # cache-hot_set alignment to enter cruise
    taser_converge_max_steps: int = 10      # max steps in converge before force cruise
    taser_drift_ema_alpha: float = 0.1      # EMA smoothing for miss ratio
    taser_drift_miss_threshold: float = 0.3 # miss ratio EMA triggering drift
    taser_drift_warmup: int = 10            # re-warmup steps during drift
    taser_cold_slots: int = 3               # reserved slots for miss experts in cruise
    # --- BriskMoE Integration ---
    # Enable SACR (Speculation-Aware Cache Replacement) eviction policy.
    enable_sacr: bool = False
    # Enable ELP (Expert Lifecycle Partitioning) pin/flex zones.
    enable_elp: bool = False
    # Enable DIPP (Draft-Informed Prioritized Preloading) for prefetch.
    enable_dipp: bool = False
    # Enable PredCache (Predictive Expert Cache Management).
    # Forward-looking eviction using router logits as Belady OPT approximation.
    enable_pred_cache: bool = False
    # PredCache LRU fallback weight (λ)
    pred_cache_lru_weight: float = 10.0
    # --- ACES (Adaptive Cache with Expert Scoring) ---
    # Use router softmax EMA as eviction signal instead of LRU.
    enable_aces: bool = False
    aces_beta: float = 0.9  # EMA momentum
    # ACES-guided TASER mapping: use EMA top-K to build remap table
    # at validation time, proactively swapping in high-EMA experts.
    aces_taser: bool = False
    # SACR alpha/beta/gamma weights
    sacr_alpha: float = 0.3
    sacr_beta: float = 0.2
    sacr_gamma: float = 0.5
    # ELP config
    elp_pin_ratio: float = 0.7
    elp_promotion_threshold: int = 5
    elp_demotion_window: int = 50
    # --- Unified Scheduling Framework (D1-D6) ---
    # Master switch: enables partial-resident experts + unified pipeline.
    # When True, bypasses TASER/HFDE/BriskMoE and uses D1-D6 pipeline.
    enable_unified_scheduling: bool = False
    # D3: Expert split ratio θ (head portion of intermediate dimension)
    unified_split_ratio: float = 0.4
    # D4: Tail pool slots per layer
    unified_tail_slots: int = 8
    # D4: GPU budget for D4 pools (bytes). 0 = auto-detect from free memory.
    unified_budget_bytes: int = 0
    # D1: Governor mode ("rule" | "fixed")
    unified_governor_mode: str = "rule"
    # D1: Speculation depth bounds
    unified_K_min: int = 1
    unified_K_max: int = 5
    unified_K_default: int = 3
    # D2: Verify decomposition pattern ("auto" | "2+2" | "none" etc.)
    unified_verify_pattern: str = "auto"
    # D6: Prefetch aggressiveness (0.3..1.0)
    unified_prefetch_aggressiveness: float = 0.7


# ---------------------------------------------------------------------------
# Per-layer GPU expert cache (LRU)
# ---------------------------------------------------------------------------

class _LayerExpertCache:
    """
    LRU cache for individual expert weights on GPU.

    Uses a pre-allocated GPU pool to avoid lazy allocation (which
    would conflict with vLLM's KV cache memory budget calculation).
    Pool is allocated during install() before vLLM profiles GPU memory.
    """

    __slots__ = (
        "layer_name", "_slot_map", "_free_slots", "_lru_order",
        "_w13_pool", "_w2_pool", "_max_slots",
        "_evictions", "_hits", "_misses",
        "_w13_scale_pool", "_w2_scale_pool",
        "_w13_bias_pool", "_w2_bias_pool",
        # RWAWE (P2)
        "_rwawe_enabled", "_rwawe_alpha", "_rwawe_lambda", "_rwawe_beta",
        "_last_access", "_ema_freq", "_rwawe_step", "_num_experts",
    )

    def __init__(
        self,
        layer_name: str,
        max_slots: int,
        w13_single_shape: tuple,
        w2_single_shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        w13_scale_shape: tuple | None = None,
        w2_scale_shape: tuple | None = None,
        w13_bias_shape: tuple | None = None,
        w2_bias_shape: tuple | None = None,
        scale_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        # RWAWE (P2)
        num_experts: int = 128,
        rwawe_enabled: bool = False,
        rwawe_alpha: float = 0.6,
        rwawe_lambda: float = 0.1,
        rwawe_beta: float = 0.05,
    ):
        self.layer_name = layer_name
        self._max_slots = max_slots
        # Pre-allocate GPU pool for expert weight storage
        self._w13_pool = torch.empty(
            (max_slots, *w13_single_shape), dtype=dtype, device=device
        )
        self._w2_pool = torch.empty(
            (max_slots, *w2_single_shape), dtype=dtype, device=device
        )
        # Optional scale/bias pools for quantized models (e.g. MXFP4 Marlin)
        if w13_scale_shape is not None:
            sdtype = scale_dtype or dtype
            self._w13_scale_pool = torch.empty(
                (max_slots, *w13_scale_shape), dtype=sdtype, device=device
            )
            self._w2_scale_pool = torch.empty(
                (max_slots, *w2_scale_shape), dtype=sdtype, device=device
            )
        else:
            self._w13_scale_pool = None
            self._w2_scale_pool = None
        if w13_bias_shape is not None:
            bdtype = bias_dtype or dtype
            self._w13_bias_pool = torch.empty(
                (max_slots, *w13_bias_shape), dtype=bdtype, device=device
            )
            self._w2_bias_pool = torch.empty(
                (max_slots, *w2_bias_shape), dtype=bdtype, device=device
            )
        else:
            self._w13_bias_pool = None
            self._w2_bias_pool = None
        # expert_id → slot_index
        self._slot_map: OrderedDict[int, int] = OrderedDict()
        # Available slot indices
        self._free_slots: list[int] = list(range(max_slots))
        self._evictions = 0
        self._hits = 0
        self._misses = 0
        # RWAWE (P2) tracking
        self._num_experts = num_experts
        self._rwawe_enabled = rwawe_enabled
        self._rwawe_alpha = rwawe_alpha
        self._rwawe_lambda = rwawe_lambda
        self._rwawe_beta = rwawe_beta
        self._last_access = [-1.0] * num_experts
        self._ema_freq = [0.0] * num_experts
        self._rwawe_step = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._slot_map

    def get(self, expert_id: int, utility: float = 1.0) -> int | None:
        """Returns slot index or None. Promotes on hit."""
        if expert_id in self._slot_map:
            self._slot_map.move_to_end(expert_id)
            self._hits += 1
            # RWAWE (P2): update access tracking, weighted by utility (P3)
            if self._rwawe_enabled:
                self._last_access[expert_id] = self._rwawe_step
                beta = self._rwawe_beta
                self._ema_freq[expert_id] = (
                    beta * utility + (1 - beta) * self._ema_freq[expert_id]
                )
            return self._slot_map[expert_id]
        self._misses += 1
        return None

    def advance_step(self) -> None:
        """Advance RWAWE step counter. Call once per forward pass per layer."""
        self._rwawe_step += 1

    def _rwawe_victim(self, exclude: set) -> int | None:
        """Select lowest-RWAWE-score cached expert as eviction victim."""
        import math
        best_victim = None
        best_score = float('inf')
        alpha = self._rwawe_alpha
        lam = self._rwawe_lambda
        step = self._rwawe_step
        for eid in self._slot_map:
            if eid in exclude:
                continue
            la = self._last_access[eid]
            age = step - la if la >= 0 else step + 1
            r = math.exp(-lam * age)
            f = self._ema_freq[eid]
            score = alpha * r + (1 - alpha) * f
            if score < best_score:
                best_score = score
                best_victim = eid
        return best_victim

    def get_slot_tensors(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (w13_view, w2_view) for the given slot index."""
        return self._w13_pool[slot], self._w2_pool[slot]

    @property
    def has_aux_pools(self) -> bool:
        """True if this cache has scale/bias pools (quantized models)."""
        return self._w13_scale_pool is not None or self._w13_bias_pool is not None

    def get_slot_scale_tensors(
        self, slot: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (w13_scale_view, w2_scale_view) or (None, None)."""
        if self._w13_scale_pool is None:
            return None, None
        return self._w13_scale_pool[slot], self._w2_scale_pool[slot]

    def get_slot_bias_tensors(
        self, slot: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (w13_bias_view, w2_bias_view) or (None, None)."""
        if self._w13_bias_pool is None:
            return None, None
        return self._w13_bias_pool[slot], self._w2_bias_pool[slot]

    def alloc_slot(self, expert_id: int, utility: float = 1.0) -> int:
        """Allocate a slot for this expert, evicting LRU/RWAWE if needed."""
        if expert_id in self._slot_map:
            self._slot_map.move_to_end(expert_id)
            return self._slot_map[expert_id]
        if not self._free_slots:
            if self._rwawe_enabled:
                victim = self._rwawe_victim(exclude=set())
                if victim is not None:
                    evict_slot = self._slot_map.pop(victim)
                else:
                    _, evict_slot = self._slot_map.popitem(last=False)
            else:
                # Evict LRU entry
                _, evict_slot = self._slot_map.popitem(last=False)
            self._free_slots.append(evict_slot)
            self._evictions += 1
        slot = self._free_slots.pop()
        self._slot_map[expert_id] = slot
        # RWAWE: update tracking, weighted by utility (P3)
        if self._rwawe_enabled:
            self._last_access[expert_id] = self._rwawe_step
            beta = self._rwawe_beta
            self._ema_freq[expert_id] = (
                beta * utility + (1 - beta) * self._ema_freq[expert_id]
            )
        return slot

    def alloc_slot_with_victim(
        self, expert_id: int, victim_eid: int | None = None
    ) -> tuple[int, int | None]:
        """Allocate with optional directed eviction. Returns (slot, evicted_eid)."""
        if expert_id in self._slot_map:
            self._slot_map.move_to_end(expert_id)
            return self._slot_map[expert_id], None
        evicted = None
        if not self._free_slots:
            if victim_eid is not None and victim_eid in self._slot_map:
                evict_slot = self._slot_map.pop(victim_eid)
                evicted = victim_eid
            else:
                # Fallback to LRU
                evicted, evict_slot = self._slot_map.popitem(last=False)
            self._free_slots.append(evict_slot)
            self._evictions += 1
        slot = self._free_slots.pop()
        self._slot_map[expert_id] = slot
        return slot, evicted

    def resize(self, new_max_slots: int) -> int:
        """
        Resize cache logical capacity. Returns actual new capacity.

        Uses logical-only resize: adjusts the capacity limit without
        reallocating GPU tensors (the physical pool stays the same size).
        If shrinking, evicts LRU entries that exceed the new limit.
        If growing beyond physical pool, caps at physical pool size.
        """
        if new_max_slots == self._max_slots:
            return self._max_slots
        if new_max_slots < 1:
            new_max_slots = 1

        # Cap at physical pool size (no tensor reallocation)
        physical_max = self._w13_pool.shape[0]
        new_max_slots = min(new_max_slots, physical_max)

        if new_max_slots < self._max_slots:
            # Shrink: evict LRU entries that exceed the new limit
            while len(self._slot_map) > new_max_slots:
                _eid, freed_slot = self._slot_map.popitem(last=False)
                self._free_slots.append(freed_slot)
                self._evictions += 1
            # Remove free slots that are beyond the new limit
            self._free_slots = [s for s in self._free_slots if s < new_max_slots]

        elif new_max_slots > self._max_slots:
            # Grow: add newly available slots from existing physical pool
            for s in range(self._max_slots, new_max_slots):
                if s not in self._slot_map.values():
                    self._free_slots.append(s)

        self._max_slots = new_max_slots
        return self._max_slots

    def reset_hit_counters(self):
        """Reset hit/miss counters (keep cached data)."""
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# Per-layer CUDA Graph for TASER stale path (ALPS Phase 1)
# ---------------------------------------------------------------------------

class _LayerCUDAGraph:
    """
    CUDA Graph capturing the Direct Dispatch kernel sequence for one layer.

    In TASER stale path, the 5-kernel sequence (moe_align + W1 GEMM + silu +
    W2 GEMM + sum) has STATIC shapes, static pool pointers, and static tile
    config.  Only the VALUES of hidden_states, topk_weights, and remapped_ids
    change between calls.  This makes the path eligible for CUDA Graph capture,
    eliminating ~5 × 7.5 µs = 37.5 µs kernel-launch overhead per layer.

    Usage::

        graph = _LayerCUDAGraph(M=4, top_k=8, hidden_dim=2048, ...)
        graph.capture(forward_fn)           # one-time
        output = graph.replay(h, w, ids)    # hot path
    """

    __slots__ = ('ph_hidden', 'ph_topk_weights', 'ph_remapped_ids',
                 'graph', 'output')

    def __init__(self, M: int, top_k: int, hidden_dim: int,
                 device: torch.device, dtype: torch.dtype):
        self.ph_hidden = torch.zeros(M, hidden_dim, device=device, dtype=dtype)
        self.ph_topk_weights = torch.zeros(M, top_k, device=device, dtype=dtype)
        self.ph_remapped_ids = torch.zeros(M, top_k, device=device, dtype=torch.long)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.output: Optional[torch.Tensor] = None

    def capture(self, forward_fn):
        """Capture *forward_fn(hidden, topk_weights, remapped_ids)*.

        Performs 3 warm-up runs (to JIT-compile / autotune Triton kernels)
        then records the kernel sequence into a ``CUDAGraph``.
        """
        torch.cuda.synchronize()
        # Warm-up: ensures Triton autotune is done and all lazy allocs happen
        for _ in range(3):
            _ = forward_fn(
                self.ph_hidden, self.ph_topk_weights, self.ph_remapped_ids)
        torch.cuda.synchronize()
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = forward_fn(
                self.ph_hidden, self.ph_topk_weights, self.ph_remapped_ids)

    def replay(self, hidden_states, topk_weights, remapped_ids):
        """Copy new values into placeholders and replay the graph.

        Returns the output tensor (lives in the graph's private pool;
        safe to read until the next ``replay`` call on this instance).
        """
        self.ph_hidden.copy_(hidden_states)
        self.ph_topk_weights.copy_(topk_weights)
        self.ph_remapped_ids.copy_(remapped_ids)
        self.graph.replay()
        return self.output


# ---------------------------------------------------------------------------
# ELMMManager — core implementation
# ---------------------------------------------------------------------------

class ELMMManager:
    """
    Manages GPU expert caches and scratchpad for all offloaded FusedMoE layers.

    Lifecycle:
      1. ``install(model)`` — scan model for offloaded FusedMoE layers,
         set up per-layer caches, allocate shared scratchpad, monkey-patch
         ``forward_impl`` on each offloaded FusedMoE instance.
      2. During inference, patched ``forward_impl`` uses scratchpad+swap:
         (a) routing → topk_ids
         (b) for each unique expert: cache hit→copy from cache to scratchpad;
             cache miss→H2D from CPU pinned to scratchpad+cache
         (c) swap param.data to scratchpad, run kernel, restore
      3. ``shutdown()`` — free scratchpad, clear caches, restore originals.
    """

    def __init__(self, config: Optional[ELMMConfig] = None):
        self.config = config or ELMMConfig()
        self._layer_caches: dict[str, _LayerExpertCache] = {}
        # Layer metadata
        self._layer_meta: dict[str, dict[str, Any]] = {}
        # Shared scratchpad (allocated once, reused across layers)
        self._scratch_w13: Optional[torch.Tensor] = None  # [E, 2N, D]
        self._scratch_w2: Optional[torch.Tensor] = None   # [E, D, N]
        self._scratchpad_on_gpu: bool = False  # True when GPU scratchpad allocated
        # Prefetch stream
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if self.config.use_prefetch_stream and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()
        # SharedExpert parallel stream (ALPS Phase 3)
        self._shared_expert_stream: Optional[torch.cuda.Stream] = None
        if self.config.enable_shared_parallel and torch.cuda.is_available():
            self._shared_expert_stream = torch.cuda.Stream()
        # Original forward_impl per (id(module)) for safe restore
        self._original_forward_impls: dict[int, Any] = {}
        # Name → FusedMoE module reference (for restore)
        self._patched_modules: dict[str, nn.Module] = {}
        self._installed = False
        # Statistics
        self._total_intercepts = 0
        self._total_cache_hits = 0
        self._total_cache_misses = 0
        self._total_pcie_bytes = 0
        self._hfde_active_count = 0
        self._dd_active_count = 0
        # Expert trace: layer_name → last step's expert set (for locality analysis)
        self._last_expert_set: dict[str, set[int]] = {}
        # Temporal locality metrics: layer_name → list of overlap ratios
        self._overlap_history: dict[str, list[float]] = {}
        self._overlap_unlimited = os.environ.get("ELMM_OVERLAP_UNLIMITED", "0") == "1"
        self._overlap_dump_path = os.environ.get("ELMM_OVERLAP_DUMP", "")
        # --- Temporal Locality Collection ---
        self._locality_analyzer: Optional[Any] = None
        self._verify_round_counter = 0
        self._current_round_experts: dict[str, set[int]] = {}
        # --- Adaptive Cache Budget ---
        self._hit_rate_ema: dict[str, float] = {}  # layer_name → smoothed hit rate
        self._unique_experts_ema: dict[str, float] = {}  # layer_name → smoothed unique experts/step
        self._rebalance_step = 0
        self._total_cache_slots = 0  # filled during install()
        # --- Draft-Guided Prefetch ---
        self._prefetch_hits = 0
        self._prefetch_total = 0
        self._pending_prefetch: dict[str, set[int]] = {}  # layer_name → set of prefetched eids
        # --- Pool-Direct Mode ---
        self._remap_table: Optional[torch.Tensor] = None  # [num_experts] int64 on GPU
        # --- Direct Dispatch ---
        self._dd_inter_w1: Optional[torch.Tensor] = None
        self._dd_inter_act: Optional[torch.Tensor] = None
        self._dd_inter_w2: Optional[torch.Tensor] = None
        self._dd_tile_config: Optional[dict] = None
        self._dd_top_k: int = 0
        self._dd_activation_name: str = "silu"
        self._dd_silu_and_mul = None  # vLLM's optimized activation
        self._dd_invoke_kernel = None  # cached function ref
        self._dd_align_fn = None       # cached function ref
        self._dd_max_M: int = 64       # max decode tokens
        # --- GPU-Side Cache Lookup ---
        # Per-layer GPU tensors: expert_id → slot_id (-1 = not cached)
        self._gpu_eid_to_slot: dict[str, torch.Tensor] = {}
        # Per-layer GPU LRU clock (monotonic counter per slot)
        self._gpu_lru_clock: dict[str, torch.Tensor] = {}
        self._gpu_cache_step: int = 0  # global monotonic clock
        # --- Stale-Remap Fast Path (TASER: Temporal-Adaptive Stale Expert Routing) ---
        # Per-layer persistent remap tables (GPU, shape [num_experts])
        self._layer_remap: dict[str, torch.Tensor] = {}
        # Per-layer call counters
        self._layer_remap_step: dict[str, int] = {}
        # Per-layer adaptive validation interval (grows when routing stable, shrinks when volatile)
        self._layer_adaptive_interval: dict[str, int] = {}
        # Per-layer next validation step number
        self._layer_next_validation: dict[str, int] = {}
        # --- Phase Profiling ---
        self._prof_events: list = []   # list of (phase_name, start_event, end_event)
        self._prof_phase_ms: dict = {}  # phase → accumulated ms
        self._prof_count: int = 0
        # --- CUDA Graph (ALPS Phase 1) ---
        # Per-layer captured graphs: layer_name → _LayerCUDAGraph
        self._layer_graphs: dict[str, _LayerCUDAGraph] = {}
        # Batch size the graphs were captured for (must match at replay)
        self._graph_M: int = 0
        # Master switch (disabled on capture failure or --enforce-eager)
        # CUDA Graph capture is OFF by default because moe_align_block_size
        # uses dynamic tensor allocations that are incompatible with graph
        # replay (causes illegal memory access on warm-up/capture).
        self._use_cuda_graph: bool = (
            self.config.enable_cuda_graph
            and os.environ.get("ELMM_CUDA_GRAPH", "0") == "1"
        )
        # Diagnostic counters
        self._graph_replay_count: int = 0
        self._graph_eager_count: int = 0
        # --- Oracle Cross-Layer Prefetch (ALPS Phase 2) ---
        # Ordered list of offloaded layer names (populated during install())
        self._ordered_layers: list[str] = []
        # layer_name → index in _ordered_layers for O(1) lookup
        self._layer_index: dict[str, int] = {}
        # Counters for prefetch effectiveness
        self._oracle_prefetch_issued: int = 0
        self._oracle_prefetch_skipped: int = 0
        # Guard: only sync prefetch stream when work was actually issued
        self._prefetch_in_flight: bool = False
        # --- Stacked Gating Predictor (P1) ---
        # Inter-layer co-activation frequency tables (built during warmup)
        # layer_idx → Tensor[num_experts, num_experts] (int32, CPU)
        self._coact_tables: dict[int, torch.Tensor] = {}
        # Pre-computed predictions: layer_idx → {expert_N: [top_k experts at N+1]}
        self._coact_top_k: dict[int, dict[int, list[int]]] = {}
        self._coact_built: bool = False
        self._coact_warmup_steps: int = 0  # counts forward passes (all layers)
        # Track previous layer's experts within a single forward pass
        self._sg_prev_layer_experts: Optional[list[int]] = None
        self._sg_prev_layer_idx: int = -1
        # Prediction accuracy tracking
        self._sg_predict_correct: int = 0
        self._sg_predict_total: int = 0
        # --- BriskMoE Integration ---
        # Per-layer SACR eviction policies
        self._briskmoe_sacr: Optional[Any] = None  # SACREvictionPolicy
        self._aces: Optional[Any] = None  # ACESPolicy
        # Per-layer ELP partitions (shared tracker with SACR)
        self._briskmoe_elp: Optional[Any] = None  # ExpertLifecyclePartition
        # DIPP preloader (shared across layers)
        self._briskmoe_dipp: Optional[Any] = None  # DraftInformedPrioritizedPreloader
        # PredCache manager (forward-looking eviction + prefetch)
        self._pred_cache: Optional[Any] = None  # PredictiveExpertCacheManager
        # Accept/Reject tracker (shared by SACR)
        self._briskmoe_tracker: Optional[Any] = None  # AcceptRejectTracker
        # Mapping layer_name → layer_id (int) for SACR/ELP
        self._layer_name_to_id: dict[str, int] = {}
        # BriskMoE step counter
        self._briskmoe_step: int = 0
        # --- Fused Remap (Phase 3 elimination) ---
        # Per-layer dirty flag: True when cache was modified since last
        # slow-path validation (prefetch/oracle/rebalance).  When False,
        # the per-layer remap table is guaranteed valid and we can skip
        # the safety check (.item() GPU→CPU sync) in the TASER fast path.
        self._layer_cache_dirty: dict[str, bool] = {}
        # --- LDR (Lazy Dirty Resolution) ---
        # When DIPP prefetch updates a layer's cache AND remap table,
        # this flag enables a lightweight GPU-side conflict check instead
        # of forcing the full slow-path validation.  If no routed expert
        # was evicted (remap[eid] != 0 for all routed eids), the updated
        # remap is used directly on the fast path.
        self._layer_dirty_from_dipp: dict[str, bool] = {}
        # --- TASER v2: Dual-Rail Hot Expert Routing ---
        # Per-layer phase: 'warmup' | 'converge' | 'cruise' | 'drift'
        self._taser_phase: dict[str, str] = {}
        # Per-layer hot expert set (top-hot_slots by frequency)
        self._hot_set: dict[str, set[int]] = {}
        # Per-layer expert frequency counter (reset each warmup/drift)
        self._expert_freq: dict[str, dict[int, int]] = {}
        # Per-layer miss ratio EMA for drift detection
        self._miss_ratio_ema: dict[str, float] = {}
        # Per-layer step when converge started
        self._converge_start_step: dict[str, int] = {}
        # Per-layer step when drift started
        self._drift_start_step: dict[str, int] = {}
        # Per-layer miss budget (cold_slots count)
        self._miss_budget: dict[str, int] = {}
        # TASER v2 diagnostic counters
        self._taser_v2_cruise_count: int = 0
        self._taser_v2_dual_rail_count: int = 0
        self._taser_v2_drift_count: int = 0
        # --- Unified Scheduling Framework (D1-D6) state ---
        self._unified_residency: Optional[Any] = None   # ResidencyManager
        self._unified_governor: Optional[Any] = None     # RuleBasedGovernor
        self._unified_scheduler: Optional[Any] = None    # VerifyScheduler
        self._unified_executor: Optional[Any] = None     # HeadTailExecutor
        self._unified_planner: Optional[Any] = None      # PrefetchPlanner (per-layer)
        self._unified_split_config: Optional[Any] = None # SplitConfig
        # Per-layer split weight references (CPU pinned)
        self._head_w13_refs: dict[str, torch.Tensor] = {}
        self._head_w2_refs: dict[str, torch.Tensor] = {}
        self._tail_w13_refs: dict[str, torch.Tensor] = {}
        self._tail_w2_refs: dict[str, torch.Tensor] = {}
        # Unified scheduling stats
        self._unified_step_count: int = 0
        self._unified_fallback_count: int = 0
        self._unified_ready_count: int = 0
        self._unified_current_decision: Optional[Any] = None
        # --- v3.1 Overflow Controller (C1-C6) ---
        self._overflow_controller: Optional[Any] = None  # OverflowController

    # -----------------------------------------------------------------------
    # Installation
    # -----------------------------------------------------------------------

    def install(self, model: nn.Module):
        """
        Scan model for offloaded FusedMoE layers and set up ELMM.

        Detection: vLLM's process_weights_after_loading calls replace_parameter
        on w13/w2, creating NEW Parameter objects that lose the
        _vllm_offloaded_cpu_data attribute. But the underlying .data is still
        the UVA view (same storage). Other params in the module (like _gate)
        retain the attribute. So we check if ANY param in the FusedMoE module
        has the attribute — if so, the entire module was offloaded.
        """
        import sys
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        offloaded_layers: list[tuple[str, FusedMoE]] = []
        total_fused_moe = 0

        for name, module in model.named_modules():
            if not isinstance(module, FusedMoE):
                continue
            total_fused_moe += 1
            w13 = getattr(module, "w13_weight", None)
            w2 = getattr(module, "w2_weight", None)
            if w13 is None or w2 is None:
                continue
            # Detect: if ANY param in this module was offloaded by
            # maybe_offload_to_cpu, then w13/w2 were also offloaded
            # (they just lost the attr due to replace_parameter).
            # Also detect CPU-device weights (e.g. from Marlin conversion
            # patch that moves converted weights to CPU-pinned memory).
            any_offloaded = any(
                hasattr(p, "_vllm_offloaded_cpu_data")
                for p in module.parameters()
            ) or w13.device.type == "cpu" or w2.device.type == "cpu"
            if total_fused_moe <= 3:
                print(f"[ELMM] FusedMoE #{total_fused_moe} '{name}': "
                      f"any_offloaded={any_offloaded}, "
                      f"w13 shape={tuple(w13.shape)}, device={w13.device}",
                      file=sys.stderr, flush=True)
            if not any_offloaded:
                continue
            offloaded_layers.append((name, module))

        print(f"[ELMM] Found {total_fused_moe} FusedMoE layers, "
              f"{len(offloaded_layers)} offloaded",
              file=sys.stderr, flush=True)

        if not offloaded_layers:
            print("[ELMM] No offloaded FusedMoE layers found, nothing to do",
                  file=sys.stderr, flush=True)
            return

        # --- Gather metadata ---
        max_w13_bytes = 0
        max_w2_bytes = 0
        max_w13_shape: tuple = ()
        max_w2_shape: tuple = ()

        for name, module in offloaded_layers:
            w13 = module.w13_weight
            w2 = module.w2_weight

            num_experts = w13.shape[0]
            expert_size = (
                w13[0].nelement() * w13[0].element_size()
                + w2[0].nelement() * w2[0].element_size()
            )

            meta: dict = {
                "num_experts": num_experts,
                "expert_size": expert_size,
                "w13_shape": tuple(w13.shape),
                "w2_shape": tuple(w2.shape),
                "dtype": w13.dtype,
            }

            # Detect per-expert quantization scales/biases (e.g. MXFP4 Marlin)
            w13_scale = getattr(module, "w13_weight_scale", None)
            w2_scale = getattr(module, "w2_weight_scale", None)
            w13_bias = getattr(module, "w13_bias", None)
            w2_bias = getattr(module, "w2_bias", None)

            if w13_scale is not None and w13_scale.dim() >= 2:
                meta["w13_scale_shape"] = tuple(w13_scale.shape)
                meta["w2_scale_shape"] = tuple(w2_scale.shape)
                meta["scale_dtype"] = w13_scale.dtype
                expert_size += (
                    w13_scale[0].nelement() * w13_scale[0].element_size()
                    + w2_scale[0].nelement() * w2_scale[0].element_size()
                )
            if w13_bias is not None and w13_bias.dim() >= 2:
                meta["w13_bias_shape"] = tuple(w13_bias.shape)
                meta["w2_bias_shape"] = tuple(w2_bias.shape)
                meta["bias_dtype"] = w13_bias.dtype
                expert_size += (
                    w13_bias[0].nelement() * w13_bias[0].element_size()
                    + w2_bias[0].nelement() * w2_bias[0].element_size()
                )
            meta["expert_size"] = expert_size
            self._layer_meta[name] = meta

            w13_bytes = w13.nelement() * w13.element_size()
            w2_bytes = w2.nelement() * w2.element_size()
            if w13_bytes > max_w13_bytes:
                max_w13_bytes = w13_bytes
                max_w13_shape = tuple(w13.shape)
            if w2_bytes > max_w2_bytes:
                max_w2_bytes = w2_bytes
                max_w2_shape = tuple(w2.shape)

        # --- Allocate shared GPU scratchpad ---
        ref_dtype = list(self._layer_meta.values())[0]["dtype"]
        device = torch.device("cuda")

        # In unified mode, D4 pools replace the legacy cache; skip heavy allocations.
        # In pool-direct mode, scratchpad is never used (kernel reads from pool
        # directly); skip allocation to save ~1.12 GiB GPU memory.
        _unified_mode = self.config.enable_unified_scheduling
        _skip_scratchpad = _unified_mode or self.config.enable_pool_direct

        if not _skip_scratchpad:
            self._scratch_w13 = torch.empty(
                max_w13_shape, dtype=ref_dtype, device=device
            )
            self._scratch_w2 = torch.empty(
                max_w2_shape, dtype=ref_dtype, device=device
            )
            self._scratchpad_on_gpu = True
        else:
            # Allocate tiny placeholders so code referencing them doesn't crash
            self._scratch_w13 = torch.empty(1, dtype=ref_dtype, device="cpu")
            self._scratch_w2 = torch.empty(1, dtype=ref_dtype, device="cpu")
            self._scratchpad_on_gpu = False
        scratch_mb = (max_w13_bytes + max_w2_bytes) / 1024**2

        # --- Distribute cache budget across layers (pre-allocate GPU pool) ---
        num_layers = len(offloaded_layers)
        per_layer_budget = self.config.gpu_cache_budget_bytes // num_layers
        device = torch.device("cuda")
        total_cache_alloc = 0

        for name, _module in offloaded_layers:
            meta = self._layer_meta[name]
            expert_size = meta["expert_size"]
            if _unified_mode:
                # In unified mode, allocate minimal 1-slot cache (placeholder)
                max_slots = 1
            else:
                max_slots = max(1, per_layer_budget // expert_size)
            w13_shape = meta["w13_shape"]  # (E, dim1, dim2)
            w2_shape = meta["w2_shape"]    # (E, dim1, dim2)
            # Single expert shapes (exclude the E dimension)
            w13_single = w13_shape[1:]
            w2_single = w2_shape[1:]
            cache = _LayerExpertCache(
                layer_name=name,
                max_slots=max_slots,
                w13_single_shape=w13_single,
                w2_single_shape=w2_single,
                dtype=meta["dtype"],
                device=device,
                w13_scale_shape=meta.get("w13_scale_shape", (None,))[1:]
                    if "w13_scale_shape" in meta else None,
                w2_scale_shape=meta.get("w2_scale_shape", (None,))[1:]
                    if "w2_scale_shape" in meta else None,
                w13_bias_shape=meta.get("w13_bias_shape", (None,))[1:]
                    if "w13_bias_shape" in meta else None,
                w2_bias_shape=meta.get("w2_bias_shape", (None,))[1:]
                    if "w2_bias_shape" in meta else None,
                scale_dtype=meta.get("scale_dtype"),
                bias_dtype=meta.get("bias_dtype"),
                # RWAWE (P2)
                num_experts=meta.get("num_experts", 128),
                rwawe_enabled=self.config.enable_rwawe,
                rwawe_alpha=self.config.rwawe_alpha,
                rwawe_lambda=self.config.rwawe_lambda,
                rwawe_beta=self.config.rwawe_beta,
            )
            self._layer_caches[name] = cache
            total_cache_alloc += max_slots * expert_size

        # --- Monkey-patch forward_impl on each offloaded FusedMoE instance ---
        for name, module in offloaded_layers:
            self._original_forward_impls[id(module)] = module.forward_impl
            self._patched_modules[name] = module

            manager = self
            layer_name = name

            def make_patched_forward(mgr, lname):
                def patched_forward_impl(
                    hidden_states: torch.Tensor,
                    router_logits: torch.Tensor,
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                    return mgr._elmm_forward_impl(lname, hidden_states, router_logits)
                return patched_forward_impl

            module.forward_impl = make_patched_forward(manager, layer_name)

        first_meta = list(self._layer_meta.values())[0]
        experts_per_layer = per_layer_budget // first_meta["expert_size"]
        self._installed = True
        self._total_cache_slots = sum(c._max_slots for c in self._layer_caches.values())
        # Initialize adaptive budget EMA for each layer
        for name in self._layer_caches:
            self._hit_rate_ema[name] = 0.5  # neutral start
            self._unique_experts_ema[name] = float(experts_per_layer)  # start at capacity
        # Initialize remap table for pool-direct mode
        if self.config.enable_pool_direct:
            max_experts = max(m["num_experts"] for m in self._layer_meta.values())
            self._remap_table = torch.arange(
                max_experts, dtype=torch.long, device=device
            )
        # Initialize GPU-side cache lookup tensors
        if self.config.enable_gpu_cache and self.config.enable_pool_direct:
            max_experts = max(m["num_experts"] for m in self._layer_meta.values())
            for name, cache in self._layer_caches.items():
                # expert_id → slot_id mapping (-1 = not cached)
                eid_to_slot = torch.full(
                    (max_experts,), -1, dtype=torch.long, device=device
                )
                # LRU timestamp per slot (0 = never used / evictable)
                lru_clock = torch.zeros(
                    cache._max_slots, dtype=torch.long, device=device
                )
                self._gpu_eid_to_slot[name] = eid_to_slot
                self._gpu_lru_clock[name] = lru_clock
            print(f"[ELMM] GPU-side cache lookup enabled ({max_experts} experts, "
                  f"{len(self._layer_caches)} layers)",
                  file=sys.stderr, flush=True)
        # Initialize direct Triton dispatch
        if self.config.enable_direct_dispatch and self.config.enable_pool_direct:
            self._setup_direct_dispatch(offloaded_layers, ref_dtype, device)
        # Initialize per-layer persistent remap tables for stale-remap mode
        if self.config.stale_remap_interval > 0 and self.config.enable_pool_direct:
            max_experts = max(m["num_experts"] for m in self._layer_meta.values())
            for name in self._layer_caches:
                # Initialize to -1 sentinel (uncached); slot 0 is valid
                self._layer_remap[name] = torch.full(
                    (max_experts,), -1, dtype=torch.long, device=device
                )
                self._layer_remap_step[name] = 0
                self._layer_adaptive_interval[name] = self.config.stale_remap_interval
                self._layer_next_validation[name] = self.config.stale_remap_warmup
            print(f"[ELMM] TASER enabled: interval={self.config.stale_remap_interval}, "
                  f"max_interval={self.config.stale_remap_max_interval}, "
                  f"warmup={self.config.stale_remap_warmup}",
                  file=sys.stderr, flush=True)
            # --- TASER v2: Initialize per-layer dual-rail state ---
            for name, cache in self._layer_caches.items():
                cold = min(self.config.taser_cold_slots,
                           cache._max_slots // 4)
                self._taser_phase[name] = 'warmup'
                self._hot_set[name] = set()
                self._expert_freq[name] = {}
                self._miss_ratio_ema[name] = 0.0
                self._converge_start_step[name] = 0
                self._drift_start_step[name] = 0
                self._miss_budget[name] = max(cold, 1)
            print(f"[ELMM] TASER v2 dual-rail enabled: "
                  f"cold_slots={self.config.taser_cold_slots}, "
                  f"miss_budget_ratio={self.config.taser_miss_budget_ratio}, "
                  f"converge_threshold={self.config.taser_converge_threshold}, "
                  f"drift_miss_threshold={self.config.taser_drift_miss_threshold}",
                  file=sys.stderr, flush=True)
        # Build ordered layer list for cross-layer oracle prefetch
        self._ordered_layers = sorted(
            self._layer_caches.keys(),
            key=lambda n: int(n.split(".")[2]) if len(n.split(".")) > 2 else 0,
        )
        self._layer_index = {n: i for i, n in enumerate(self._ordered_layers)}
        if self.config.enable_oracle_prefetch:
            print(f"[ELMM] Oracle prefetch enabled ({len(self._ordered_layers)} layers)",
                  file=sys.stderr, flush=True)
        # --- Stacked Gating: initialize co-activation tables ---
        if self.config.enable_stacked_gating and len(self._ordered_layers) > 1:
            # Determine num_experts from the first layer's metadata
            first_meta = next(iter(self._layer_meta.values()), {})
            num_experts = first_meta.get("num_experts", 128)
            for i in range(len(self._ordered_layers) - 1):
                self._coact_tables[i] = torch.zeros(
                    num_experts, num_experts, dtype=torch.int32)
            print(f"[ELMM] Stacked Gating enabled: "
                  f"{len(self._coact_tables)} layer pairs, "
                  f"{num_experts} experts, "
                  f"top-{self.config.stacked_gating_top_k} predictions",
                  file=sys.stderr, flush=True)
        # --- RWAWE (P2) log ---
        if self.config.enable_rwawe:
            print(f"[ELMM] RWAWE eviction enabled: "
                  f"alpha={self.config.rwawe_alpha}, "
                  f"lambda={self.config.rwawe_lambda}, "
                  f"beta={self.config.rwawe_beta}",
                  file=sys.stderr, flush=True)
        # --- BriskMoE: Initialize SACR + ELP + DIPP ---
        # Build layer_name → layer_id mapping
        for name in self._layer_caches:
            parts = name.split(".")
            layer_id = 0
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                    except ValueError:
                        pass
            self._layer_name_to_id[name] = layer_id

        if self.config.enable_sacr:
            from adapters.accept_reject_tracker import AcceptRejectTracker
            from adapters.sacr import SACRConfig, SACREvictionPolicy
            self._briskmoe_tracker = AcceptRejectTracker()
            sacr_config = SACRConfig(
                alpha=self.config.sacr_alpha,
                beta=self.config.sacr_beta,
                gamma=self.config.sacr_gamma,
                adaptive_gamma=True,
            )
            max_experts = max(m["num_experts"] for m in self._layer_meta.values())
            self._briskmoe_sacr = SACREvictionPolicy(
                config=sacr_config, tracker=self._briskmoe_tracker,
                num_experts=max_experts,
            )
            print(f"[ELMM] BriskMoE SACR enabled (α={sacr_config.alpha}, "
                  f"β={sacr_config.beta}, γ={sacr_config.gamma}, adaptive=True)",
                  file=sys.stderr, flush=True)

        if self.config.enable_elp:
            from adapters.elp import ELPConfig, ExpertLifecyclePartition
            # Use per-layer slot count for ELP total_slots
            sample_slots = list(self._layer_caches.values())[0]._max_slots
            elp_config = ELPConfig(
                pin_ratio=self.config.elp_pin_ratio,
                promotion_threshold=self.config.elp_promotion_threshold,
                demotion_window=self.config.elp_demotion_window,
            )
            max_experts_elp = max(m["num_experts"] for m in self._layer_meta.values())
            self._briskmoe_elp = ExpertLifecyclePartition(
                config=elp_config, total_slots=sample_slots,
                num_experts=max_experts_elp,
            )
            print(f"[ELMM] BriskMoE ELP enabled (pin_ratio={elp_config.pin_ratio}, "
                  f"threshold={elp_config.promotion_threshold}, slots={sample_slots})",
                  file=sys.stderr, flush=True)

        if self.config.enable_dipp:
            from adapters.dipp import DIPPConfig, DraftInformedPrioritizedPreloader
            dipp_config = DIPPConfig()
            self._briskmoe_dipp = DraftInformedPrioritizedPreloader(config=dipp_config)
            print(f"[ELMM] BriskMoE DIPP enabled (max_prefetch={dipp_config.max_prefetch_experts})",
                  file=sys.stderr, flush=True)

        if self.config.enable_pred_cache:
            from adapters.pred_cache import PredCacheConfig, PredictiveExpertCacheManager
            max_experts_pc = max(m["num_experts"] for m in self._layer_meta.values())
            sample_top_k = list(self._layer_meta.values())[0].get("top_k", 8)
            pc_config = PredCacheConfig(
                num_experts=max_experts_pc,
                top_k=sample_top_k,
                lru_fallback_weight=self.config.pred_cache_lru_weight,
            )
            self._pred_cache = PredictiveExpertCacheManager(config=pc_config)
            print(f"[ELMM] BriskMoE PredCache enabled "
                  f"(num_experts={max_experts_pc}, top_k={sample_top_k}, "
                  f"λ={pc_config.lru_fallback_weight})",
                  file=sys.stderr, flush=True)

        if self.config.enable_aces:
            from adapters.aces import ACESConfig, ACESPolicy
            max_experts_ac = max(m["num_experts"] for m in self._layer_meta.values())
            aces_cfg = ACESConfig(beta=self.config.aces_beta,
                                  num_experts=max_experts_ac)
            self._aces = ACESPolicy(config=aces_cfg)
            print(f"[ELMM] ACES enabled (β={aces_cfg.beta}, "
                  f"num_experts={aces_cfg.num_experts})",
                  file=sys.stderr, flush=True)

        # Check if any layer has quant aux pools
        has_aux = any(c.has_aux_pools for c in self._layer_caches.values())
        if has_aux:
            print("[ELMM] Scale/bias pools enabled (quantized model detected)",
                  file=sys.stderr, flush=True)

        msg = (
            f"ELMM installed: {num_layers} offloaded layers, "
            f"scratchpad={'SKIPPED (pool-direct)' if not self._scratchpad_on_gpu else f'{scratch_mb:.0f} MB'}, "
            f"cache={total_cache_alloc / 1024**3:.2f} GB "
            f"(~{experts_per_layer} experts/layer)"
        )
        print(f"[ELMM] {msg}", file=sys.stderr, flush=True)
        logger.info(msg)

        # Auto-dump overlap history on exit if ELMM_OVERLAP_DUMP is set.
        if self._overlap_dump_path:
            import atexit
            atexit.register(self._dump_overlap_history)

        # --- Unified Scheduling Framework (D1-D6) Setup ---
        if self.config.enable_unified_scheduling:
            self._setup_unified_scheduling(offloaded_layers, ref_dtype, device)

    # -----------------------------------------------------------------------
    # Unified Scheduling Framework (D1-D6) Setup
    # -----------------------------------------------------------------------

    def _setup_unified_scheduling(self, offloaded_layers, dtype, device):
        """
        Initialize D1-D6 modules: split weights, create ResidencyManager,
        Governor, VerifyScheduler, HeadTailExecutor, PrefetchPlanner.
        """
        import sys
        from adapters.expert_split import SplitConfig, split_layer_experts
        from adapters.residency_manager import ResidencyConfig, ResidencyManager
        from adapters.governor import GovernorConfig, RuleBasedGovernor
        from adapters.verify_scheduler import VerifySchedulerConfig, VerifyScheduler
        from adapters.head_tail_executor import HeadTailExecutor
        from adapters.prefetch_planner import PrefetchPlanner

        first_meta = next(iter(self._layer_meta.values()))
        num_experts = first_meta["num_experts"]
        w13_shape = first_meta["w13_shape"]  # (E, 2*inter, hidden)
        w2_shape = first_meta["w2_shape"]    # (E, hidden, inter)
        inter_size = w2_shape[2]   # intermediate_size (e.g. 768)
        hidden_size = w2_shape[1]  # hidden_size (e.g. 2048)
        num_layers = len(offloaded_layers)

        # D3: SplitConfig
        split_cfg = SplitConfig(
            split_ratio=self.config.unified_split_ratio,
            intermediate_size=inter_size,
            hidden_size=hidden_size,
        )
        self._unified_split_config = split_cfg
        print(f"[UNIFIED] D3 SplitConfig: θ={split_cfg.split_ratio}, "
              f"head_inter={split_cfg.head_inter}, tail_inter={split_cfg.tail_inter}, "
              f"head={split_cfg.head_expert_bytes/1e6:.1f}MB, "
              f"tail={split_cfg.tail_expert_bytes/1e6:.1f}MB",
              file=sys.stderr, flush=True)

        # D4: ResidencyManager
        # Auto-detect GPU budget: use configured value or measure free memory
        d4_budget = self.config.unified_budget_bytes
        if d4_budget <= 0:
            # Auto: reserve 10 GiB for vLLM KV cache, profiling, and overhead
            # The profiling forward pass temporarily allocates ~3-6 GiB for activations
            # KV cache needs ~0.5-2 GiB, plus miscellaneous overhead
            free_mem = torch.cuda.mem_get_info(device)[0]
            kv_overhead = 10 * 1024**3  # 10 GiB reserved
            d4_budget = max(0, int(free_mem - kv_overhead))
            print(f"[UNIFIED] D4 auto-budget: free_gpu={free_mem/1024**3:.2f} GiB, "
                  f"reserve={kv_overhead/1024**3:.0f} GiB, "
                  f"d4_budget={d4_budget/1024**3:.2f} GiB",
                  file=sys.stderr, flush=True)

        res_cfg = ResidencyConfig(
            gpu_cache_budget_bytes=d4_budget,
            num_layers=num_layers,
            num_experts=num_experts,
            split_ratio=self.config.unified_split_ratio,
            intermediate_size=inter_size,
            hidden_size=hidden_size,
            dtype=dtype,
            tail_slots_per_layer=self.config.unified_tail_slots,
            head_eviction="rwawe",
            rwawe_alpha=self.config.rwawe_alpha,
            rwawe_lambda=self.config.rwawe_lambda,
            rwawe_beta=self.config.rwawe_beta,
            enable_reservation=False,  # Executor manages allocation directly
        )
        residency = ResidencyManager(config=res_cfg, device=str(device))
        self._unified_residency = residency
        print(f"[UNIFIED] D4 ResidencyManager: head_slots={residency.head_slots_per_layer}, "
              f"tail_slots={residency.tail_slots_per_layer}",
              file=sys.stderr, flush=True)

        # D3+D4: Split weights & register layers
        for name, module in offloaded_layers:
            # Ensure weights are on CPU for splitting (UVA views report as CUDA)
            w13 = module.w13_weight.data.cpu()  # (E, 2*inter, hidden)
            w2 = module.w2_weight.data.cpu()    # (E, hidden, inter)
            head_w13, head_w2, tail_w13, tail_w2 = split_layer_experts(
                w13, w2, split_cfg
            )
            # Pin CPU tensors for async H2D
            if not head_w13.is_pinned():
                head_w13 = head_w13.pin_memory()
                head_w2 = head_w2.pin_memory()
                tail_w13 = tail_w13.pin_memory()
                tail_w2 = tail_w2.pin_memory()
            # Store CPU pinned references
            self._head_w13_refs[name] = head_w13
            self._head_w2_refs[name] = head_w2
            self._tail_w13_refs[name] = tail_w13
            self._tail_w2_refs[name] = tail_w2
            # Register with ResidencyManager (creates GPU pools)
            residency.register_layer(
                name, head_w13, head_w2, tail_w13, tail_w2
            )

        print(f"[UNIFIED] D3 weight splitting done ({num_layers} layers × {num_experts} experts)",
              file=sys.stderr, flush=True)

        # D1: Governor
        gov_cfg = GovernorConfig(
            mode=self.config.unified_governor_mode,
            K_min=self.config.unified_K_min,
            K_max=self.config.unified_K_max,
            K_default=self.config.unified_K_default,
            top_k=8,  # Qwen3-30B top_k
        )
        self._unified_governor = RuleBasedGovernor(config=gov_cfg)
        print(f"[UNIFIED] D1 Governor: mode={gov_cfg.mode}, K={gov_cfg.K_default} "
              f"[{gov_cfg.K_min}..{gov_cfg.K_max}]",
              file=sys.stderr, flush=True)

        # D2: VerifyScheduler
        sched_cfg = VerifySchedulerConfig(
            default_pattern=self.config.unified_verify_pattern,
        )
        self._unified_scheduler = VerifyScheduler(config=sched_cfg)
        print(f"[UNIFIED] D2 VerifyScheduler: pattern={sched_cfg.default_pattern}",
              file=sys.stderr, flush=True)

        # D5: HeadTailExecutor
        stream = self._prefetch_stream
        self._unified_executor = HeadTailExecutor(
            prefetch_stream=stream,
            max_M=64,
            top_k=8,
            head_inter=split_cfg.head_inter,
            tail_inter=split_cfg.tail_inter,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        print(f"[UNIFIED] D5 HeadTailExecutor: head_inter={split_cfg.head_inter}, "
              f"tail_inter={split_cfg.tail_inter}",
              file=sys.stderr, flush=True)

        # D6: PrefetchPlanner (created fresh per-step, not stored)
        # We store a factory lambda instead
        self._unified_planner = None  # created per-step

        print(f"[UNIFIED] D1-D6 unified scheduling framework initialized",
              file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Direct Triton Dispatch setup
    # -----------------------------------------------------------------------

    def _setup_direct_dispatch(self, offloaded_layers, dtype, device):
        """
        Pre-allocate intermediate buffers and cache kernel functions
        for bypassing quant_method.apply() on Pool-Direct offloaded layers.

        Benefits:
          - Eliminates ~8 levels of Python function call overhead per layer
          - Pre-allocated intermediates avoid per-call buffer creation
          - Passes pool_num_slots (17) instead of 128 to moe_align_block_size,
            reducing internal sort work
          - Opens the door for future INT8 pool quantization
        """
        import sys
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                invoke_fused_moe_triton_kernel,
                try_get_optimal_moe_config,
            )
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
                moe_align_block_size,
            )
        except ImportError as e:
            print(f"[ELMM] Direct dispatch unavailable: {e}",
                  file=sys.stderr, flush=True)
            self.config.enable_direct_dispatch = False
            return

        self._dd_invoke_kernel = invoke_fused_moe_triton_kernel
        self._dd_align_fn = moe_align_block_size

        # Get model topology from first offloaded layer
        first_module = offloaded_layers[0][1]
        first_meta = list(self._layer_meta.values())[0]

        self._dd_top_k = getattr(first_module, 'top_k', 8)
        self._dd_activation_name = getattr(first_module, 'activation', 'silu')

        w13_shape = first_meta["w13_shape"]  # (E, 2N, K)
        w2_shape = first_meta["w2_shape"]    # (E, K_out, N_in)

        N_w1 = w13_shape[1]      # 2*intermediate = 1536
        K_hidden = w13_shape[2]  # hidden_dim = 2048
        # Activation halves the gate+up dim
        act_dim = N_w1 // 2      # intermediate = 768

        max_M = self._dd_max_M   # 64
        top_k = self._dd_top_k
        EM = max_M * top_k       # 512

        # Pre-allocate shared intermediate buffers
        self._dd_inter_w1 = torch.empty(
            [max_M, top_k, N_w1], dtype=dtype, device=device
        )
        self._dd_inter_act = torch.empty(
            [EM, act_dim], dtype=dtype, device=device
        )
        self._dd_inter_w2 = torch.empty(
            [max_M, top_k, K_hidden], dtype=dtype, device=device
        )

        dd_bytes = sum(
            t.nelement() * t.element_size()
            for t in [self._dd_inter_w1, self._dd_inter_act, self._dd_inter_w2]
        )

        # Cache tile config (shape-based, num_experts dim is ignored).
        # Use M=4 (typical decode batch: K+1 tokens with K=3) instead of
        # max_M=64 so the config is tuned for the decode hot path.
        max_slots = max(c._max_slots for c in self._layer_caches.values())
        w1_size = torch.Size([max_slots, N_w1, K_hidden])
        w2_size = torch.Size([max_slots, K_hidden, act_dim])
        typical_decode_M = 4
        try:
            self._dd_tile_config = try_get_optimal_moe_config(
                w1_size, w2_size, top_k,
                "bf16" if dtype == torch.bfloat16 else "fp16",
                typical_decode_M,
                block_shape=None,
            )
        except Exception:
            self._dd_tile_config = None

        # A6000-tuned tile config (microbenchmark-validated).
        # Overrides vLLM default (BSM=16/BSN=32/BSK=64) which lacks
        # num_warps/num_stages and uses smaller tiles.
        # W8 (8 warps) + S3 (3 pipeline stages) improves HBM utilization
        # from 59% to ~73% on the decode hot path (M=4, E=17, bf16).
        if self._dd_tile_config is None or "num_warps" not in self._dd_tile_config:
            self._dd_tile_config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
                "num_stages": 3,
            }

        # Import optimized activation
        try:
            from vllm._custom_ops import silu_and_mul
            self._dd_silu_and_mul = silu_and_mul
        except ImportError:
            self._dd_silu_and_mul = None

        print(
            f"[ELMM] Direct dispatch enabled: "
            f"intermediates={dd_bytes / 1024**2:.1f} MB, "
            f"top_k={top_k}, activation={self._dd_activation_name}, "
            f"pool_slots={max_slots}, "
            f"tile_config=M{self._dd_tile_config.get('BLOCK_SIZE_M')}"
            f"/N{self._dd_tile_config.get('BLOCK_SIZE_N')}"
            f"/K{self._dd_tile_config.get('BLOCK_SIZE_K')}",
            file=sys.stderr, flush=True,
        )

    # -----------------------------------------------------------------------
    # CUDA Graph capture for TASER stale path (ALPS Phase 1)
    # -----------------------------------------------------------------------

    def _try_capture_layer_graph(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        remapped_ids: torch.Tensor,
    ):
        """Attempt to capture the Direct Dispatch kernel sequence as a CUDA Graph.

        Called once per layer, on the first TASER stale-path invocation after
        warmup.  If capture fails (e.g. Triton/CUDA Graph incompatibility),
        CUDA Graph is disabled globally and we fall back to eager dispatch.
        """
        import sys

        M = hidden_states.shape[0]
        top_k = remapped_ids.shape[1]
        K = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        cache = self._layer_caches[layer_name]
        module = self._patched_modules[layer_name]

        graph_obj = _LayerCUDAGraph(M, top_k, K, device, dtype)
        # Seed placeholders with actual data for realistic warm-up
        graph_obj.ph_hidden.copy_(hidden_states)
        graph_obj.ph_topk_weights.copy_(topk_weights)
        graph_obj.ph_remapped_ids.copy_(remapped_ids)

        # Closure: captures the Direct Dispatch kernel sequence
        def forward_fn(h, w, ids):
            return self._direct_dispatch_kernel(h, w, ids, cache, module)

        try:
            graph_obj.capture(forward_fn)
        except Exception as e:
            print(f"[ELMM] CUDA Graph capture failed for {layer_name}: {e}",
                  file=sys.stderr, flush=True)
            self._use_cuda_graph = False
            return

        self._layer_graphs[layer_name] = graph_obj
        if self._graph_M == 0:
            self._graph_M = M

        num_captured = len(self._layer_graphs)
        num_total = len(self._layer_caches)
        if num_captured == 1:
            print(f"[ELMM] CUDA Graph captured for first layer '{layer_name}' "
                  f"(M={M}, top_k={top_k}, K={K})",
                  file=sys.stderr, flush=True)
        if num_captured == num_total:
            print(f"[ELMM] CUDA Graph captured for all {num_total} layers",
                  file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Oracle Cross-Layer Prefetch (ALPS Phase 2)
    # -----------------------------------------------------------------------

    def _oracle_prefetch_next_layer(self, layer_name: str,
                                    topk_ids: Optional[torch.Tensor] = None):
        """Prefetch experts for the next offloaded layer on _prefetch_stream.

        Called at the start of Phase 4 (MoE kernel) so that PCIe H2D
        transfers overlap with the HBM-bound MoE compute.

        Prediction source (in priority order):
          1. Stacked Gating: use Layer N's actual topk_ids + co-activation table
          2. Fallback: current layer's cached expert set (Jaccard ~33% overlap)

        During stale path (99%+ of decode calls) the next-layer cache
        is already fully populated → no-op.  Benefit comes during:
          - Warmup (first ~32 calls per layer): cache not yet complete
          - Prefill (cold-start): cascade loading across layers
          - Post-validation: next layer may have changed routing
        """
        idx = self._layer_index.get(layer_name)
        if idx is None or idx + 1 >= len(self._ordered_layers):
            return  # last layer or unknown
        next_name = self._ordered_layers[idx + 1]
        next_cache = self._layer_caches[next_name]
        next_module = self._patched_modules.get(next_name)
        if next_module is None:
            return

        # Check if next layer's cache is already fully occupied
        if len(next_cache._slot_map) >= next_cache._max_slots:
            self._oracle_prefetch_skipped += 1
            return

        # Determine which experts next_layer will need.
        # If TASER remap is active for next_layer, the remap table is
        # authoritative — any expert with remap[e] == 0 AND e != 0
        # is NOT cached.  Expert 0 always maps to slot 0 (default).
        next_remap = self._layer_remap.get(next_name)
        if next_remap is None:
            return

        cached_eids = set(next_cache._slot_map.keys())
        if len(cached_eids) >= next_cache._max_slots:
            self._oracle_prefetch_skipped += 1
            return

        # Prediction: Stacked Gating (P1) or fallback to cache overlap
        if (self.config.enable_stacked_gating
                and self._coact_built
                and topk_ids is not None):
            predicted_eids = self._predict_next_layer_experts(
                layer_name, topk_ids)
            # Track prediction accuracy (after warmup, when we have ground truth)
            if predicted_eids:
                self._sg_predict_total += len(predicted_eids)
        else:
            # Fallback: use current layer's cached experts as prediction
            cur_cache = self._layer_caches[layer_name]
            predicted_eids = set(cur_cache._slot_map.keys())

        # Only prefetch experts NOT already in next_cache
        to_prefetch = predicted_eids - cached_eids
        if not to_prefetch:
            self._oracle_prefetch_skipped += 1
            return

        # Limit to available free slots to avoid unnecessary eviction
        free_slots = len(next_cache._free_slots)
        if free_slots == 0:
            self._oracle_prefetch_skipped += 1
            return

        stream = self._prefetch_stream
        if stream is None:
            return

        w13_ref = next_module.w13_weight
        w2_ref = next_module.w2_weight

        # Launch async H2D on prefetch stream (PCIe, not HBM)
        stream.wait_stream(torch.cuda.current_stream())
        count = 0
        next_remap_tbl = self._layer_remap.get(next_name)
        with torch.cuda.stream(stream):
            for eid in to_prefetch:
                if count >= free_slots:
                    break
                if next_cache.contains(eid):
                    continue
                # If cache is full, eviction will happen — zero the
                # evicted expert's remap so TASER's has_unmapped catches it.
                if (not next_cache._free_slots
                        and eid not in next_cache._slot_map):
                    evict_eid = next(iter(next_cache._slot_map))
                    if next_remap_tbl is not None:
                        next_remap_tbl[evict_eid] = -1
                slot = next_cache.alloc_slot(eid)
                pool_w13, pool_w2 = next_cache.get_slot_tensors(slot)
                pool_w13.copy_(w13_ref[eid], non_blocking=True)
                pool_w2.copy_(w2_ref[eid], non_blocking=True)
                if next_cache.has_aux_pools:
                    s13, s2 = next_cache.get_slot_scale_tensors(slot)
                    b13, b2 = next_cache.get_slot_bias_tensors(slot)
                    if s13 is not None:
                        s13.copy_(next_module.w13_weight_scale[eid], non_blocking=True)
                        s2.copy_(next_module.w2_weight_scale[eid], non_blocking=True)
                    if b13 is not None:
                        b13.copy_(next_module.w13_bias[eid], non_blocking=True)
                        b2.copy_(next_module.w2_bias[eid], non_blocking=True)
                # Update the per-layer remap table for the prefetched expert
                if next_remap_tbl is not None:
                    next_remap_tbl[eid] = slot
                count += 1
        if count > 0:
            self._prefetch_in_flight = True
            # Mark NEXT layer dirty (its cache was modified)
            self._layer_cache_dirty[next_name] = True
        self._oracle_prefetch_issued += count

    # -----------------------------------------------------------------------
    # Stacked Gating: co-activation table finalization + prediction
    # -----------------------------------------------------------------------

    def _finalize_coact_tables(self):
        """Convert raw co-activation counts into top-K prediction tables."""
        sg_k = self.config.stacked_gating_top_k
        for layer_idx, tbl in self._coact_tables.items():
            # For each source expert, find top-K most co-activated next-layer experts
            top_k_dict: dict[int, list[int]] = {}
            for src_e in range(tbl.shape[0]):
                row = tbl[src_e]
                if row.sum().item() == 0:
                    continue
                # Get top-K indices (most frequently co-activated)
                k = min(sg_k, (row > 0).sum().item())
                if k > 0:
                    _, top_indices = row.topk(k)
                    top_k_dict[src_e] = top_indices.tolist()
            self._coact_top_k[layer_idx] = top_k_dict
        self._coact_built = True
        # Print diagnostic
        total_entries = sum(len(d) for d in self._coact_top_k.values())
        import sys
        print(f"[ELMM] Stacked Gating: co-activation tables built "
              f"({len(self._coact_top_k)} layer pairs, "
              f"{total_entries} source experts, "
              f"top-{sg_k} predictions, "
              f"after {self._coact_warmup_steps} warmup steps)",
              file=sys.stderr, flush=True)

    def _predict_next_layer_experts(
        self, layer_name: str, topk_ids: torch.Tensor,
    ) -> set[int]:
        """Predict Layer N+1's experts from Layer N's actual routing."""
        idx = self._layer_index.get(layer_name)
        if idx is None or idx not in self._coact_top_k:
            return set()
        pred_table = self._coact_top_k[idx]
        if not pred_table:
            return set()
        predictions: set[int] = set()
        routed = topk_ids.reshape(-1).unique().tolist()
        for e in routed:
            preds = pred_table.get(e)
            if preds:
                predictions.update(preds)
        return predictions

    # -----------------------------------------------------------------------
    # GPU-Side Cache Lookup (E1 optimization)
    # -----------------------------------------------------------------------

    def _gpu_cache_phase3(
        self,
        layer_name: str,
        topk_ids: torch.Tensor,
        cache: '_LayerExpertCache',
        meta: dict,
        module: nn.Module,
    ) -> tuple[torch.Tensor, int, int] | None:
        """
        GPU-side Phase 3: cache lookup + miss handling + remap.

        Returns (remapped_ids, step_hits, step_misses), or None if
        fallback to CPU path is needed (e.g., too many misses for pool).

        The common case (0 misses, >99% of calls) runs entirely on GPU
        without any GPU→CPU synchronization. Only when misses occur do
        we sync to perform CPU-side LRU eviction.
        """
        self._gpu_cache_step += 1
        gpu_e2s = self._gpu_eid_to_slot[layer_name]
        gpu_lru = self._gpu_lru_clock[layer_name]
        remap = self._remap_table

        # 1. Unique experts — stays on GPU
        unique_eids = topk_ids.reshape(-1).unique()
        num_unique = unique_eids.shape[0]

        # 2. Lookup slots on GPU (no sync)
        slots = gpu_e2s[unique_eids]  # [num_unique], -1 = miss

        # 3. Count misses — single scalar sync
        miss_mask = (slots == -1)
        num_misses = miss_mask.sum().item()  # one sync point (scalar)

        step_hits = num_unique - num_misses
        step_misses = num_misses

        # 4. Handle misses (rare path: <1% of calls after warmup)
        if num_misses > 0:
            # If more misses than pool can hold, fall back to CPU path
            # (this only happens during cold start)
            if num_misses > cache._max_slots:
                return None

            miss_eids_gpu = unique_eids[miss_mask]
            miss_eids_list = miss_eids_gpu.tolist()

            w13_ref = module.w13_weight
            w2_ref = module.w2_weight

            new_slots = []
            evicted_eids = []
            for eid in miss_eids_list:
                # Track evictions to clear GPU mapping
                if not cache._free_slots and cache._slot_map:
                    evict_eid = next(iter(cache._slot_map))
                    evicted_eids.append(evict_eid)
                new_slot = cache.alloc_slot(eid)
                pool_w13, pool_w2 = cache.get_slot_tensors(new_slot)
                pool_w13.copy_(w13_ref[eid], non_blocking=True)
                pool_w2.copy_(w2_ref[eid], non_blocking=True)
                if cache.has_aux_pools:
                    s13, s2 = cache.get_slot_scale_tensors(new_slot)
                    b13, b2 = cache.get_slot_bias_tensors(new_slot)
                    if s13 is not None:
                        s13.copy_(module.w13_weight_scale[eid], non_blocking=True)
                        s2.copy_(module.w2_weight_scale[eid], non_blocking=True)
                    if b13 is not None:
                        b13.copy_(module.w13_bias[eid], non_blocking=True)
                        b2.copy_(module.w2_bias[eid], non_blocking=True)
                new_slots.append(new_slot)

            # Clear GPU mapping for evicted experts
            if evicted_eids:
                evict_t = torch.tensor(
                    evicted_eids, dtype=torch.long, device=gpu_e2s.device
                )
                gpu_e2s.scatter_(
                    0, evict_t,
                    torch.full_like(evict_t, -1),
                )

            # Set GPU mapping for newly loaded experts
            miss_slots_t = torch.tensor(
                new_slots, dtype=torch.long, device=gpu_e2s.device
            )
            gpu_e2s.scatter_(0, miss_eids_gpu, miss_slots_t)

            # Re-read all slots (some may have changed due to eviction)
            slots = gpu_e2s[unique_eids]

            # Safety: if any slots still -1 (eviction collision), fall back
            if (slots == -1).any().item():
                return None

        # 5. Update LRU timestamps on GPU (all unique experts touched)
        touched_slots = slots
        if touched_slots.numel() > 0:
            gpu_lru.scatter_(
                0, touched_slots,
                torch.full_like(touched_slots, self._gpu_cache_step),
            )

        # 6. Build remap table on GPU
        remap.scatter_(0, unique_eids, slots)
        remapped_ids = remap[topk_ids]

        # 7. Stats
        pcie_bytes = step_misses * meta["expert_size"]
        self._total_cache_hits += step_hits
        self._total_cache_misses += step_misses
        self._total_pcie_bytes += pcie_bytes

        return remapped_ids, step_hits, step_misses

    # -----------------------------------------------------------------------
    # TASER v2 helpers
    # -----------------------------------------------------------------------

    def _taser_v2_build_hot_set(self, layer_name: str):
        """Build hot_set from frequency stats (top hot_slots experts)."""
        freq = self._expert_freq[layer_name]
        cache = self._layer_caches[layer_name]
        cold = min(self.config.taser_cold_slots, cache._max_slots // 4)
        hot_slots = cache._max_slots - cold
        sorted_experts = sorted(freq.keys(), key=lambda e: freq[e], reverse=True)
        self._hot_set[layer_name] = set(sorted_experts[:hot_slots])

    def _taser_v2_freeze_remap(self, layer_name: str):
        """Freeze remap table with ALL cached experts.
        
        Maps every cached expert to its slot.  Unmapped experts get -1
        sentinel so CRUISE fast path can detect misses and route them
        to the HFDE dual-rail path for exact computation.
        """
        lr = self._layer_remap[layer_name]
        cache = self._layer_caches[layer_name]
        lr.fill_(-1)  # sentinel: unmapped experts are detectable
        for eid, slot in cache._slot_map.items():
            lr[eid] = slot

    def _taser_v2_compute_convergence(self, layer_name: str) -> float:
        """Fraction of hot_set experts present in cache."""
        hot_set = self._hot_set[layer_name]
        if not hot_set:
            return 0.0
        cache = self._layer_caches[layer_name]
        cached = set(cache._slot_map.keys())
        return len(cached & hot_set) / len(hot_set)

    def _taser_v2_collect_freq(self, layer_name: str, topk_ids: torch.Tensor):
        """Accumulate expert frequency from topk_ids."""
        freq = self._expert_freq[layer_name]
        for eid in topk_ids.reshape(-1).tolist():
            freq[eid] = freq.get(eid, 0) + 1

    # -----------------------------------------------------------------------
    # Core: ELMM forward_impl (scratchpad + swap)
    # -----------------------------------------------------------------------

    def _elmm_forward_impl(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Replacement for FusedMoE.forward_impl on offloaded layers.

        Protocol:
          1. Run the original routing + shared-expert logic.
          2. After routing gives topk_ids, determine unique experts.
          3. For each unique expert:
             - Cache HIT → copy from per-layer GPU cache to scratchpad (HBM→HBM).
             - Cache MISS → H2D from CPU pinned to scratchpad + insert into cache.
          4. Swap module.w13_weight.data / w2_weight.data → scratchpad.
          5. Run kernel via quant_method.apply() (reads fast GPU scratchpad).
          6. Restore original UVA param.data.
          7. Handle shared experts output.
        """
        module = self._patched_modules[layer_name]
        cache = self._layer_caches[layer_name]
        meta = self._layer_meta[layer_name]
        self._total_intercepts += 1

        # --- Phase 1: Pre-routing setup (mirrors original forward_impl) ---
        p_tok = self._maybe_profile_start("P1_setup")
        module.ensure_moe_quant_config_init()
        module.ensure_dp_chunking_init()

        has_separate_shared = (
            not module.quant_method.mk_owns_shared_expert
            and module.shared_experts is not None
        )

        use_shared_stream, hidden_clone = (
            module._maybe_setup_shared_experts_stream(
                hidden_states, has_separate_shared, False
            )
        )

        if module.gate is not None:
            router_logits, _ = module.gate(hidden_states)

        # Start shared experts early if not using stream
        # (skip if ALPS Phase 3 parallelization is active — shared expert
        # will run concurrently with MoE kernel in Phase 4)
        shared_output = None
        run_shared_parallel = (
            self.config.enable_shared_parallel
            and has_separate_shared
            and not use_shared_stream
            and self._shared_expert_stream is not None
        )
        if has_separate_shared and not use_shared_stream and not run_shared_parallel:
            shared_input = module._get_shared_experts_input(hidden_states)
            shared_output = module.shared_experts(shared_input)

        # --- Phase 2: Routing ---
        self._maybe_profile_end(p_tok)
        p_tok = self._maybe_profile_start("P2_routing")
        topk_weights, topk_ids = module.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # --- Unified Scheduling (D1-D6): delegate to unified pipeline ---
        if self.config.enable_unified_scheduling and self._unified_residency is not None:
            self._maybe_profile_end(p_tok)
            final_hidden = self._unified_forward_layer(
                layer_name, hidden_states, topk_weights, topk_ids, module
            )
            # Handle shared experts
            if has_separate_shared:
                if run_shared_parallel:
                    torch.cuda.current_stream().wait_stream(self._shared_expert_stream)
                elif use_shared_stream:
                    from vllm.utils.torch_utils import current_stream
                    with torch.cuda.stream(module.shared_experts_stream):
                        shared_output = module.shared_experts(hidden_clone)
                    current_stream().wait_stream(module.shared_experts_stream)
                final_hidden = (shared_output, final_hidden)
            self._total_intercepts += 0  # already incremented above
            return final_hidden

        # --- ACES: update EMA from router logits ---
        if self._aces is not None:
            _lid = self._layer_name_to_id.get(layer_name, -1)
            if _lid >= 0:
                self._aces.update(_lid, router_logits)

        # --- Stacked Gating: record co-activations during warmup ---
        if (self.config.enable_stacked_gating
                and not self._coact_built
                and self._ordered_layers):
            cur_idx = self._layer_index.get(layer_name, -1)
            if cur_idx >= 0:
                cur_experts = topk_ids.reshape(-1).unique().tolist()
                # Record co-activations between previous layer and current
                if (self._sg_prev_layer_idx >= 0
                        and self._sg_prev_layer_idx == cur_idx - 1
                        and self._sg_prev_layer_experts is not None):
                    prev_idx = self._sg_prev_layer_idx
                    tbl = self._coact_tables.get(prev_idx)
                    if tbl is not None:
                        for ep in self._sg_prev_layer_experts:
                            for ec in cur_experts:
                                tbl[ep, ec] += 1
                self._sg_prev_layer_experts = cur_experts
                self._sg_prev_layer_idx = cur_idx
                # Check if warmup is complete (last layer of a forward pass)
                if cur_idx == len(self._ordered_layers) - 1:
                    self._coact_warmup_steps += 1
                    # Reset for next forward pass
                    self._sg_prev_layer_experts = None
                    self._sg_prev_layer_idx = -1
                    if self._coact_warmup_steps >= self.config.stale_remap_warmup:
                        self._finalize_coact_tables()

        # --- Phase 3: Cache lookup + scratchpad fill ---
        self._maybe_profile_end(p_tok)
        p_tok = self._maybe_profile_start("P3_cache")

        # Sync any in-flight prefetch copies (draft or oracle) before
        # accessing cache.  Without this wait, Phase 3 could see a slot
        # as "cached" while data is still copying on _prefetch_stream.
        if self._prefetch_in_flight and self._prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            self._prefetch_in_flight = False

        use_pool_direct = (
            self.config.enable_pool_direct and self._remap_table is not None
        )

        # === TASER v2: Four-Phase Dual-Rail Hot Expert Routing ===
        # Phases: WARMUP → CONVERGE → CRUISE → DRIFT (→ CONVERGE → ...)
        # CRUISE uses dual-rail: hit experts → pool-direct, miss → HFDE.
        stale_remap_ok = False
        taser_v2_dual_rail = False  # True when cruise dual-rail path is active
        _taser_v2_miss_data = None  # (remapped_ids, is_miss_flat, miss_eid_list)
        _taser_debug_layer = ("layers.1." in layer_name)
        hfde_active = False  # set True inside CPU cache path when eligible
        sri = self.config.stale_remap_interval
        if sri > 0 and use_pool_direct and layer_name in self._layer_remap:
            step = self._layer_remap_step[layer_name]
            phase = self._taser_phase.get(layer_name, 'warmup')
            lr = self._layer_remap[layer_name]

            # ── Phase A: WARMUP (collect freq, always slow path) ──
            if phase == 'warmup':
                self._taser_v2_collect_freq(layer_name, topk_ids)
                if step >= self.config.stale_remap_warmup:
                    self._taser_v2_build_hot_set(layer_name)
                    self._taser_phase[layer_name] = 'converge'
                    self._converge_start_step[layer_name] = step
                    phase = 'converge'
                # else: fall through to slow path

            # ── Phase D: DRIFT (short re-warmup, always slow path) ──
            elif phase == 'drift':
                self._taser_v2_collect_freq(layer_name, topk_ids)
                drift_steps = step - self._drift_start_step[layer_name]
                if drift_steps >= self.config.taser_drift_warmup:
                    self._taser_v2_build_hot_set(layer_name)
                    self._taser_phase[layer_name] = 'converge'
                    self._converge_start_step[layer_name] = step
                    phase = 'converge'
                # else: fall through to slow path

            # ── Phase B: CONVERGE (slow path, check convergence) ──
            if phase == 'converge':
                # Convergence is checked after Phase 3 remap rebuild below
                pass  # fall through to slow path

            # ── Phase C: CRUISE (fast path, skip Phase 3) ──
            elif phase == 'cruise':
                # Check dirty flag: prefetch/oracle/rebalance may have
                # modified the cache since freeze. If dirty, rebuild remap
                # from current cache state (keeps CRUISE, doesn't drift).
                if self._layer_cache_dirty.get(layer_name, False):
                    lr.fill_(-1)
                    cache = self._layer_caches[layer_name]
                    for _eid, _slot in cache._slot_map.items():
                        lr[_eid] = _slot
                    self._layer_cache_dirty[layer_name] = False

                # Periodic validation: every N steps, fall to slow path
                # to refresh remap. Use v1 adaptive interval.
                cruise_interval = self._layer_adaptive_interval.get(layer_name, sri)
                cruise_next = self._layer_next_validation.get(layer_name, 0)
                if step >= cruise_next:
                    # Validation step: fall through to slow path
                    # Drift detection happens after Phase 3 rebuild
                    pass
                else:
                    # CRUISE FAST PATH: use frozen remap, detect misses
                    remapped_ids = lr[topk_ids]
                    is_miss = (remapped_ids < 0)  # [M, top_k] bool
                    n_miss = is_miss.any().item()  # quick check: any miss?

                    if not n_miss:
                        # Pure fast path: ALL experts in remap
                        step_hits = 0
                        step_misses = 0
                        stale_remap_ok = True
                        self._taser_v2_cruise_count += 1
                    else:
                        # Has misses — count and identify
                        miss_expert_ids = topk_ids[is_miss].unique()
                        n_unique_miss = miss_expert_ids.numel()
                        miss_budget = self._miss_budget.get(layer_name, 3)

                        if (n_unique_miss <= miss_budget
                                and self._prefetch_stream is not None):
                            # DUAL-RAIL: hit → single-pass kernel,
                            # miss → HFDE async load + second pass.
                            miss_list = miss_expert_ids.tolist()
                            taser_v2_dual_rail = True
                            _taser_v2_miss_data = (remapped_ids, is_miss, miss_list)
                            self._taser_v2_dual_rail_count += 1
                            self._layer_cache_dirty[layer_name] = True

                            # Drift detection
                            n_total = topk_ids.reshape(-1).unique().numel()
                            miss_ratio = n_unique_miss / max(n_total, 1)
                            step_hits = n_total - n_unique_miss
                            step_misses = n_unique_miss
                            alpha = self.config.taser_drift_ema_alpha
                            ema = self._miss_ratio_ema.get(layer_name, 0.0)
                            self._miss_ratio_ema[layer_name] = (
                                (1 - alpha) * ema + alpha * miss_ratio
                            )
                            if self._miss_ratio_ema[layer_name] > self.config.taser_drift_miss_threshold:
                                self._taser_phase[layer_name] = 'drift'
                                self._drift_start_step[layer_name] = step
                                self._expert_freq[layer_name] = {}
                                self._taser_v2_drift_count += 1
                                # Cancel dual-rail, fall to slow path
                                taser_v2_dual_rail = False
                                _taser_v2_miss_data = None
                        else:
                            # Too many misses or no stream: fall to slow path
                            pass  # stale_remap_ok stays False

            self._layer_remap_step[layer_name] = step + 1

            # --- TASER diagnostic ---
            if _taser_debug_layer and step > 0 and step % 500 == 0:
                import sys
                _fp = getattr(self, '_taser_fast_count', 0)
                _sp = getattr(self, '_taser_slow_count', 0)
                _cr = self._taser_v2_cruise_count
                _dr = self._taser_v2_dual_rail_count
                _df = self._taser_v2_drift_count
                print(f"[TASER-v2] {layer_name} step={step}: "
                      f"phase={self._taser_phase.get(layer_name,'?')}, "
                      f"fast={_fp}, slow={_sp}, cruise={_cr}, "
                      f"dual_rail={_dr}, drift={_df}",
                      file=sys.stderr, flush=True)
            if stale_remap_ok:
                self._taser_fast_count = getattr(self, '_taser_fast_count', 0) + 1

        if not stale_remap_ok and not taser_v2_dual_rail:
            self._taser_slow_count = getattr(self, '_taser_slow_count', 0) + 1
            # Need full Phase 3 (either stale-remap disabled, warmup, or validation step)
            use_gpu_cache = (
                self.config.enable_gpu_cache
                and use_pool_direct
                and layer_name in self._gpu_eid_to_slot
            )

            if use_gpu_cache:
                # === GPU-Side Cache Path (E1 optimization) ===
                # Common case (0 misses) runs without GPU→CPU sync.
                gpu_result = self._gpu_cache_phase3(
                    layer_name, topk_ids, cache, meta, module,
                )
                if gpu_result is not None:
                    remapped_ids, step_hits, step_misses = gpu_result
                else:
                    # Fallback: too many misses for pool, use CPU path
                    use_gpu_cache = False

            if not use_gpu_cache:
                # === Original CPU-Side Cache Path ===
                p3a_tok = self._maybe_profile_start("P3a_sync")
                unique_experts = topk_ids.reshape(-1).unique()
                unique_list = unique_experts.tolist()
                unique_set = set(unique_list)

                # Track temporal locality
                prev_set = self._last_expert_set.get(layer_name)
                if prev_set is not None and len(prev_set) > 0:
                    overlap = len(prev_set & unique_set) / len(prev_set | unique_set)
                    hist = self._overlap_history.get(layer_name)
                    if hist is None:
                        hist = []
                        self._overlap_history[layer_name] = hist
                    hist.append(overlap)
                    if not self._overlap_unlimited and len(hist) > 100:
                        hist.pop(0)
                self._last_expert_set[layer_name] = unique_set

                # Collect for structured locality analyzer
                if self._locality_analyzer is not None:
                    self._current_round_experts[layer_name] = unique_set

                # Track prefetch accuracy
                pending = self._pending_prefetch.get(layer_name)
                if pending:
                    prefetch_hits = len(pending & unique_set)
                    self._prefetch_hits += prefetch_hits
                    self._prefetch_total += len(pending)
                    self._pending_prefetch[layer_name] = set()

                # Weight references (UVA views for offloaded layers)
                w13_ref = module.w13_weight
                w2_ref = module.w2_weight

                # Classify experts into cache hits and misses
                if cache._rwawe_enabled:
                    cache.advance_step()
                # Draft-Utility (P3): compute per-expert utility from token coverage
                M = hidden_states.shape[0]
                if (self.config.enable_draft_utility
                        and cache._rwawe_enabled and M > 1):
                    _flat_ids = topk_ids.view(-1).tolist()
                    _ucounts: dict[int, int] = {}
                    for _uid in _flat_ids:
                        _ucounts[_uid] = _ucounts.get(_uid, 0) + 1
                    _utility_map = {e: min(c / M, 1.0)
                                    for e, c in _ucounts.items()}
                else:
                    _utility_map = None
                hit_eids: list[int] = []
                hit_slots: list[int] = []
                miss_eids: list[int] = []

                for eid in unique_list:
                    _u = _utility_map.get(eid, 1.0) if _utility_map else 1.0
                    slot = cache.get(eid, utility=_u)
                    if slot is not None:
                        hit_eids.append(eid)
                        hit_slots.append(slot)
                    else:
                        miss_eids.append(eid)

                step_hits = len(hit_eids)
                step_misses = len(miss_eids)
                pcie_bytes = step_misses * meta["expert_size"]
                self._maybe_profile_end(p3a_tok)

                # Pool overflow guard: when the number of unique experts
                # exceeds cache capacity (common during prefill with many
                # tokens), fall back to UVA weights directly.  Without
                # this, cache eviction during miss-loading overwrites hits,
                # making the remap table inconsistent with slot contents.
                if use_pool_direct and len(unique_list) > cache._max_slots:
                    use_pool_direct = False

                # Handle cache misses: copy UVA → pool slot
                p3b_tok = self._maybe_profile_start("P3b_load")
                miss_slots: list[int] = []
                use_sacr = self._briskmoe_sacr is not None
                use_elp = self._briskmoe_elp is not None
                use_briskmoe = use_sacr or use_elp
                # ACES eviction only when NOT in TASER-mapping mode
                use_aces = (self._aces is not None
                            and not self.config.aces_taser)

                # --- HFDE eligibility check ---
                # Overlap miss-loading (PCIe) with hit-expert compute (GPU)
                # by loading misses on _prefetch_stream while the hit
                # kernel runs on the default stream.
                M = hidden_states.shape[0]
                hfde_active = (
                    self.config.enable_hfde
                    and step_misses > 0
                    and step_hits > 0
                    and use_pool_direct
                    and self._prefetch_stream is not None
                    and self.config.enable_direct_dispatch
                    and self._dd_invoke_kernel is not None
                    and M <= self._dd_max_M
                )
                if hfde_active:
                    _hfde_stream = self._prefetch_stream
                    _hfde_stream.wait_stream(torch.cuda.current_stream())
                    _hfde_ctx = torch.cuda.stream(_hfde_stream)
                else:
                    _hfde_ctx = contextlib.nullcontext()
                use_pred_cache = self._pred_cache is not None
                layer_id = self._layer_name_to_id.get(layer_name, 0)

                # BriskMoE: record accesses for ALL touched experts (hits+misses)
                # Fused batch call — single loop over unique_list instead of two
                if use_sacr:
                    self._briskmoe_sacr.record_access_batch(
                        layer_id, unique_list, self._briskmoe_step
                    )
                    # Feed hit rate to SACR for adaptive γ
                    step_total = step_hits + step_misses
                    if step_total > 0:
                        self._briskmoe_sacr.update_hit_rate(
                            layer_id, step_hits / step_total
                        )
                if use_elp:
                    self._briskmoe_elp.access_batch(
                        layer_id, unique_list, self._briskmoe_step
                    )
                if use_pred_cache:
                    self._pred_cache.record_access_batch(layer_id, unique_list)
                    # Update demand with current routing (predicts next step
                    # via EMA — no extra .tolist() needed since we reuse unique_list)
                    self._pred_cache.update_predictions_from_flat(layer_id, unique_list)

                for eid in miss_eids:
                    if use_briskmoe and not cache._free_slots and eid not in cache._slot_map:
                        if use_pred_cache:
                            # PredCache: LRU with demand-aware protection.
                            # Iterate cache in LRU order, skip experts with
                            # predicted demand. Falls back to pure LRU if all
                            # candidates have demand (or no predictions yet).
                            pc = self._pred_cache
                            victim = None
                            for k in cache._slot_map:
                                if k in unique_set:
                                    continue  # need this step
                                if pc._get_demand(layer_id, k) <= 0.0:
                                    victim = k
                                    break
                            if victim is None:
                                # All candidates have demand → fall back to LRU
                                for k in cache._slot_map:
                                    if k not in unique_set:
                                        victim = k
                                        break
                            if victim is None:
                                victim = next(iter(cache._slot_map))
                        elif use_elp:
                            # BriskMoE victim selection (SACR/ELP path)
                            candidates = self._briskmoe_elp.get_flex_candidates(layer_id)
                            candidates = [c for c in candidates if c in cache._slot_map]
                            if not candidates:
                                candidates = list(cache._slot_map.keys())

                            if use_sacr:
                                victim = self._briskmoe_sacr.select_victim(layer_id, candidates)
                            else:
                                # ELP-only: pick LRU among candidates
                                cand_set = set(candidates)
                                victim = None
                                for k in cache._slot_map:
                                    if k in cand_set:
                                        victim = k
                                        break
                                if victim is None:
                                    victim = candidates[0]
                        else:
                            # SACR-only path
                            candidates = list(cache._slot_map.keys())
                            victim = self._briskmoe_sacr.select_victim(layer_id, candidates)

                        new_slot, evicted = cache.alloc_slot_with_victim(eid, victim)
                        if evicted is not None:
                            if use_sacr:
                                self._briskmoe_sacr.remove_expert(layer_id, evicted)
                            if use_elp:
                                self._briskmoe_elp.remove_expert(layer_id, evicted)
                    elif use_aces and not cache._free_slots and eid not in cache._slot_map:
                        # ACES: evict the cached expert with the lowest
                        # router-probability EMA score.
                        victim = self._aces.select_victim(
                            layer_id, list(cache._slot_map.keys()), unique_set
                        )
                        new_slot, evicted = cache.alloc_slot_with_victim(eid, victim)
                    else:
                        _mu = _utility_map.get(eid, 1.0) if _utility_map else 1.0
                        new_slot = cache.alloc_slot(eid, utility=_mu)
                    pool_w13, pool_w2 = cache.get_slot_tensors(new_slot)
                    # GPU copies: on _prefetch_stream if HFDE active,
                    # else default stream.  Eviction above is CPU-only
                    # (Python dict ops), unaffected by stream context.
                    with _hfde_ctx:
                        pool_w13.copy_(w13_ref[eid], non_blocking=True)
                        pool_w2.copy_(w2_ref[eid], non_blocking=True)
                        if cache.has_aux_pools:
                            s13, s2 = cache.get_slot_scale_tensors(new_slot)
                            b13, b2 = cache.get_slot_bias_tensors(new_slot)
                            if s13 is not None:
                                s13.copy_(module.w13_weight_scale[eid], non_blocking=True)
                                s2.copy_(module.w2_weight_scale[eid], non_blocking=True)
                            if b13 is not None:
                                b13.copy_(module.w13_bias[eid], non_blocking=True)
                                b2.copy_(module.w2_bias[eid], non_blocking=True)
                    miss_slots.append(new_slot)

                # BriskMoE: periodic ELP rebalance
                if (self._briskmoe_elp is not None
                        and self._briskmoe_step % self._briskmoe_elp.config.rebalance_interval == 0):
                    self._briskmoe_elp.rebalance(layer_id)

                self._maybe_profile_end(p3b_tok)

                p3c_tok = self._maybe_profile_start("P3c_copy")
                # Pool-direct overflow guard: when unique experts exceed
                # cache capacity, remap has unmapped entries pointing to
                # slot 0 (wrong weights).  Fall back to scratch path.
                if use_pool_direct and len(unique_list) > cache._max_slots:
                    use_pool_direct = False
                if use_pool_direct:
                    remap = self._remap_table
                    all_eids = hit_eids + miss_eids
                    all_slots = hit_slots + miss_slots
                    if all_eids:
                        eid_t = torch.tensor(
                            all_eids, dtype=torch.long, device=remap.device
                        )
                        slot_t = torch.tensor(
                            all_slots, dtype=torch.long, device=remap.device
                        )
                        remap.scatter_(0, eid_t, slot_t)
                    remapped_ids = remap[topk_ids]
                    # Persist to per-layer remap for stale-remap mode
                    if sri > 0 and layer_name in self._layer_remap:
                        lr = self._layer_remap[layer_name]

                        # --- ACES-TASER: proactive remap from EMA ---
                        # Instead of passively reflecting LRU cache content,
                        # use ACES EMA top-K to decide which experts SHOULD
                        # be cached.  Evict low-EMA experts and load high-EMA
                        # experts that are missing.
                        if (self.config.aces_taser
                                and self._aces is not None
                                and step >= self.config.stale_remap_warmup):
                            _lid = self._layer_name_to_id.get(layer_name, -1)
                            if _lid >= 0:
                                desired = self._aces.get_top_k(_lid, cache._max_slots)
                                desired_set = set(desired)
                                cached_set = set(cache._slot_map.keys())
                                # Experts to load: desired but not cached
                                to_load = [e for e in desired if e not in cached_set]
                                # Experts to evict: cached but not desired
                                # (and not needed this step)
                                to_evict = [e for e in cache._slot_map
                                            if e not in desired_set and e not in unique_set]
                                # Swap: evict one, load one
                                n_swaps = min(len(to_load), len(to_evict))
                                for i in range(n_swaps):
                                    victim_eid = to_evict[i]
                                    new_eid = to_load[i]
                                    new_slot, _ = cache.alloc_slot_with_victim(new_eid, victim_eid)
                                    pool_w13_s, pool_w2_s = cache.get_slot_tensors(new_slot)
                                    pool_w13_s.copy_(w13_ref[new_eid], non_blocking=True)
                                    pool_w2_s.copy_(w2_ref[new_eid], non_blocking=True)
                                    if cache.has_aux_pools:
                                        s13, s2 = cache.get_slot_scale_tensors(new_slot)
                                        b13, b2 = cache.get_slot_bias_tensors(new_slot)
                                        if s13 is not None:
                                            s13.copy_(module.w13_weight_scale[new_eid], non_blocking=True)
                                            s2.copy_(module.w2_weight_scale[new_eid], non_blocking=True)
                                        if b13 is not None:
                                            b13.copy_(module.w13_bias[new_eid], non_blocking=True)
                                            b2.copy_(module.w2_bias[new_eid], non_blocking=True)

                        # Rebuild remap from the full cache slot_map, not
                        # just the current step's experts.  This ensures
                        # that stale-path lookups for ANY cached expert
                        # return the correct slot (not the default -1
                        # sentinel which means uncached).
                        lr.fill_(-1)
                        for cached_eid, cached_slot in cache._slot_map.items():
                            lr[cached_eid] = cached_slot
                        # --- TASER diagnostic: verify remap after rebuild ---
                        if _taser_debug_layer and step <= 40:
                            import sys
                            _n_cached = len(cache._slot_map)
                            _n_pos = (lr >= 0).sum().item()
                            _ris = remapped_ids.reshape(-1)
                            _tis = topk_ids.reshape(-1)
                            print(f"[TASER-DBG] REBUILD step={step}: cached={_n_cached}, "
                                  f"remap_pos={_n_pos}, maxslots={cache._max_slots}, "
                                  f"topk_ids={_tis[:8].tolist()}, "
                                  f"remapped={_ris[:8].tolist()}, "
                                  f"hits={step_hits}, misses={step_misses}",
                                  file=sys.stderr, flush=True)
                        # Remap rebuilt from ground truth → cache is clean
                        self._layer_cache_dirty[layer_name] = False
                        # --- TASER adaptive interval update ---
                        # Compare with previous expert set: if unchanged,
                        # grow interval (routing stable); if changed, shrink.
                        # Inspired by adaptive branch prediction (MICRO'91).
                        cur_interval = self._layer_adaptive_interval.get(layer_name, sri)
                        max_interval = self.config.stale_remap_max_interval
                        if prev_set is not None and prev_set == unique_set:
                            # Routing stable → grow interval (exponential backoff)
                            new_interval = min(cur_interval * 2, max_interval)
                        else:
                            # Routing changed → reset to initial interval
                            new_interval = sri
                        self._layer_adaptive_interval[layer_name] = new_interval
                        self._layer_next_validation[layer_name] = step + 1 + new_interval
                        # --- TASER v2: Phase transition checks ---
                        _tv2_phase = self._taser_phase.get(layer_name)
                        if _tv2_phase == 'converge':
                            conv = self._taser_v2_compute_convergence(layer_name)
                            conv_steps = step - self._converge_start_step.get(layer_name, 0)
                            if (conv >= self.config.taser_converge_threshold
                                    or conv_steps > self.config.taser_converge_max_steps):
                                # Force-load missing hot experts into cache
                                _hot = self._hot_set.get(layer_name, set())
                                _cached = set(cache._slot_map.keys())
                                _missing = _hot - _cached
                                for _meid in _missing:
                                    _victim = None
                                    for _k in cache._slot_map:
                                        if _k not in _hot:
                                            _victim = _k
                                            break
                                    if _victim is not None:
                                        _ns, _ = cache.alloc_slot_with_victim(_meid, _victim)
                                    else:
                                        _ns = cache.alloc_slot(_meid)
                                    _pw13, _pw2 = cache.get_slot_tensors(_ns)
                                    _pw13.copy_(w13_ref[_meid], non_blocking=True)
                                    _pw2.copy_(w2_ref[_meid], non_blocking=True)
                                    if cache.has_aux_pools:
                                        _s13, _s2 = cache.get_slot_scale_tensors(_ns)
                                        _b13, _b2 = cache.get_slot_bias_tensors(_ns)
                                        if _s13 is not None:
                                            _s13.copy_(module.w13_weight_scale[_meid], non_blocking=True)
                                            _s2.copy_(module.w2_weight_scale[_meid], non_blocking=True)
                                        if _b13 is not None:
                                            _b13.copy_(module.w13_bias[_meid], non_blocking=True)
                                            _b2.copy_(module.w2_bias[_meid], non_blocking=True)
                                self._taser_v2_freeze_remap(layer_name)
                                self._taser_phase[layer_name] = 'cruise'
                                self._miss_ratio_ema[layer_name] = 0.0
                                if _taser_debug_layer:
                                    import sys
                                    _conv2 = self._taser_v2_compute_convergence(layer_name)
                                    print(f"[TASER-v2] {layer_name} → CRUISE at step={step}, "
                                          f"convergence={conv:.2f}→{_conv2:.2f}, "
                                          f"force_loaded={len(_missing)}, "
                                          f"hot_set_size={len(self._hot_set.get(layer_name, set()))}",
                                          file=sys.stderr, flush=True)
                        elif _tv2_phase == 'cruise':
                            # CRUISE validation step: refresh remap from
                            # current cache state.  The slow path just ran,
                            # so the cache is up-to-date.
                            self._taser_v2_freeze_remap(layer_name)
                else:
                    # Standard Scratchpad Mode (fallback)
                    if self._scratchpad_on_gpu:
                        scratch_w13 = self._scratch_w13[:meta["w13_shape"][0]]
                        scratch_w2 = self._scratch_w2[:meta["w2_shape"][0]]
                        if len(unique_list) > cache._max_slots:
                            # Pool overflow: copy directly from UVA to scratch,
                            # bypassing cache (whose slots were corrupted by
                            # eviction during miss-loading).
                            w13_ref_sc = module.w13_weight
                            w2_ref_sc = module.w2_weight
                            for eid in unique_list:
                                scratch_w13[eid].copy_(w13_ref_sc[eid],
                                                       non_blocking=True)
                                scratch_w2[eid].copy_(w2_ref_sc[eid],
                                                      non_blocking=True)
                        else:
                            for eid, slot in zip(hit_eids, hit_slots):
                                pool_w13, pool_w2 = cache.get_slot_tensors(slot)
                                scratch_w13[eid].copy_(pool_w13, non_blocking=True)
                                scratch_w2[eid].copy_(pool_w2, non_blocking=True)
                            for eid, slot in zip(miss_eids, miss_slots):
                                pool_w13, pool_w2 = cache.get_slot_tensors(slot)
                                scratch_w13[eid].copy_(pool_w13, non_blocking=True)
                                scratch_w2[eid].copy_(pool_w2, non_blocking=True)
                    else:
                        # Scratchpad-free mode (pool-direct): kernel will
                        # use original UVA weights directly (slow but rare,
                        # only during prefill overflow).
                        scratch_w13 = None
                        scratch_w2 = None
                self._maybe_profile_end(p3c_tok)

                self._total_cache_hits += step_hits
                self._total_cache_misses += step_misses
                self._total_pcie_bytes += pcie_bytes

        # Update adaptive budget EMA for this layer
        if self.config.enable_adaptive_budget:
            step_total = step_hits + step_misses
            if step_total > 0:
                inst_rate = step_hits / step_total
                alpha = self.config.hit_rate_ema_alpha
                old = self._hit_rate_ema.get(layer_name, 0.5)
                self._hit_rate_ema[layer_name] = (1 - alpha) * old + alpha * inst_rate

            # Entropy-Aware Budget (P4): track unique experts per step
            if self.config.enable_entropy_budget:
                n_unique = int(topk_ids.reshape(-1).unique().numel())
                gamma = self.config.entropy_ema_gamma
                old_ue = self._unique_experts_ema.get(layer_name, float(n_unique))
                self._unique_experts_ema[layer_name] = (
                    (1 - gamma) * old_ue + gamma * n_unique
                )

            self._rebalance_step += 1
            if (self.config.rebalance_interval > 0
                    and self._rebalance_step % self.config.rebalance_interval == 0):
                self._rebalance_cache_budget()

        # --- Phase 4: Kernel compute ---
        self._maybe_profile_end(p_tok)
        p_tok = self._maybe_profile_start("P4_kernel")

        # --- ALPS Phase 3: Launch shared expert on parallel stream ---
        # SharedExpert (~0.035 ms) runs concurrently with HBM-bound
        # MoE kernel (~0.35 ms). Completes well before MoE finishes.
        if run_shared_parallel:
            shared_input = module._get_shared_experts_input(hidden_states)
            se_stream = self._shared_expert_stream
            se_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(se_stream):
                shared_output = module.shared_experts(shared_input)

        M = hidden_states.shape[0]

        if taser_v2_dual_rail and _taser_v2_miss_data is not None:
            # === TASER v2 Dual-Rail: Hit → pool-direct, Miss → HFDE async ===
            _dr_remapped, _dr_is_miss, _dr_miss_list = _taser_v2_miss_data

            # Async load miss experts on prefetch_stream
            _dr_stream = self._prefetch_stream
            _dr_stream.wait_stream(torch.cuda.current_stream())

            w13_ref = module.w13_weight
            w2_ref = module.w2_weight
            _dr_miss_slots = {}
            hot_set = self._hot_set.get(layer_name, set())

            with torch.cuda.stream(_dr_stream):
                for eid in _dr_miss_list:
                    # Check if already cached
                    existing_slot = cache.get(eid)
                    if existing_slot is not None:
                        _dr_miss_slots[eid] = existing_slot
                        continue
                    # Alloc slot: prefer evicting non-hot experts
                    if not cache._free_slots:
                        victim = None
                        for k in cache._slot_map:
                            if k not in hot_set:
                                victim = k
                                break
                        if victim is not None:
                            new_slot, _ = cache.alloc_slot_with_victim(eid, victim)
                        else:
                            new_slot = cache.alloc_slot(eid)
                    else:
                        new_slot = cache.alloc_slot(eid)
                    pool_w13, pool_w2 = cache.get_slot_tensors(new_slot)
                    pool_w13.copy_(w13_ref[eid], non_blocking=True)
                    pool_w2.copy_(w2_ref[eid], non_blocking=True)
                    if cache.has_aux_pools:
                        s13, s2 = cache.get_slot_scale_tensors(new_slot)
                        b13, b2 = cache.get_slot_bias_tensors(new_slot)
                        if s13 is not None:
                            s13.copy_(module.w13_weight_scale[eid], non_blocking=True)
                            s2.copy_(module.w2_weight_scale[eid], non_blocking=True)
                        if b13 is not None:
                            b13.copy_(module.w13_bias[eid], non_blocking=True)
                            b2.copy_(module.w2_bias[eid], non_blocking=True)
                    _dr_miss_slots[eid] = new_slot

            # Pass 1: Hit experts (default stream, overlaps with PCIe)
            # Find a safe redirect slot from any hit position
            _dr_hit_mask = ~_dr_is_miss
            _dr_hit_remapped = _dr_remapped[_dr_hit_mask].reshape(-1)
            safe_slot = _dr_hit_remapped[0].item() if _dr_hit_remapped.numel() > 0 else 0

            hit_remapped = _dr_remapped.clone()
            hit_remapped[_dr_is_miss] = safe_slot
            hit_w = topk_weights.clone()
            hit_w[_dr_is_miss] = 0.0

            # Choose kernel path: direct dispatch or quant_method.apply
            _dr_use_dd = (
                self.config.enable_direct_dispatch
                and self._dd_invoke_kernel is not None
                and M <= self._dd_max_M
            )
            if _dr_use_dd:
                hit_output = self._direct_dispatch_kernel(
                    hidden_states, hit_w, hit_remapped, cache, module,
                )
            else:
                # Weight-swap path: swap module weights to pool, run kernel
                _dr_orig_w13 = module.w13_weight.data
                _dr_orig_w2 = module.w2_weight.data
                _dr_orig_ne = module.global_num_experts
                _dr_orig_s13 = None
                _dr_orig_s2 = None
                _dr_orig_b13 = None
                _dr_orig_b2 = None
                module.w13_weight.data = cache._w13_pool
                module.w2_weight.data = cache._w2_pool
                if cache._w13_scale_pool is not None:
                    _dr_orig_s13 = module.w13_weight_scale.data
                    _dr_orig_s2 = module.w2_weight_scale.data
                    module.w13_weight_scale.data = cache._w13_scale_pool
                    module.w2_weight_scale.data = cache._w2_scale_pool
                if cache._w13_bias_pool is not None:
                    _dr_orig_b13 = module.w13_bias.data
                    _dr_orig_b2 = module.w2_bias.data
                    module.w13_bias.data = cache._w13_bias_pool
                    module.w2_bias.data = cache._w2_bias_pool
                module.global_num_experts = cache._w13_pool.shape[0]
                try:
                    hit_output = module.quant_method.apply(
                        layer=module, x=hidden_states,
                        topk_weights=hit_w, topk_ids=hit_remapped,
                    )
                finally:
                    module.w13_weight.data = _dr_orig_w13
                    module.w2_weight.data = _dr_orig_w2
                    module.global_num_experts = _dr_orig_ne
                    if _dr_orig_s13 is not None:
                        module.w13_weight_scale.data = _dr_orig_s13
                        module.w2_weight_scale.data = _dr_orig_s2
                    if _dr_orig_b13 is not None:
                        module.w13_bias.data = _dr_orig_b13
                        module.w2_bias.data = _dr_orig_b2

            # Wait for miss loading
            torch.cuda.current_stream().wait_stream(_dr_stream)
            self._prefetch_in_flight = False

            # Build miss remapped: fill miss positions with loaded slot ids
            miss_remapped = _dr_remapped.clone()
            for eid, slot in _dr_miss_slots.items():
                miss_remapped[topk_ids == eid] = slot
            miss_remapped[_dr_hit_mask] = safe_slot
            miss_w = topk_weights.clone()
            miss_w[_dr_hit_mask] = 0.0

            if _dr_use_dd:
                miss_output = self._direct_dispatch_kernel(
                    hidden_states, miss_w, miss_remapped, cache, module,
                )
            else:
                module.w13_weight.data = cache._w13_pool
                module.w2_weight.data = cache._w2_pool
                if cache._w13_scale_pool is not None:
                    module.w13_weight_scale.data = cache._w13_scale_pool
                    module.w2_weight_scale.data = cache._w2_scale_pool
                if cache._w13_bias_pool is not None:
                    module.w13_bias.data = cache._w13_bias_pool
                    module.w2_bias.data = cache._w2_bias_pool
                module.global_num_experts = cache._w13_pool.shape[0]
                try:
                    miss_output = module.quant_method.apply(
                        layer=module, x=hidden_states,
                        topk_weights=miss_w, topk_ids=miss_remapped,
                    )
                finally:
                    module.w13_weight.data = _dr_orig_w13
                    module.w2_weight.data = _dr_orig_w2
                    module.global_num_experts = _dr_orig_ne
                    if _dr_orig_s13 is not None:
                        module.w13_weight_scale.data = _dr_orig_s13
                        module.w2_weight_scale.data = _dr_orig_s2
                    if _dr_orig_b13 is not None:
                        module.w13_bias.data = _dr_orig_b13
                        module.w2_bias.data = _dr_orig_b2

            final_hidden = hit_output + miss_output

            # Mark cache dirty: miss loading may have evicted hot experts,
            # so lr remap needs refresh on next validation step.
            self._layer_cache_dirty[layer_name] = True

        elif hfde_active:
            self._hfde_active_count += 1
            # === HFDE: Hit-First Disaggregated Execution ===
            # Two-pass kernel: overlap miss loading (PCIe, _prefetch_stream)
            # with hit expert compute (GPU HBM, default stream).
            #
            # Math: output = Σ_hits(w_i·E_i(x)) + Σ_misses(w_j·E_j(x))
            # The two sums are computed independently and added.
            #
            # Safety: hit pass redirects miss positions to a valid hit slot
            # with w=0, so the kernel only reads from HIT slots (no data
            # race with _prefetch_stream writing to MISS slots).

            # Build miss mask on GPU (no sync — all ops are GPU-side)
            remap_size = self._remap_table.shape[0]
            _dev = topk_ids.device
            miss_indicator = torch.zeros(remap_size, dtype=torch.bool, device=_dev)
            _miss_t = torch.tensor(miss_eids, dtype=torch.long, device=_dev)
            miss_indicator.scatter_(0, _miss_t, True)
            is_miss = miss_indicator[topk_ids]  # [M, top_k]

            # Safe redirect: first hit expert's slot (valid data, in cache)
            safe_slot = hit_slots[0]

            # --- Pass 1: Hit experts (default stream, overlaps with PCIe) ---
            hit_remapped = remapped_ids.clone()
            hit_remapped[is_miss] = safe_slot  # redirect misses to valid slot
            hit_w = topk_weights.clone()
            hit_w[is_miss] = 0.0  # zero contribution from redirected misses

            hit_output = self._direct_dispatch_kernel(
                hidden_states, hit_w, hit_remapped, cache, module,
            )

            # --- Wait for miss loading on _prefetch_stream ---
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            self._prefetch_in_flight = False

            # --- Pass 2: Miss experts (miss data now in pool) ---
            miss_remapped = remapped_ids.clone()
            miss_remapped[~is_miss] = safe_slot  # redirect hits to any valid slot
            miss_w = topk_weights.clone()
            miss_w[~is_miss] = 0.0  # zero contribution from hit experts

            miss_output = self._direct_dispatch_kernel(
                hidden_states, miss_w, miss_remapped, cache, module,
            )

            final_hidden = hit_output + miss_output

            # Oracle prefetch: skip this layer (_prefetch_stream was busy).
            # Next layer will benefit from TASER stale path instead.
        else:
            # --- Original Phase 4 path ---
            # ALPS Phase 2: Oracle cross-layer prefetch
            if self.config.enable_oracle_prefetch and self._ordered_layers:
                self._oracle_prefetch_next_layer(layer_name, topk_ids)

            # ALPS Phase 1: CUDA Graph replay for stale path
            use_graph = (
                self._use_cuda_graph
                and stale_remap_ok
                and layer_name in self._layer_graphs
                and M == self._graph_M
            )

            if use_graph:
                # CUDA Graph fast path (~99% of decode steps after warmup)
                self._graph_replay_count += 1
                final_hidden = self._layer_graphs[layer_name].replay(
                    hidden_states, topk_weights, remapped_ids,
                )
            else:
                self._graph_eager_count += 1
                use_direct = (
                    self.config.enable_direct_dispatch
                    and use_pool_direct
                    and self._dd_invoke_kernel is not None
                    and M <= self._dd_max_M
                )

                if use_direct:
                    self._dd_active_count += 1
                    # === Direct Triton Dispatch (bypass quant_method.apply chain) ===
                    final_hidden = self._direct_dispatch_kernel(
                        hidden_states, topk_weights, remapped_ids, cache, module,
                    )
                else:
                    # === Original path via quant_method.apply ===
                    orig_w13_data = module.w13_weight.data
                    orig_w2_data = module.w2_weight.data
                    orig_w13_scale = None
                    orig_w2_scale = None
                    orig_w13_bias = None
                    orig_w2_bias = None

                    if use_pool_direct:
                        module.w13_weight.data = cache._w13_pool
                        module.w2_weight.data = cache._w2_pool
                        # Swap scale/bias pools for quantized models
                        if cache._w13_scale_pool is not None:
                            orig_w13_scale = module.w13_weight_scale.data
                            orig_w2_scale = module.w2_weight_scale.data
                            module.w13_weight_scale.data = cache._w13_scale_pool
                            module.w2_weight_scale.data = cache._w2_scale_pool
                        if cache._w13_bias_pool is not None:
                            orig_w13_bias = module.w13_bias.data
                            orig_w2_bias = module.w2_bias.data
                            module.w13_bias.data = cache._w13_bias_pool
                            module.w2_bias.data = cache._w2_bias_pool
                        kernel_topk_ids = remapped_ids
                        orig_num_experts = module.global_num_experts
                        module.global_num_experts = cache._w13_pool.shape[0]
                    else:
                        if scratch_w13 is not None:
                            module.w13_weight.data = scratch_w13
                            module.w2_weight.data = scratch_w2
                        # else: scratchpad-free mode — keep original UVA
                        # weights; kernel reads via PCIe (rare fallback)
                        kernel_topk_ids = topk_ids

                    try:
                        final_hidden = module.quant_method.apply(
                            layer=module,
                            x=hidden_states,
                            topk_weights=topk_weights,
                            topk_ids=kernel_topk_ids,
                        )
                    finally:
                        module.w13_weight.data = orig_w13_data
                        module.w2_weight.data = orig_w2_data
                        if orig_w13_scale is not None:
                            module.w13_weight_scale.data = orig_w13_scale
                            module.w2_weight_scale.data = orig_w2_scale
                        if orig_w13_bias is not None:
                            module.w13_bias.data = orig_w13_bias
                            module.w2_bias.data = orig_w2_bias
                        if use_pool_direct:
                            module.global_num_experts = orig_num_experts

        # --- Phase 5: Shared experts ---
        self._maybe_profile_end(p_tok)
        p_tok = self._maybe_profile_start("P5_shared")
        if has_separate_shared:
            if run_shared_parallel:
                # ALPS Phase 3: wait for parallel shared expert stream
                torch.cuda.current_stream().wait_stream(self._shared_expert_stream)
            elif use_shared_stream:
                from vllm.utils.torch_utils import current_stream
                with torch.cuda.stream(module.shared_experts_stream):
                    shared_output = module.shared_experts(hidden_clone)
                current_stream().wait_stream(module.shared_experts_stream)
            final_hidden = (shared_output, final_hidden)

        # --- Optional logging ---
        self._maybe_profile_end(p_tok)
        self._maybe_profile_report()
        # LDR stats: periodic summary
        if (self._total_intercepts > 0
                and self._total_intercepts % 1000 == 0):
            ldr_fast = getattr(self, '_ldr_fast_count', 0)
            ldr_conflict = getattr(self, '_ldr_conflict_count', 0)
            if ldr_fast + ldr_conflict > 0:
                import sys
                print(f"[ELMM] LDR stats@{self._total_intercepts}: "
                      f"fast={ldr_fast}, conflict={ldr_conflict}, "
                      f"rate={ldr_fast/(ldr_fast+ldr_conflict)*100:.1f}%",
                      file=sys.stderr, flush=True)
        if (self.config.log_interval > 0
                and self._total_intercepts % self.config.log_interval == 0):
            self.log_stats()
        # CUDA Graph diagnostic: periodic summary (every 1000 intercepts)
        if (self._use_cuda_graph
                and self._total_intercepts > 0
                and self._total_intercepts % 1000 == 0):
            import sys
            total_g = self._graph_replay_count + self._graph_eager_count
            pct = self._graph_replay_count / total_g * 100 if total_g > 0 else 0
            print(f"[ELMM] Graph stats@{self._total_intercepts}: "
                  f"replay={self._graph_replay_count}, eager={self._graph_eager_count}, "
                  f"ratio={pct:.1f}%, M_captured={self._graph_M}",
                  file=sys.stderr, flush=True)

        # BriskMoE: advance step counter (once per layer per forward pass)
        if self._briskmoe_sacr is not None or self._briskmoe_elp is not None:
            self._briskmoe_step += 1
        if self._pred_cache is not None:
            self._pred_cache.advance_step()

        # --- v3.1 C6: Step feedback for overflow controller ---
        if self._overflow_controller is not None:
            cache = self._layer_caches.get(layer_name)
            _c6_evictions = cache._evictions if cache else 0
            self._overflow_controller.on_layer_complete(
                layer_name, step_hits, step_misses, _c6_evictions,
            )
            # Finalize step after last offloaded layer
            if (self._ordered_layers
                    and layer_name == self._ordered_layers[-1]):
                self._overflow_controller.on_step_complete()

        return final_hidden

    # -----------------------------------------------------------------------
    # Unified Scheduling Framework (D1-D6) per-layer forward
    # -----------------------------------------------------------------------

    def _unified_forward_layer(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        module: nn.Module,
    ) -> torch.Tensor:
        """
        Unified scheduling pipeline for one layer.

        Runs D1→D2→D6→D4→D5 pipeline:
          1. Governor decides K, pattern, aggressiveness
          2. VerifyScheduler decomposes into substeps
          3. PrefetchPlanner classifies experts per substep
          4. ResidencyManager allocates/pins slots
          5. HeadTailExecutor runs head+tail dual-stage kernel
          6. Stats feedback to Governor
        """
        from adapters.prefetch_planner import PrefetchPlanner

        residency = self._unified_residency
        governor = self._unified_governor
        scheduler = self._unified_scheduler
        executor = self._unified_executor

        # Track which layer we're on (for per-step Governor feedback)
        layer_idx = self._layer_name_to_id.get(layer_name, 0)
        is_first_layer = (layer_idx == 0)
        is_last_layer = (layer_idx == len(self._ordered_layers) - 1
                         if self._ordered_layers else True)

        # D1: Governor decide (once per step, on first layer)
        if is_first_layer:
            self._unified_step_count += 1
            self._unified_current_decision = governor.decide()

        decision = self._unified_current_decision

        # D2: VerifyScheduler plan substeps
        scheduler.set_pattern(decision.verify_pattern)
        plan = scheduler.plan(topk_ids, residency_manager=residency)

        # Classify experts for this layer
        unique_experts = topk_ids.reshape(-1).unique().tolist()
        full_list, partial_list, cold_list = residency.classify_experts(
            layer_name, unique_experts
        )

        # Track head coverage for stats
        head_hits = len(full_list) + len(partial_list)
        total_needed = len(unique_experts)

        # D6: PrefetchPlanner — create per-substep plan
        planner = PrefetchPlanner(
            residency=residency,
            stats=governor.stats,
            max_pcie_budget_bytes=self.config.unified_tail_slots
                * self._unified_split_config.tail_expert_bytes,
            tail_bytes=self._unified_split_config.tail_expert_bytes,
        )
        step_plan = planner.plan(
            plan,
            aggressiveness=decision.prefetch_aggressiveness,
            layer=layer_name,
        )

        # D6+D4: Execute prefetch plan (load tails for hard/soft experts)
        loaded_experts = planner.execute_plan(
            step_plan, layer=layer_name, stream=self._prefetch_stream
        )

        # Sync prefetch stream before execution
        if self._prefetch_stream is not None and loaded_experts:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # D4: Touch and step
        residency.touch_experts(layer_name, unique_experts)
        residency.step(layer_name)

        # D5: HeadTailExecutor forward
        final_hidden = executor.forward(
            hidden_states, topk_weights, topk_ids, residency, layer_name
        )

        # D4: Release any substep reservations
        for ss in plan.substeps:
            residency.release_substep_reservation(layer_name, ss.substep_id)

        # Determine readiness: did this layer need fallback?
        # Fallback = any cold expert with no loaded tail
        tail_ready_count = len(loaded_experts) + len(full_list)
        fell_back = len(cold_list) > 0 and not loaded_experts

        # D1: Governor feedback (once per step, on last layer)
        if is_last_layer:
            if fell_back:
                self._unified_fallback_count += 1
            else:
                self._unified_ready_count += 1
            governor.on_step_complete(
                fell_back=fell_back,
                accepted_tokens=topk_ids.shape[0],  # M tokens
                unique_experts=total_needed,
                head_hits=head_hits,
                tail_ready_count=tail_ready_count,
                total_needed=total_needed,
            )

        return final_hidden

    # -----------------------------------------------------------------------
    # Direct Triton Dispatch (Direction D)
    # -----------------------------------------------------------------------

    def _direct_dispatch_kernel(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        cache: _LayerExpertCache,
        module: nn.Module,
    ) -> torch.Tensor:
        """
        Directly invoke Triton FusedMoE kernels on the cache pool,
        bypassing the quant_method.apply() → FusedMoEModularKernel →
        TritonExperts call chain (~8 levels of Python indirection).

        Flow:
          1. moe_align_block_size(topk_ids, num_experts=pool_slots)
          2. invoke_fused_moe_triton_kernel(hidden → W1 → intermediate)
          3. silu_and_mul activation
          4. invoke_fused_moe_triton_kernel(intermediate → W2 → output)
          5. Sum over top_k dimension
        """
        import triton.language as tl

        invoke_kernel = self._dd_invoke_kernel
        align_fn = self._dd_align_fn
        config = self._dd_tile_config

        M = hidden_states.shape[0]
        top_k = self._dd_top_k

        w1 = cache._w13_pool   # [S, 2N, K]
        w2 = cache._w2_pool    # [S, K_out, N_in]
        pool_num_slots = w1.shape[0]

        N_w1 = w1.shape[1]          # 2*intermediate = 1536
        K_hidden = w1.shape[2]      # hidden_dim = 2048
        act_dim = N_w1 // 2         # intermediate = 768

        # Compute type
        dt = hidden_states.dtype
        if dt == torch.bfloat16:
            compute_type = tl.bfloat16
        elif dt == torch.float16:
            compute_type = tl.float16
        else:
            compute_type = tl.float32

        # 1. Align tokens by expert (using pool slot count, not 128)
        sorted_token_ids, expert_ids, num_tokens_post_padded = align_fn(
            topk_ids, config["BLOCK_SIZE_M"], pool_num_slots, None
        )

        # 2. Slice pre-allocated intermediate buffers
        inter_w1 = self._dd_inter_w1[:M, :top_k, :N_w1]
        inter_act = self._dd_inter_act[:M * top_k, :act_dim]
        inter_w2 = self._dd_inter_w2[:M, :top_k, :K_hidden]

        # 3. W1 kernel: hidden[M, K] × w1[slot, 2N, K]ᵀ → inter_w1[M, top_k, 2N]
        invoke_kernel(
            hidden_states, w1, inter_w1,
            None, None,          # A_scale, B_scale
            None,                # topk_weights (apply during W2)
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            False,               # mul_routed_weight
            top_k,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None, B_bias=None,
        )

        # 4. Activation: silu(gate) * up
        flat_w1 = inter_w1.reshape(-1, N_w1)
        if self._dd_silu_and_mul is not None:
            self._dd_silu_and_mul(inter_act, flat_w1)
        else:
            gate = flat_w1[:, :act_dim]
            up = flat_w1[:, act_dim:]
            inter_act.copy_(torch.nn.functional.silu(gate) * up)

        # 5. W2 kernel: inter_act[M*top_k, N] × w2[slot, K, N]ᵀ → inter_w2[M, top_k, K]
        apply_rw = not getattr(module, 'apply_router_weight_on_input', False)
        invoke_kernel(
            inter_act, w2, inter_w2,
            None, None,          # A_scale, B_scale
            topk_weights,        # for router weight multiplication
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            apply_rw,            # mul_routed_weight
            1,                   # top_k=1 for W2 (input is flat M*top_k)
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None, B_bias=None,
        )

        # 6. Sum over top_k dimension → [M, K]
        return inter_w2.sum(dim=1)

    # -----------------------------------------------------------------------
    # Phase Profiling
    # -----------------------------------------------------------------------

    def _maybe_profile_start(self, phase_name: str):
        """Record a CUDA event for phase start (only in profiling mode)."""
        if not self.config.enable_phase_profiling:
            return None
        n = self._total_intercepts
        warmup = self.config.profile_warmup
        steps = self.config.profile_steps
        if n < warmup or n >= warmup + steps:
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return (phase_name, ev)

    def _maybe_profile_end(self, token):
        """Record a CUDA event for phase end and store the pair."""
        if token is None:
            return
        phase_name, start_ev = token
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        self._prof_events.append((phase_name, start_ev, end_ev))

    def _maybe_profile_report(self):
        """After profiling window completes, sync and print timing report."""
        if not self.config.enable_phase_profiling:
            return
        n = self._total_intercepts
        warmup = self.config.profile_warmup
        steps = self.config.profile_steps
        target = warmup + steps
        if n != target:
            return
        import sys
        torch.cuda.synchronize()
        phase_ms: dict[str, list[float]] = {}
        for phase_name, start_ev, end_ev in self._prof_events:
            elapsed = start_ev.elapsed_time(end_ev)
            phase_ms.setdefault(phase_name, []).append(elapsed)
        num_layers = len(self._layer_caches) or 1
        num_steps = steps // num_layers  # rough step count
        print(f"\n[ELMM] === Phase Profiling Report ({steps} intercepts, "
              f"~{num_steps} decode steps) ===", file=sys.stderr, flush=True)
        total_ms = 0.0
        for phase, times in sorted(phase_ms.items()):
            avg = sum(times) / len(times)
            total = sum(times)
            total_ms += total
            print(f"  {phase:20s}: avg={avg:.3f} ms × {len(times)} = "
                  f"{total:.1f} ms total ({total / max(num_steps, 1):.2f} ms/step)",
                  file=sys.stderr, flush=True)
        print(f"  {'TOTAL':20s}: {total_ms:.1f} ms "
              f"({total_ms / max(num_steps, 1):.2f} ms/step)",
              file=sys.stderr, flush=True)
        print("[ELMM] === End Profiling ===\n", file=sys.stderr, flush=True)
        self._prof_events.clear()
        self.config.enable_phase_profiling = False  # auto-disable

    # -----------------------------------------------------------------------
    # Prefetch API (called from draft phase)
    # -----------------------------------------------------------------------

    def prefetch_experts(
        self,
        layer_name: str,
        expert_ids: list[int],
    ):
        """
        Async prefetch experts into GPU cache on a separate CUDA stream.
        Call during draft phase to overlap with draft model computation.
        """
        if not self.config.enable_prefetch:
            return
        cache = self._layer_caches.get(layer_name)
        if cache is None:
            return

        # --- CFDB: Cache-Full DIPP Bypass ---
        # When the cache is fully occupied, any prefetch of uncached
        # experts requires eviction.  Profiling shows DIPP evictions
        # always conflict with verify-step routing (100% LDR conflict
        # rate), because the evicted expert is typically still needed.
        # This forces TASER slow path, costing ~3.5ms per dirty layer.
        # Skip prefetch entirely when cache is full.
        if not cache._free_slots:
            return

        stream = self._prefetch_stream or torch.cuda.current_stream()
        module = self._patched_modules.get(layer_name)
        if module is None:
            return

        # Track what we prefetched for accuracy measurement
        if layer_name not in self._pending_prefetch:
            self._pending_prefetch[layer_name] = set()

        remap_tbl = self._layer_remap.get(layer_name)
        use_briskmoe = self._briskmoe_sacr is not None
        layer_id = self._layer_name_to_id.get(layer_name, 0)
        with torch.cuda.stream(stream):
            for eid in expert_ids:
                if cache.contains(eid):
                    continue
                # If cache is full, eviction will happen — update remap
                # so TASER's has_unmapped check detects the evicted expert.
                if not cache._free_slots and eid not in cache._slot_map:
                    if use_briskmoe:
                        # BriskMoE victim selection for prefetch eviction
                        if self._briskmoe_elp is not None:
                            candidates = self._briskmoe_elp.get_flex_candidates(layer_id)
                            candidates = [c for c in candidates if c in cache._slot_map]
                            if not candidates:
                                candidates = list(cache._slot_map.keys())
                        else:
                            candidates = list(cache._slot_map.keys())
                        victim = self._briskmoe_sacr.select_victim(layer_id, candidates)
                        if remap_tbl is not None:
                            remap_tbl[victim] = -1
                        slot, evicted = cache.alloc_slot_with_victim(eid, victim)
                        if evicted is not None:
                            self._briskmoe_sacr.remove_expert(layer_id, evicted)
                            if self._briskmoe_elp is not None:
                                self._briskmoe_elp.remove_expert(layer_id, evicted)
                    else:
                        evict_eid = next(iter(cache._slot_map))
                        if remap_tbl is not None:
                            remap_tbl[evict_eid] = -1
                        # Allocate a cache slot and H2D copy from UVA weight
                        slot = cache.alloc_slot(eid)
                else:
                    slot = cache.alloc_slot(eid)
                pool_w13, pool_w2 = cache.get_slot_tensors(slot)
                pool_w13.copy_(module.w13_weight[eid], non_blocking=True)
                pool_w2.copy_(module.w2_weight[eid], non_blocking=True)
                if cache.has_aux_pools:
                    s13, s2 = cache.get_slot_scale_tensors(slot)
                    b13, b2 = cache.get_slot_bias_tensors(slot)
                    if s13 is not None:
                        s13.copy_(module.w13_weight_scale[eid], non_blocking=True)
                        s2.copy_(module.w2_weight_scale[eid], non_blocking=True)
                    if b13 is not None:
                        b13.copy_(module.w13_bias[eid], non_blocking=True)
                        b2.copy_(module.w2_bias[eid], non_blocking=True)
                # Update remap for newly loaded expert
                if remap_tbl is not None:
                    remap_tbl[eid] = slot
                self._pending_prefetch[layer_name].add(eid)
        self._prefetch_in_flight = True
        # Mark layer dirty so TASER validates.  LDR flag enables the
        # lightweight conflict check instead of forcing full slow path.
        self._layer_cache_dirty[layer_name] = True
        self._layer_dirty_from_dipp[layer_name] = True

    def prefetch_for_draft_routing(
        self,
        draft_routing: dict[int, list[int]],
    ):
        """
        Prefetch experts predicted by draft model routing decisions.

        Called by the draft-phase hook after each draft token is generated.
        Maps layer indices to layer names and triggers async prefetch.

        Args:
            draft_routing: {layer_index: [expert_ids]} from draft model routing
        """
        if not self.config.enable_prefetch or not self._installed:
            return
        # Build layer_index → layer_name mapping (cached)
        if not hasattr(self, "_layer_idx_to_name"):
            self._layer_idx_to_name: dict[int, str] = {}
            for name in self._layer_caches:
                # Extract layer index from name like "model.layers.5.mlp.experts"
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            self._layer_idx_to_name[int(parts[i + 1])] = name
                        except ValueError:
                            pass
        for layer_idx, expert_ids in draft_routing.items():
            layer_name = self._layer_idx_to_name.get(layer_idx)
            if layer_name:
                self.prefetch_experts(layer_name, expert_ids)

    def sync_prefetch(self):
        """Wait for all prefetch transfers to complete."""
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()

    # -----------------------------------------------------------------------
    # Verify round lifecycle (for locality collection)
    # -----------------------------------------------------------------------

    def on_verify_round_end(self):
        """
        Signal end of a verify round. Flushes accumulated per-layer expert
        data to the structured locality analyzer.
        """
        if self._locality_analyzer is not None and self._current_round_experts:
            # Convert to analyzer format: {layer_id: [[expert_ids]]}
            expert_indices = {}
            for layer_name, eset in self._current_round_experts.items():
                # Extract layer index
                parts = layer_name.split(".")
                layer_id = 0
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_id = int(parts[i + 1])
                        except ValueError:
                            pass
                expert_indices[layer_id] = [list(eset)]
            self._locality_analyzer.record_verify_round(
                round_id=self._verify_round_counter,
                expert_indices=expert_indices,
            )
            self._verify_round_counter += 1
        self._current_round_experts.clear()

        # Periodic locality export
        if (self.config.locality_export_interval > 0
                and self.config.locality_export_dir
                and self._verify_round_counter % self.config.locality_export_interval == 0):
            self.export_locality_data()

    # -----------------------------------------------------------------------
    # Adaptive Cache Budget Rebalancing
    # -----------------------------------------------------------------------

    def _rebalance_cache_budget(self):
        """
        Redistribute cache slots across layers proportional to their
        smoothed hit rates. Layers with higher hit rates get more slots,
        layers with lower hit rates get fewer (subject to minimum).

        This is safe because layers execute sequentially and the scratchpad
        is shared — we only resize the per-layer LRU pool.
        """
        if not self._layer_caches or not self._hit_rate_ema:
            return

        total_slots = self._total_cache_slots
        num_layers = len(self._layer_caches)
        min_slots = max(1, int(total_slots * self.config.min_slot_fraction))

        # Compute weight per layer: higher hit rate → more valuable → more slots
        # Inverse logic: layers with LOW hit rate benefit MORE from extra slots.
        # But empirically, layers that already have high hit rates should keep
        # their slots. We use miss_rate as the allocation weight: layers with
        # more misses need more cache capacity.
        weights = {}
        for name, ema in self._hit_rate_ema.items():
            # miss_rate as weight — layers that miss more need more capacity
            miss_w = max(0.01, 1.0 - ema)
            # Entropy-Aware Budget (P4): scale by demand/capacity ratio
            if self.config.enable_entropy_budget and name in self._unique_experts_ema:
                ue = self._unique_experts_ema[name]
                slots = self._layer_caches[name]._max_slots if name in self._layer_caches else 23
                entropy_factor = (ue / max(slots, 1)) ** self.config.entropy_kappa
                weights[name] = miss_w * entropy_factor
            else:
                weights[name] = miss_w

        total_weight = sum(weights.values())
        if total_weight <= 0:
            return

        # Distribute slots proportional to miss rate
        allocatable = total_slots - min_slots * num_layers
        if allocatable <= 0:
            return

        new_allocations = {}
        for name in self._layer_caches:
            w = weights.get(name, 1.0 / num_layers)
            extra = int(allocatable * w / total_weight)
            new_allocations[name] = min_slots + extra

        # Fix rounding: distribute remainder to highest-weight layers
        allocated = sum(new_allocations.values())
        remainder = total_slots - allocated
        if remainder > 0:
            sorted_layers = sorted(weights, key=lambda n: weights[n], reverse=True)
            for i in range(remainder):
                new_allocations[sorted_layers[i % len(sorted_layers)]] += 1

        # Apply resizes
        resized = 0
        for name, cache in self._layer_caches.items():
            target = new_allocations.get(name, cache._max_slots)
            if target != cache._max_slots:
                cache.resize(target)
                # Mark dirty — remap table may reference evicted slots
                self._layer_cache_dirty[name] = True
                # Force immediate TASER validation so CRUISE takes
                # slow-path next step (v1 behavior: heal cache immediately
                # instead of deferring to next periodic validation).
                step = self._layer_remap_step.get(name, 0)
                self._layer_next_validation[name] = step
                resized += 1

        if resized > 0:
            import sys
            print(
                f"[ELMM] Rebalanced cache: {resized}/{num_layers} layers resized "
                f"(total={total_slots} slots)",
                file=sys.stderr, flush=True,
            )

    # -----------------------------------------------------------------------
    # Locality Data Export
    # -----------------------------------------------------------------------

    def _dump_overlap_history(self):
        """Auto-dump overlap history to JSON (called via atexit)."""
        path = self._overlap_dump_path
        if not path or not self._overlap_history:
            return
        import json as _json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, hist in sorted(self._overlap_history.items()):
            if hist:
                import numpy as _np
                arr = _np.array(hist)
                data[name] = {
                    "n": len(arr),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "p5": float(_np.percentile(arr, 5)),
                    "p25": float(_np.percentile(arr, 25)),
                    "median": float(_np.percentile(arr, 50)),
                    "p75": float(_np.percentile(arr, 75)),
                    "p95": float(_np.percentile(arr, 95)),
                    "p99": float(_np.percentile(arr, 99)),
                    "max": float(arr.max()),
                    "pct_ge_90": float((arr >= 0.90).mean() * 100),
                    "pct_ge_95": float((arr >= 0.95).mean() * 100),
                    "pct_ge_99": float((arr >= 0.99).mean() * 100),
                    "pct_eq_100": float((arr >= 1.0).mean() * 100),
                    "values": hist,
                }
        with open(path, "w") as f:
            _json.dump(data, f, indent=2)
        print(f"[ELMM] Overlap history dumped to {path} "
              f"({sum(len(v) for v in self._overlap_history.values())} samples)",
              file=sys.stderr, flush=True)

    def export_locality_data(self, output_dir: str = ""):
        """
        Export temporal locality data to files.

        Exports both the raw overlap history and the structured analyzer report.
        """
        import json
        from pathlib import Path

        out = Path(output_dir or self.config.locality_export_dir or "results/elmm_locality")
        out.mkdir(parents=True, exist_ok=True)

        # 1. Raw overlap history
        raw = {}
        for name, overlaps in self._overlap_history.items():
            if overlaps:
                raw[name] = {
                    "samples": len(overlaps),
                    "mean": sum(overlaps) / len(overlaps),
                    "last_10_mean": (
                        sum(overlaps[-10:]) / len(overlaps[-10:])
                        if len(overlaps) >= 10 else sum(overlaps) / len(overlaps)
                    ),
                }
        with open(out / "overlap_history.json", "w") as f:
            json.dump(raw, f, indent=2)

        # 2. Structured analyzer report
        if self._locality_analyzer is not None:
            report = self._locality_analyzer.generate_report()
            with open(out / "locality_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            self._locality_analyzer.export_csv(str(out))

        # 3. Per-layer cache stats
        cache_stats = {}
        for name, cache in self._layer_caches.items():
            cache_stats[name] = {
                "max_slots": cache._max_slots,
                "cached": len(cache._slot_map),
                "hit_rate": cache.hit_rate,
                "hits": cache._hits,
                "misses": cache._misses,
                "evictions": cache._evictions,
                "ema_hit_rate": self._hit_rate_ema.get(name, 0.0),
            }
        with open(out / "cache_stats.json", "w") as f:
            json.dump(cache_stats, f, indent=2)

        # 4. Prefetch accuracy
        if self._prefetch_total > 0:
            prefetch_stats = {
                "prefetch_hits": self._prefetch_hits,
                "prefetch_total": self._prefetch_total,
                "prefetch_accuracy": self._prefetch_hits / self._prefetch_total,
            }
            with open(out / "prefetch_stats.json", "w") as f:
                json.dump(prefetch_stats, f, indent=2)

        logger.info(f"Locality data exported to {out}")

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict:
        total_h = self._total_cache_hits
        total_m = self._total_cache_misses
        total = total_h + total_m
        stats = {
            "total_intercepts": self._total_intercepts,
            "total_cache_hits": total_h,
            "total_cache_misses": total_m,
            "overall_hit_rate": total_h / total if total > 0 else 0.0,
            "total_pcie_MB": self._total_pcie_bytes / 1024**2,
            "verify_rounds": self._verify_round_counter,
            "per_layer": {
                name: {
                    "hit_rate": c.hit_rate,
                    "hits": c._hits,
                    "misses": c._misses,
                    "evictions": c._evictions,
                    "cached": len(c._slot_map),
                    "max_slots": c._max_slots,
                    "ema_hit_rate": round(self._hit_rate_ema.get(name, 0.0), 4),
                }
                for name, c in self._layer_caches.items()
            },
            "temporal_locality": {
                name: {
                    "mean_overlap": (
                        sum(h) / len(h) if h else 0.0
                    ),
                    "samples": len(h),
                }
                for name, h in self._overlap_history.items()
            } if self._overlap_history else {},
        }
        # Prefetch accuracy
        if self._prefetch_total > 0:
            stats["prefetch"] = {
                "hits": self._prefetch_hits,
                "total": self._prefetch_total,
                "accuracy": round(self._prefetch_hits / self._prefetch_total, 4),
            }
        # Locality analyzer summary
        if self._locality_analyzer is not None and self._verify_round_counter > 0:
            loc = self._locality_analyzer.compute_statistics()
            stats["locality_summary"] = {
                "inter_round_overlap": loc.mean_interround_overlap,
                "mean_reuse_distance": loc.mean_reuse_distance,
                "draft_target_correlation": loc.mean_draft_target_correlation,
                "cache_hit_rate_estimate": loc.cache_hit_rate_estimate,
            }
        # CUDA Graph stats
        if self._layer_graphs:
            stats["cuda_graph"] = {
                "enabled": self._use_cuda_graph,
                "captured_layers": len(self._layer_graphs),
                "total_layers": len(self._layer_caches),
                "graph_M": self._graph_M,
            }
        # Oracle prefetch stats
        if self._oracle_prefetch_issued > 0 or self._oracle_prefetch_skipped > 0:
            stats["oracle_prefetch"] = {
                "issued": self._oracle_prefetch_issued,
                "skipped": self._oracle_prefetch_skipped,
            }
        # DD / HFDE stats
        stats["dd_active_count"] = self._dd_active_count
        stats["hfde_active_count"] = self._hfde_active_count
        if self._total_intercepts > 0:
            stats["hfde_activation_rate"] = round(
                self._hfde_active_count / self._total_intercepts, 4
            )
            stats["dd_activation_rate"] = round(
                self._dd_active_count / self._total_intercepts, 4
            )
        # TASER v2 stats
        _tv2_fast = getattr(self, '_taser_fast_count', 0)
        _tv2_slow = getattr(self, '_taser_slow_count', 0)
        if _tv2_fast + _tv2_slow + self._taser_v2_dual_rail_count > 0:
            stats["taser_v2"] = {
                "fast_pure": _tv2_fast,
                "cruise_pure": self._taser_v2_cruise_count,
                "dual_rail": self._taser_v2_dual_rail_count,
                "slow": _tv2_slow,
                "drift": self._taser_v2_drift_count,
                "phases": {n: p for n, p in list(self._taser_phase.items())[:3]},
            }
        # Unified scheduling (D1-D6) stats
        if self.config.enable_unified_scheduling and self._unified_governor is not None:
            gov = self._unified_governor
            gov_diag = gov.get_diagnostics()
            total_steps = self._unified_step_count
            stats["unified"] = {
                "enabled": True,
                "steps": total_steps,
                "fallback_count": self._unified_fallback_count,
                "ready_count": self._unified_ready_count,
                "fallback_ratio": round(self._unified_fallback_count / max(total_steps, 1), 4),
                "ready_ratio": round(self._unified_ready_count / max(total_steps, 1), 4),
                "governor_K": gov_diag.get("current_K", -1),
                "governor_fallback_ema": round(gov_diag.get("fallback_ratio_ema", 0), 4),
                "governor_coverage_ema": round(gov_diag.get("head_coverage_ema", 0), 4),
                "executor_stats": self._unified_executor.get_stats() if self._unified_executor else {},
            }
        return stats

    def log_stats(self):
        import sys
        s = self.get_stats()
        print(
            f"[ELMM] step={s['total_intercepts']}: "
            f"hit_rate={s['overall_hit_rate']:.1%}, "
            f"PCIe={s['total_pcie_MB']:.0f}MB",
            file=sys.stderr, flush=True,
        )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def shutdown(self):
        """Release all cache memory and restore original forward_impl."""
        if not self._installed:
            return

        # Always export final stats on shutdown
        if self.config.locality_export_dir:
            self.export_locality_data()

        # Also export if verify rounds were collected
        if not self.config.locality_export_dir and self._verify_round_counter > 0:
            pass  # no export dir configured

        # Restore original forward_impl on each patched module
        for name, module in self._patched_modules.items():
            orig = self._original_forward_impls.get(id(module))
            if orig is not None:
                module.forward_impl = orig

        # Free scratchpad
        self._scratch_w13 = None
        self._scratch_w2 = None

        # Clear caches
        for cache in self._layer_caches.values():
            cache._slot_map.clear()
            cache._w13_pool = None
            cache._w2_pool = None
        self._layer_caches.clear()
        # Release CUDA Graphs
        self._layer_graphs.clear()
        self._patched_modules.clear()
        self._original_forward_impls.clear()
        self._installed = False
        logger.info("ELMM shutdown complete")


# ---------------------------------------------------------------------------
# Global singleton (lives in GPU worker process)
# ---------------------------------------------------------------------------

_elmm_manager: Optional[ELMMManager] = None
_elmm_lock = threading.Lock()


def get_elmm_manager() -> Optional[ELMMManager]:
    return _elmm_manager


def activate_elmm(
    model: nn.Module,
    config: Optional[ELMMConfig] = None,
) -> ELMMManager:
    """
    Activate ELMM on a loaded vLLM model.

    Call this after model loading completes in the GPU worker process.

    Returns:
        The ELMMManager instance.
    """
    global _elmm_manager
    with _elmm_lock:
        if _elmm_manager is not None:
            logger.warning("ELMM already activated, returning existing manager")
            return _elmm_manager

        manager = ELMMManager(config)
        manager.install(model)
        _elmm_manager = manager
        return manager


def deactivate_elmm():
    """Deactivate and cleanup ELMM."""
    global _elmm_manager
    with _elmm_lock:
        if _elmm_manager is not None:
            _elmm_manager.shutdown()
            _elmm_manager = None
