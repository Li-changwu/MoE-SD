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

import logging
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
    )

    def __init__(
        self,
        layer_name: str,
        max_slots: int,
        w13_single_shape: tuple,
        w2_single_shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
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
        # expert_id → slot_index
        self._slot_map: OrderedDict[int, int] = OrderedDict()
        # Available slot indices
        self._free_slots: list[int] = list(range(max_slots))
        self._evictions = 0
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._slot_map

    def get(self, expert_id: int) -> int | None:
        """Returns slot index or None. Promotes on hit."""
        if expert_id in self._slot_map:
            self._slot_map.move_to_end(expert_id)
            self._hits += 1
            return self._slot_map[expert_id]
        self._misses += 1
        return None

    def get_slot_tensors(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (w13_view, w2_view) for the given slot index."""
        return self._w13_pool[slot], self._w2_pool[slot]

    def alloc_slot(self, expert_id: int) -> int:
        """Allocate a slot for this expert, evicting LRU if needed."""
        if expert_id in self._slot_map:
            self._slot_map.move_to_end(expert_id)
            return self._slot_map[expert_id]
        if not self._free_slots:
            # Evict LRU entry
            evict_eid, evict_slot = self._slot_map.popitem(last=False)
            self._free_slots.append(evict_slot)
            self._evictions += 1
        slot = self._free_slots.pop()
        self._slot_map[expert_id] = slot
        return slot


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
        # Prefetch stream
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if self.config.use_prefetch_stream and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()
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
        # Expert trace: layer_name → last step's expert set (for locality analysis)
        self._last_expert_set: dict[str, set[int]] = {}
        # Temporal locality metrics: layer_name → list of overlap ratios
        self._overlap_history: dict[str, list[float]] = {}

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
            any_offloaded = any(
                hasattr(p, "_vllm_offloaded_cpu_data")
                for p in module.parameters()
            )
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

            self._layer_meta[name] = {
                "num_experts": num_experts,
                "expert_size": expert_size,
                "w13_shape": tuple(w13.shape),
                "w2_shape": tuple(w2.shape),
                "dtype": w13.dtype,
            }

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
        self._scratch_w13 = torch.empty(
            max_w13_shape, dtype=ref_dtype, device=device
        )
        self._scratch_w2 = torch.empty(
            max_w2_shape, dtype=ref_dtype, device=device
        )
        scratch_mb = (max_w13_bytes + max_w2_bytes) / 1024**2

        # --- Distribute cache budget across layers (pre-allocate GPU pool) ---
        num_layers = len(offloaded_layers)
        per_layer_budget = self.config.gpu_cache_budget_bytes // num_layers
        device = torch.device("cuda")
        total_cache_alloc = 0

        for name, _module in offloaded_layers:
            meta = self._layer_meta[name]
            expert_size = meta["expert_size"]
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
        msg = (
            f"ELMM installed: {num_layers} offloaded layers, "
            f"scratchpad={scratch_mb:.0f} MB, "
            f"cache={total_cache_alloc / 1024**3:.2f} GB "
            f"(~{experts_per_layer} experts/layer)"
        )
        print(f"[ELMM] {msg}", file=sys.stderr, flush=True)
        logger.info(msg)

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
        shared_output = None
        if has_separate_shared and not use_shared_stream:
            shared_input = module._get_shared_experts_input(hidden_states)
            shared_output = module.shared_experts(shared_input)

        # --- Phase 2: Routing ---
        topk_weights, topk_ids = module.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # --- Phase 3: Cache lookup + scratchpad fill ---
        unique_experts = topk_ids.reshape(-1).unique()
        unique_list = unique_experts.tolist()
        unique_set = set(unique_list)

        # Track temporal locality
        prev_set = self._last_expert_set.get(layer_name)
        if prev_set is not None and len(prev_set) > 0:
            overlap = len(prev_set & unique_set) / len(prev_set | unique_set)
            if layer_name not in self._overlap_history:
                self._overlap_history[layer_name] = []
            self._overlap_history[layer_name].append(overlap)
        self._last_expert_set[layer_name] = unique_set

        step_hits = 0
        step_misses = 0
        pcie_bytes = 0

        # Scratchpad views (same shape as this layer's weights)
        scratch_w13 = self._scratch_w13[:meta["w13_shape"][0]]
        scratch_w2 = self._scratch_w2[:meta["w2_shape"][0]]

        # Weight references (UVA views for offloaded layers)
        w13_ref = module.w13_weight
        w2_ref = module.w2_weight

        for eid in unique_list:
            slot = cache.get(eid)
            if slot is not None:
                # Cache HIT: copy from pre-alloc pool → scratchpad (HBM→HBM fast)
                pool_w13, pool_w2 = cache.get_slot_tensors(slot)
                scratch_w13[eid].copy_(pool_w13, non_blocking=True)
                scratch_w2[eid].copy_(pool_w2, non_blocking=True)
                step_hits += 1
            else:
                # Cache MISS: UVA (PCIe ~25 GB/s) → scratchpad
                scratch_w13[eid].copy_(w13_ref[eid], non_blocking=True)
                scratch_w2[eid].copy_(w2_ref[eid], non_blocking=True)
                pcie_bytes += meta["expert_size"]
                step_misses += 1
                # Allocate a cache slot and copy scratchpad → pool
                new_slot = cache.alloc_slot(eid)
                pool_w13, pool_w2 = cache.get_slot_tensors(new_slot)
                pool_w13.copy_(scratch_w13[eid], non_blocking=True)
                pool_w2.copy_(scratch_w2[eid], non_blocking=True)

        # No explicit sync needed — all copies and kernel run on the same CUDA
        # stream, so operations execute in FIFO order.

        self._total_cache_hits += step_hits
        self._total_cache_misses += step_misses
        self._total_pcie_bytes += pcie_bytes

        # --- Phase 4: Swap param.data → scratchpad, run kernel, restore ---
        orig_w13_data = module.w13_weight.data
        orig_w2_data = module.w2_weight.data

        module.w13_weight.data = scratch_w13
        module.w2_weight.data = scratch_w2

        final_hidden = module.quant_method.apply(
            layer=module,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        # Restore UVA data immediately
        module.w13_weight.data = orig_w13_data
        module.w2_weight.data = orig_w2_data

        # --- Phase 5: Shared experts ---
        if has_separate_shared:
            if use_shared_stream:
                from vllm.utils.torch_utils import current_stream
                with torch.cuda.stream(module.shared_experts_stream):
                    shared_output = module.shared_experts(hidden_clone)
                current_stream().wait_stream(module.shared_experts_stream)
            final_hidden = (shared_output, final_hidden)

        # --- Optional logging ---
        if (self.config.log_interval > 0
                and self._total_intercepts % self.config.log_interval == 0):
            self.log_stats()

        return final_hidden

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

        stream = self._prefetch_stream or torch.cuda.current_stream()
        module = self._patched_modules.get(layer_name)
        if module is None:
            return
        device = torch.device("cuda")

        with torch.cuda.stream(stream):
            for eid in expert_ids:
                if cache.contains(eid):
                    continue
                # Allocate a cache slot and H2D copy from UVA weight
                slot = cache.alloc_slot(eid)
                pool_w13, pool_w2 = cache.get_slot_tensors(slot)
                pool_w13.copy_(module.w13_weight[eid], non_blocking=True)
                pool_w2.copy_(module.w2_weight[eid], non_blocking=True)
                cache.put(eid, w13_gpu, w2_gpu)

    def sync_prefetch(self):
        """Wait for all prefetch transfers to complete."""
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict:
        total_h = self._total_cache_hits
        total_m = self._total_cache_misses
        total = total_h + total_m
        return {
            "total_intercepts": self._total_intercepts,
            "total_cache_hits": total_h,
            "total_cache_misses": total_m,
            "overall_hit_rate": total_h / total if total > 0 else 0.0,
            "total_pcie_MB": self._total_pcie_bytes / 1024**2,
            "per_layer": {
                name: {
                    "hit_rate": c.hit_rate,
                    "hits": c._hits,
                    "misses": c._misses,
                    "evictions": c._evictions,
                    "cached": len(c._slot_map),
                    "max_slots": c._max_slots,
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

    def log_stats(self):
        s = self.get_stats()
        logger.info(
            f"ELMM step={s['total_intercepts']}: "
            f"hit_rate={s['overall_hit_rate']:.1%}, "
            f"PCIe={s['total_pcie_MB']:.0f}MB"
        )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def shutdown(self):
        """Release all cache memory and restore original forward_impl."""
        if not self._installed:
            return

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
