"""
Expert Weight Cache — GPU/CPU Tiered Caching with Prefetch
============================================================
For MoE models with many experts (e.g. Qwen3-30B: 128 experts × 48 layers),
not all expert weights fit in GPU memory simultaneously.

This module provides:
  1. LRU GPU cache: keeps hot experts in GPU HBM
  2. CPU staging buffer: all experts resident in CPU pinned memory
  3. Async prefetch: pre-load experts predicted by speculative draft routing
  4. Cache-aware scheduling: tells SpecFusedMoE which experts are "free" (cached)

Memory model (Qwen3-30B, bf16):
  Per expert: gate_proj + up_proj + down_proj = 3 × 768 × 2048 × 2B ≈ 9.4 MB
  Per layer:  128 experts × 9.4 MB ≈ 1.2 GB
  All layers: 48 × 1.2 GB ≈ 57.6 GB (far exceeds 48 GB GPU)
  GPU cache budget: configurable, e.g. 8 GB → ~850 experts across all layers

Key insight for SD: The draft model predicts which experts will be needed
in the upcoming verify pass. We can prefetch those experts while the draft
model is still running.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ExpertCacheConfig:
    """Configuration for the expert weight cache."""
    # Maximum GPU memory budget for expert cache (bytes)
    gpu_budget_bytes: int = 8 * 1024**3  # 8 GB default
    # Number of experts to keep in GPU cache per layer (0 = auto)
    max_cached_per_layer: int = 0
    # Enable async prefetch from CPU to GPU
    enable_prefetch: bool = True
    # Prefetch queue depth (how many experts to prefetch ahead)
    prefetch_depth: int = 16
    # Pin CPU memory for faster transfers
    pin_cpu_memory: bool = True
    # Eviction policy: "lru" | "frequency" | "speculative"
    eviction_policy: str = "lru"
    # Estimated bytes per expert (auto-computed if 0)
    expert_size_bytes: int = 0


@dataclass
class CacheStats:
    """Running cache performance statistics."""
    hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0
    prefetch_wasted: int = 0
    evictions: int = 0
    total_transfer_bytes: int = 0
    total_transfer_time_ms: float = 0.0
    total_lookups: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.total_lookups)

    @property
    def prefetch_accuracy(self) -> float:
        total_pf = self.prefetch_hits + self.prefetch_wasted
        return self.prefetch_hits / max(1, total_pf)

    @property
    def avg_transfer_ms(self) -> float:
        return self.total_transfer_time_ms / max(1, self.misses)

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_wasted": self.prefetch_wasted,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "prefetch_accuracy": round(self.prefetch_accuracy, 4),
            "total_transfer_mb": round(self.total_transfer_bytes / 1024**2, 1),
            "avg_transfer_ms": round(self.avg_transfer_ms, 3),
        }


class ExpertWeightCache:
    """
    GPU/CPU tiered expert weight cache with LRU eviction and prefetch support.

    Architecture:
    ┌─────────────────────────────────────────────┐
    │                  GPU HBM                     │
    │  ┌──────────────────────────────────────┐   │
    │  │  LRU Cache: (layer, expert) → tensor │   │
    │  │  Budget: configurable (e.g., 8 GB)   │   │
    │  └──────────────────────────────────────┘   │
    │        ↑ async copy (cudaMemcpyAsync)       │
    │        │                                     │
    │  ┌──────────────────────────────────────┐   │
    │  │  Prefetch Buffer (staging area)      │   │
    │  └──────────────────────────────────────┘   │
    └─────────────────────────────────────────────┘
              ↑ H2D transfer
    ┌─────────────────────────────────────────────┐
    │              CPU Pinned Memory               │
    │  Full expert weights for all layers/experts │
    │  (always resident, read-only source)        │
    └─────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: Optional[ExpertCacheConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or ExpertCacheConfig()
        self.device = torch.device(device) if isinstance(device, str) else device

        # GPU LRU cache: (layer_id, expert_id) -> {w1: Tensor, w2: Tensor}
        self._gpu_cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self._gpu_cache_bytes: int = 0

        # CPU weight store: (layer_id, expert_id) -> {w1: Tensor, w2: Tensor}
        self._cpu_store: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

        # Prefetch tracking
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_pending: set[tuple[int, int]] = set()
        self._prefetch_lock = threading.Lock()

        # Access frequency for frequency-based eviction
        self._access_count: dict[tuple[int, int], int] = {}

        # Stats
        self.stats = CacheStats()

        # Expert size (auto-detect on first registration)
        self._expert_size_bytes = self.config.expert_size_bytes

    def register_experts(
        self,
        layer_id: int,
        w1: torch.Tensor,  # [E, 2*N, D]
        w2: torch.Tensor,  # [E, D, N]
    ):
        """
        Register expert weights for a layer. Stores in CPU pinned memory.

        Args:
            layer_id: MoE layer index
            w1: Gate+up projection weights [num_experts, 2*intermediate, hidden]
            w2: Down projection weights [num_experts, hidden, intermediate]
        """
        num_experts = w1.shape[0]

        for eid in range(num_experts):
            key = (layer_id, eid)
            w1_cpu = w1[eid].contiguous().cpu()
            w2_cpu = w2[eid].contiguous().cpu()

            if self.config.pin_cpu_memory and torch.cuda.is_available():
                w1_cpu = w1_cpu.pin_memory()
                w2_cpu = w2_cpu.pin_memory()

            self._cpu_store[key] = {"w1": w1_cpu, "w2": w2_cpu}

        # Auto-detect expert size
        if self._expert_size_bytes == 0 and num_experts > 0:
            sample_w1 = w1[0]
            sample_w2 = w2[0]
            self._expert_size_bytes = (
                sample_w1.nelement() * sample_w1.element_size() +
                sample_w2.nelement() * sample_w2.element_size()
            )

        logger.info(
            f"Registered {num_experts} experts for layer {layer_id} "
            f"(~{self._expert_size_bytes / 1024**2:.1f} MB each)"
        )

    def get_expert(
        self,
        layer_id: int,
        expert_id: int,
    ) -> dict[str, torch.Tensor]:
        """
        Retrieve expert weights, loading from CPU if not cached.

        Returns:
            {"w1": Tensor on GPU, "w2": Tensor on GPU}
        """
        key = (layer_id, expert_id)
        self.stats.total_lookups += 1

        # Check GPU cache
        if key in self._gpu_cache:
            # Move to end (most recently used)
            self._gpu_cache.move_to_end(key)
            self._access_count[key] = self._access_count.get(key, 0) + 1
            self.stats.hits += 1
            if key in self._prefetch_pending:
                self.stats.prefetch_hits += 1
                self._prefetch_pending.discard(key)
            return self._gpu_cache[key]

        # Cache miss — load from CPU
        self.stats.misses += 1
        return self._load_to_gpu(key)

    def get_experts_batch(
        self,
        layer_id: int,
        expert_ids: list[int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """
        Batch retrieve multiple experts for a layer.
        Optimizes by batching CPU→GPU transfers.

        Returns:
            {expert_id: {"w1": Tensor, "w2": Tensor}}
        """
        result = {}
        to_load = []

        for eid in expert_ids:
            key = (layer_id, eid)
            self.stats.total_lookups += 1

            if key in self._gpu_cache:
                self._gpu_cache.move_to_end(key)
                self._access_count[key] = self._access_count.get(key, 0) + 1
                self.stats.hits += 1
                if key in self._prefetch_pending:
                    self.stats.prefetch_hits += 1
                    self._prefetch_pending.discard(key)
                result[eid] = self._gpu_cache[key]
            else:
                self.stats.misses += 1
                to_load.append(eid)

        # Batch load missing experts
        if to_load:
            for eid in to_load:
                result[eid] = self._load_to_gpu((layer_id, eid))

        return result

    def _load_to_gpu(self, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        """Load a single expert from CPU to GPU cache."""
        if key not in self._cpu_store:
            raise KeyError(f"Expert {key} not registered in CPU store")

        cpu_weights = self._cpu_store[key]

        # Ensure space in GPU cache
        self._ensure_space(self._expert_size_bytes)

        # Transfer
        t0 = time.monotonic()
        gpu_weights = {
            "w1": cpu_weights["w1"].to(self.device, non_blocking=True),
            "w2": cpu_weights["w2"].to(self.device, non_blocking=True),
        }
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        transfer_ms = (time.monotonic() - t0) * 1000

        # Insert into cache
        self._gpu_cache[key] = gpu_weights
        self._gpu_cache_bytes += self._expert_size_bytes
        self._access_count[key] = 1

        self.stats.total_transfer_bytes += self._expert_size_bytes
        self.stats.total_transfer_time_ms += transfer_ms

        return gpu_weights

    def _ensure_space(self, needed_bytes: int):
        """Evict LRU entries until there's enough space."""
        while (self._gpu_cache_bytes + needed_bytes > self.config.gpu_budget_bytes
               and self._gpu_cache):
            if self.config.eviction_policy == "frequency":
                # Evict least frequently accessed
                min_key = min(
                    self._gpu_cache.keys(),
                    key=lambda k: self._access_count.get(k, 0),
                )
                self._evict(min_key)
            else:
                # LRU: evict first (oldest) entry
                oldest_key = next(iter(self._gpu_cache))
                self._evict(oldest_key)

    def _evict(self, key: tuple[int, int]):
        """Evict an expert from GPU cache."""
        if key in self._gpu_cache:
            del self._gpu_cache[key]
            self._gpu_cache_bytes -= self._expert_size_bytes
            self._access_count.pop(key, None)
            self.stats.evictions += 1

    # -----------------------------------------------------------------------
    # Prefetch API
    # -----------------------------------------------------------------------

    def prefetch_experts(
        self,
        layer_id: int,
        expert_ids: list[int],
        priority: Optional[list[float]] = None,
    ):
        """
        Async prefetch experts predicted by draft model routing.
        Call this during draft generation to overlap with computation.

        Args:
            layer_id: Target MoE layer
            expert_ids: Predicted expert IDs to prefetch
            priority: Optional priority scores (higher = prefetch first)
        """
        if not self.config.enable_prefetch:
            return

        # Sort by priority if provided
        if priority is not None:
            sorted_pairs = sorted(zip(priority, expert_ids), reverse=True)
            expert_ids = [eid for _, eid in sorted_pairs]

        # Limit prefetch depth
        expert_ids = expert_ids[:self.config.prefetch_depth]

        for eid in expert_ids:
            key = (layer_id, eid)
            if key in self._gpu_cache:
                continue  # Already cached
            if key not in self._cpu_store:
                continue  # Not registered

            with self._prefetch_lock:
                if key in self._prefetch_pending:
                    continue
                self._prefetch_pending.add(key)

            # Async transfer
            if torch.cuda.is_available() and self._prefetch_stream is None:
                self._prefetch_stream = torch.cuda.Stream()

            self._async_load(key)

    def _async_load(self, key: tuple[int, int]):
        """Async CPU→GPU transfer on prefetch stream."""
        if key not in self._cpu_store:
            return

        cpu_weights = self._cpu_store[key]
        self._ensure_space(self._expert_size_bytes)

        if torch.cuda.is_available() and self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                gpu_weights = {
                    "w1": cpu_weights["w1"].to(self.device, non_blocking=True),
                    "w2": cpu_weights["w2"].to(self.device, non_blocking=True),
                }
        else:
            gpu_weights = {
                "w1": cpu_weights["w1"].to(self.device, non_blocking=True),
                "w2": cpu_weights["w2"].to(self.device, non_blocking=True),
            }

        self._gpu_cache[key] = gpu_weights
        self._gpu_cache_bytes += self._expert_size_bytes
        self._access_count[key] = 0

    def sync_prefetch(self):
        """Wait for all pending prefetch transfers to complete."""
        if self._prefetch_stream is not None and torch.cuda.is_available():
            self._prefetch_stream.synchronize()

    def mark_prefetch_wasted(self, layer_id: int, expert_ids: list[int]):
        """Mark prefetched experts that were not actually used (waste tracking)."""
        for eid in expert_ids:
            key = (layer_id, eid)
            if key in self._prefetch_pending:
                self.stats.prefetch_wasted += 1
                self._prefetch_pending.discard(key)

    # -----------------------------------------------------------------------
    # Cache State Queries
    # -----------------------------------------------------------------------

    def is_cached(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self._gpu_cache

    def get_cached_experts(self, layer_id: int) -> list[int]:
        """Return list of expert IDs currently cached for a layer."""
        return [eid for (lid, eid) in self._gpu_cache if lid == layer_id]

    def get_cache_occupancy(self) -> dict:
        """Return cache memory usage stats."""
        return {
            "cached_experts": len(self._gpu_cache),
            "gpu_cache_bytes": self._gpu_cache_bytes,
            "gpu_budget_bytes": self.config.gpu_budget_bytes,
            "utilization": round(self._gpu_cache_bytes / max(1, self.config.gpu_budget_bytes), 4),
            "registered_experts": len(self._cpu_store),
        }

    def get_statistics(self) -> dict:
        result = self.stats.to_dict()
        result.update(self.get_cache_occupancy())
        return result

    def reset_statistics(self):
        self.stats = CacheStats()

    def clear_gpu_cache(self):
        """Evict all entries from GPU cache."""
        self._gpu_cache.clear()
        self._gpu_cache_bytes = 0
        self._access_count.clear()
        self._prefetch_pending.clear()


class PrefetchScheduler:
    """
    Coordinates expert prefetch based on draft model routing predictions.

    Integrates with ExpertWeightCache to prefetch experts during draft
    generation, overlapping CPU→GPU transfer with compute.

    Usage:
        scheduler = PrefetchScheduler(cache)
        # During draft generation:
        scheduler.on_draft_routing(layer=5, experts=[3, 7, 12])
        # Before verify:
        scheduler.prepare_verify(K=3)
        # After verify:
        scheduler.report_verify_result(used_experts={5: [3, 7, 12, 45]})
    """

    def __init__(
        self,
        cache: ExpertWeightCache,
        num_layers: int = 48,
    ):
        self.cache = cache
        self.num_layers = num_layers

        # Predicted expert sets from draft model: layer -> set of expert_ids
        self._draft_predictions: dict[int, set[int]] = {}
        # Actually used experts: layer -> set of expert_ids
        self._verify_used: dict[int, set[int]] = {}

        # Running accuracy tracking
        self._total_predicted = 0
        self._total_correct = 0

    def on_draft_routing(
        self,
        layer_id: int,
        expert_ids: list[int],
        token_acceptance_probs: Optional[list[float]] = None,
    ):
        """
        Called when draft model produces routing decisions.
        Triggers async prefetch for predicted experts.

        Args:
            layer_id: MoE layer
            expert_ids: Experts selected by draft routing for this layer
            token_acceptance_probs: Per-token acceptance probability (for priority)
        """
        if layer_id not in self._draft_predictions:
            self._draft_predictions[layer_id] = set()
        self._draft_predictions[layer_id].update(expert_ids)

        # Compute priority: experts needed by high-acceptance tokens are more valuable
        if token_acceptance_probs is not None:
            # Weight expert priority by sum of acceptance probs of tokens needing it
            expert_priority = {}
            for eid in expert_ids:
                expert_priority[eid] = expert_priority.get(eid, 0) + max(token_acceptance_probs)
            sorted_eids = sorted(expert_priority.keys(), key=lambda e: expert_priority[e], reverse=True)
            priorities = [expert_priority[e] for e in sorted_eids]
            self.cache.prefetch_experts(layer_id, sorted_eids, priority=priorities)
        else:
            self.cache.prefetch_experts(layer_id, expert_ids)

    def prepare_verify(self, K: int):
        """
        Called before verify phase starts. Ensures all prefetch transfers complete.
        """
        self.cache.sync_prefetch()

    def report_verify_result(
        self,
        used_experts: dict[int, list[int]],
    ):
        """
        Called after verify completes. Reports which experts were actually used.

        Args:
            used_experts: {layer_id: [expert_ids]} actually used during verify
        """
        for layer_id, eids in used_experts.items():
            self._verify_used[layer_id] = set(eids)

            predicted = self._draft_predictions.get(layer_id, set())
            actual = set(eids)

            # Accuracy tracking
            self._total_predicted += len(predicted)
            self._total_correct += len(predicted & actual)

            # Mark wasted prefetches
            wasted = predicted - actual
            if wasted:
                self.cache.mark_prefetch_wasted(layer_id, list(wasted))

        # Clear for next round
        self._draft_predictions.clear()
        self._verify_used.clear()

    @property
    def prefetch_accuracy(self) -> float:
        return self._total_correct / max(1, self._total_predicted)

    def get_statistics(self) -> dict:
        return {
            "total_predicted": self._total_predicted,
            "total_correct": self._total_correct,
            "prefetch_accuracy": round(self.prefetch_accuracy, 4),
            "cache_stats": self.cache.get_statistics(),
        }

    def reset_statistics(self):
        self._total_predicted = 0
        self._total_correct = 0
        self.cache.reset_statistics()
