"""
BriskMoE Cache — Unified Facade for SACR + ELP + DIPP
========================================================
Integrates the three core algorithms into a single cache interface
compatible with ExpertWeightCache's API.

Data flow per SD cycle:

  Draft Phase (~30ms):
    Draft model generates K tokens
    ├── on_draft_token() → DIPP updates + schedules prefetch
    └── Progressive: each draft token → incremental prefetch

  Verify Phase:
    Target model verifies K+1 tokens
    ├── get_expert() → cache hit/miss with ELP+SACR eviction
    └── on_verify_complete() → tracker + SACR + ELP update

  Eviction (on miss):
    1. ELP: only Flex Zone candidates
    2. SACR: select argmin Score(e) from Flex candidates
    3. Execute eviction

Synergy effects:
  - SACR + ELP: SACR operates in smaller Flex Zone → higher precision
  - DIPP + SACR: DIPP reduces miss count, SACR improves replacement quality
  - ELP + DIPP: Pinned experts don't need prefetch → DIPP budget used better
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from adapters.accept_reject_tracker import (
    AcceptRejectTracker,
    AcceptRejectTrackerConfig,
)
from adapters.sacr import SACREvictionPolicy, SACRConfig, CacheMeta
from adapters.elp import ExpertLifecyclePartition, ELPConfig
from adapters.dipp import DraftInformedPrioritizedPreloader, DIPPConfig


@dataclass
class BriskMoECacheConfig:
    """Configuration for the unified BriskMoE cache."""
    # Total cache slots per layer
    total_slots_per_layer: int = 17
    # Sub-component configs
    tracker: AcceptRejectTrackerConfig = field(
        default_factory=AcceptRejectTrackerConfig
    )
    sacr: SACRConfig = field(default_factory=SACRConfig)
    elp: ELPConfig = field(default_factory=ELPConfig)
    dipp: DIPPConfig = field(default_factory=DIPPConfig)
    # Rebalance ELP every N steps
    rebalance_interval: int = 10
    # Number of MoE layers
    num_layers: int = 48


@dataclass
class BriskMoECacheStats:
    """Aggregated cache statistics."""
    total_accesses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    verify_callbacks: int = 0
    draft_callbacks: int = 0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(1, self.total_accesses)


class BriskMoECache:
    """
    Unified facade integrating SACR + ELP + DIPP.

    This is a logical cache manager — it tracks which experts are cached
    and makes eviction/prefetch decisions. Actual GPU memory management
    is delegated to the underlying ExpertWeightCache (not owned here).

    Usage:
        cache = BriskMoECache(config)

        # On each access
        is_hit, victim = cache.access_expert(layer_id=0, expert_id=5, step=10)
        if not is_hit:
            # Load expert from CPU, evict victim if not None

        # After SD verify
        cache.on_verify_complete(layer_id, token_expert_map, accepted_mask, step)

        # During draft
        schedule = cache.on_draft_token(k, router_preds, step)
    """

    def __init__(self, config: Optional[BriskMoECacheConfig] = None):
        self.config = config or BriskMoECacheConfig()

        # Core components
        self.tracker = AcceptRejectTracker(self.config.tracker)
        self.sacr = SACREvictionPolicy(
            config=self.config.sacr, tracker=self.tracker
        )
        self.elp = ExpertLifecyclePartition(
            config=self.config.elp,
            total_slots=self.config.total_slots_per_layer,
        )
        self.dipp = DraftInformedPrioritizedPreloader(config=self.config.dipp)

        # Logical cache state: layer_id → set of cached expert_ids
        self._cache_state: dict[int, set[int]] = {}

        # Statistics
        self.stats = BriskMoECacheStats()
        self._step = 0

    def access_expert(
        self,
        layer_id: int,
        expert_id: int,
        step: Optional[int] = None,
    ) -> tuple[bool, Optional[int]]:
        """
        Access an expert. Returns (is_cache_hit, victim_to_evict).

        If hit: (True, None)
        If miss and cache not full: (False, None) — just load
        If miss and cache full: (False, victim_expert_id) — evict victim first

        The caller is responsible for actually loading/evicting tensors.
        """
        if step is not None:
            self._step = max(self._step, step)

        self._ensure_layer(layer_id)
        self.stats.total_accesses += 1

        # Record access in SACR and ELP
        self.sacr.record_access(layer_id, expert_id, step=self._step)
        self.elp.access(layer_id, expert_id, step=self._step)

        # Cache hit?
        if expert_id in self._cache_state[layer_id]:
            self.stats.cache_hits += 1
            # Feed hit rate to SACR for adaptive γ
            self.sacr.update_hit_rate(layer_id, self.stats.hit_rate)
            return True, None

        # Cache miss
        self.stats.cache_misses += 1
        # Feed hit rate to SACR for adaptive γ
        self.sacr.update_hit_rate(layer_id, self.stats.hit_rate)

        # Need eviction?
        current_size = len(self._cache_state[layer_id])
        if current_size < self.config.total_slots_per_layer:
            # Space available — just insert
            self._cache_state[layer_id].add(expert_id)
            return False, None

        # Eviction needed — select victim from Flex Zone via SACR
        victim = self._select_victim(layer_id)
        if victim is not None:
            self._evict(layer_id, victim)
        self._cache_state[layer_id].add(expert_id)

        return False, victim

    def on_verify_complete(
        self,
        layer_id: int,
        token_expert_map: dict[int, list[int]],
        accepted_mask: list[bool],
        step: Optional[int] = None,
    ) -> None:
        """
        Callback after SD verify step for one layer.
        Updates tracker → SACR scores → ELP lifecycle.
        """
        if step is not None:
            self._step = max(self._step, step)
        self.stats.verify_callbacks += 1

        # 1. Update AcceptRejectTracker
        self.tracker.record_verify_result(
            layer_id=layer_id,
            token_expert_map=token_expert_map,
            accepted_mask=accepted_mask,
            step_id=self._step,
        )

        # 2. Periodic ELP rebalance
        if self._step % self.config.rebalance_interval == 0:
            self.elp.rebalance(layer_id)

    def on_draft_token(
        self,
        draft_token_idx: int,
        router_predictions: dict[int, list[int]],
        step: Optional[int] = None,
    ) -> list[tuple[int, int, float]]:
        """
        Callback after each draft token generation.
        Triggers DIPP progressive prefetch scheduling.

        Returns:
            Newly scheduled (layer_id, expert_id, value) for prefetch.
        """
        if step is not None:
            self._step = max(self._step, step)
        self.stats.draft_callbacks += 1

        return self.dipp.on_draft_token(
            draft_token_idx, router_predictions, self._cache_state
        )

    def compute_full_prefetch_schedule(
        self,
        predictions: dict[int, dict[int, list[int]]],
    ) -> list[tuple[int, int, float]]:
        """
        Compute full DIPP schedule using current cache state.

        Args:
            predictions: {layer_id: {token_pos: [expert_ids]}}.

        Returns:
            Prioritized prefetch list.
        """
        return self.dipp.compute_schedule(predictions, self._cache_state)

    def begin_draft_round(self) -> None:
        """Reset DIPP state for a new draft round."""
        self.dipp.reset_round()

    def get_cache_state(self, layer_id: int) -> set[int]:
        """Get currently cached expert IDs for a layer."""
        self._ensure_layer(layer_id)
        return set(self._cache_state[layer_id])

    def get_stats_summary(self) -> dict:
        """Get combined statistics from all components."""
        return {
            "cache": {
                "hit_rate": round(self.stats.hit_rate, 4),
                "hits": self.stats.cache_hits,
                "misses": self.stats.cache_misses,
                "evictions": self.stats.evictions,
            },
            "tracker": {
                "verify_callbacks": self.stats.verify_callbacks,
            },
            "dipp": {
                "schedules": self.dipp.stats.total_schedules,
                "experts_scheduled": self.dipp.stats.total_experts_scheduled,
                "over_budget": self.dipp.stats.total_experts_over_budget,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_layer(self, layer_id: int) -> None:
        if layer_id not in self._cache_state:
            self._cache_state[layer_id] = set()

    def _select_victim(self, layer_id: int) -> Optional[int]:
        """Select eviction victim: Flex Zone candidates scored by SACR."""
        flex_candidates = self.elp.get_flex_candidates(layer_id)

        if not flex_candidates:
            # All experts are pinned — fallback: use full cache set
            all_cached = list(self._cache_state.get(layer_id, set()))
            if not all_cached:
                return None
            return self.sacr.select_victim(layer_id, all_cached)

        # Filter to actually cached experts
        cached = self._cache_state.get(layer_id, set())
        eligible = [e for e in flex_candidates if e in cached]

        if not eligible:
            # Flex candidates not in cache — use full cache
            all_cached = list(cached)
            if not all_cached:
                return None
            return self.sacr.select_victim(layer_id, all_cached)

        return self.sacr.select_victim(layer_id, eligible)

    def _evict(self, layer_id: int, expert_id: int) -> None:
        """Evict an expert: update all components."""
        self._cache_state.get(layer_id, set()).discard(expert_id)
        self.sacr.remove_expert(layer_id, expert_id)
        self.elp.remove_expert(layer_id, expert_id)
        self.stats.evictions += 1
