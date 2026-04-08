"""
PredCache — Predictive Expert Cache Management
================================================
Forward-looking eviction + prefetch using speculation-provided future demand.

Core insight: SD's draft phase provides "future knowledge" about expert
demand — the exact information Belady's OPT algorithm needs. PredCache
uses this to approximate OPT online:

  - PredEvict: evict experts with lowest predicted future demand
  - PredLoad:  prioritize prefetching by predicted demand × urgency
  - PredReserve: protect cached experts with high predicted demand

PredScore(e) = D_hat(e) + λ·R(e)

  D_hat(e) = predicted demand count (from last verify's router logits)
  R(e)     = normalized LRU recency (fallback for unpredicted experts)
  λ        = fallback weight (decreases as prediction confidence grows)

When D_hat is unavailable (cold start), PredCache degrades gracefully to LRU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredCacheConfig:
    """Configuration for PredCache."""
    # Fallback weight for LRU recency when prediction is uncertain
    lru_fallback_weight: float = 10.0
    # Top-k used by router (for demand counting)
    top_k: int = 8
    # Number of experts per layer
    num_experts: int = 128
    # Max prefetch experts per round (PCIe bandwidth budget)
    max_prefetch_experts: int = 79
    # Urgency decay for cross-layer prefetch priority
    urgency_decay: str = "inverse"  # "inverse", "linear", "exp"
    # EMA decay factor for demand signal (0 = only latest, 1 = never forget)
    demand_decay: float = 0.5


class PredictiveExpertCacheManager:
    """
    Unified forward-looking cache manager for SD-aware MoE inference.

    Uses router logits from the previous verification step as predictions
    for the next step's expert demand. This is a lightweight approximation
    that exploits temporal locality in routing patterns.

    Phase A (real draft routing): In future, can be upgraded to use
    draft model's hidden states → target router gate for true K×L prediction.
    """

    def __init__(self, config: Optional[PredCacheConfig] = None):
        self.config = config or PredCacheConfig()
        # Per-layer predicted demand: layer_id → list[float] of length num_experts
        # demand[e] = raw demand (before lazy decay)
        self._predicted_demand: dict[int, list[float]] = {}
        # Per-layer demand update step: layer_id → list[int]
        # Tracks when demand was last updated for lazy decay computation
        self._demand_step: dict[int, list[int]] = {}
        # Per-layer LRU timestamps: layer_id → list[int] (last access step per expert)
        self._last_access: dict[int, list[int]] = {}
        # Global step counter
        self._step: int = 0
        # Stats
        self._evictions_pred: int = 0     # evictions guided by prediction
        self._evictions_lru: int = 0      # evictions falling back to LRU
        self._prefetch_scheduled: int = 0

    def _ensure_layer(self, layer_id: int) -> None:
        """Lazy-init arrays for a new layer."""
        if layer_id not in self._predicted_demand:
            n = self.config.num_experts
            self._predicted_demand[layer_id] = [0.0] * n
            self._demand_step[layer_id] = [0] * n
            self._last_access[layer_id] = [0] * n

    def _get_demand(self, layer_id: int, expert_id: int) -> float:
        """Get lazily-decayed demand for an expert. O(1)."""
        raw = self._predicted_demand[layer_id][expert_id]
        if raw == 0.0:
            return 0.0
        age = self._step - self._demand_step[layer_id][expert_id]
        if age <= 0:
            return raw
        return raw * (self.config.demand_decay ** age)

    def record_access(self, layer_id: int, expert_id: int) -> None:
        """Record that expert was accessed at current step."""
        self._ensure_layer(layer_id)
        self._last_access[layer_id][expert_id] = self._step

    def record_access_batch(self, layer_id: int, expert_ids: list[int]) -> None:
        """Record batch of expert accesses."""
        self._ensure_layer(layer_id)
        la = self._last_access[layer_id]
        step = self._step
        for eid in expert_ids:
            la[eid] = step

    def update_predictions_from_logits(
        self, layer_id: int, router_logits_topk_ids: list[list[int]]
    ) -> None:
        """
        Update predicted demand from router's topk_ids.

        Uses lazy decay — only touches the experts that appear in routing.
        O(top_k × num_tokens) instead of O(num_experts).
        """
        self._ensure_layer(layer_id)
        demand = self._predicted_demand[layer_id]
        ds = self._demand_step[layer_id]
        step = self._step
        decay = self.config.demand_decay
        n = self.config.num_experts
        # Update only touched experts (lazy decay + add)
        for token_experts in router_logits_topk_ids:
            for eid in token_experts:
                if 0 <= eid < n:
                    age = step - ds[eid]
                    demand[eid] = demand[eid] * (decay ** age) + 1.0
                    ds[eid] = step

    def update_predictions_from_flat(
        self, layer_id: int, topk_ids_flat: list[int]
    ) -> None:
        """
        Update predicted demand from a flat list of expert IDs.
        Uses lazy decay — O(len(topk_ids_flat)) not O(num_experts).
        """
        self._ensure_layer(layer_id)
        demand = self._predicted_demand[layer_id]
        ds = self._demand_step[layer_id]
        step = self._step
        decay = self.config.demand_decay
        n = self.config.num_experts
        for eid in topk_ids_flat:
            if 0 <= eid < n:
                age = step - ds[eid]
                demand[eid] = demand[eid] * (decay ** age) + 1.0
                ds[eid] = step

    def pred_score(self, layer_id: int, expert_id: int) -> float:
        """
        Compute PredScore for a cached expert.

        PredScore = D_hat(e) + λ·R(e)

        Higher score = more valuable = should NOT be evicted.
        """
        self._ensure_layer(layer_id)
        demand = self._get_demand(layer_id, expert_id)
        # Normalized recency: recent access → high R
        age = self._step - self._last_access[layer_id][expert_id]
        recency = 1.0 / (1.0 + age)
        return demand + self.config.lru_fallback_weight * recency

    def select_victim(
        self, layer_id: int, candidates: list[int]
    ) -> int:
        """
        Select the best eviction victim from candidates.

        Evicts the expert with the lowest PredScore
        (= lowest predicted future demand + lowest recency).
        """
        if not candidates:
            raise ValueError("No candidates for eviction")

        self._ensure_layer(layer_id)
        demand_arr = self._predicted_demand[layer_id]
        ds = self._demand_step[layer_id]
        la = self._last_access[layer_id]
        step = self._step
        lam = self.config.lru_fallback_weight
        decay = self.config.demand_decay

        best_victim = candidates[0]
        eid = best_victim
        d_age = step - ds[eid]
        d_raw = demand_arr[eid]
        d_val = d_raw * (decay ** d_age) if d_raw != 0.0 and d_age > 0 else d_raw
        best_score = d_val + lam / (1.0 + step - la[eid])

        for i in range(1, len(candidates)):
            eid = candidates[i]
            d_age = step - ds[eid]
            d_raw = demand_arr[eid]
            d_val = d_raw * (decay ** d_age) if d_raw != 0.0 and d_age > 0 else d_raw
            score = d_val + lam / (1.0 + step - la[eid])
            if score < best_score:
                best_score = score
                best_victim = eid

        return best_victim

    def get_demand_boost(self, layer_id: int, expert_id: int) -> float:
        """
        Get demand-based priority boost for an expert. O(1).

        Returns the lazily-decayed EMA demand. Used by DIPP to enhance
        its Value function: V(l,e) = (1 + demand_boost) × urgency(l).
        """
        if layer_id not in self._predicted_demand:
            return 0.0
        n = self.config.num_experts
        if expert_id < 0 or expert_id >= n:
            return 0.0
        return self._get_demand(layer_id, expert_id)

    def compute_prefetch_schedule(
        self,
        cache_states: dict[int, set[int]],
        num_layers: int = 48,
    ) -> list[tuple[int, int, float]]:
        """
        Compute cross-layer prefetch schedule using predicted demand.

        Returns list of (layer_id, expert_id, priority) sorted by
        priority descending, truncated to bandwidth budget.

        Priority = demand(e) × urgency(layer) for miss experts.
        """
        candidates: list[tuple[float, int, int]] = []

        for layer_id in range(num_layers):
            if layer_id not in self._predicted_demand:
                continue
            cached = cache_states.get(layer_id, set())
            urgency = self._urgency(layer_id)

            for eid in range(self.config.num_experts):
                d = self._get_demand(layer_id, eid)
                if d > 0 and eid not in cached:
                    value = d * urgency
                    candidates.append((value, layer_id, eid))

        candidates.sort(reverse=True)
        budget = self.config.max_prefetch_experts
        selected = candidates[:budget]
        self._prefetch_scheduled += len(selected)

        return [(lid, eid, val) for val, lid, eid in selected]

    def advance_step(self) -> None:
        """Advance the global step counter."""
        self._step += 1

    def get_stats(self) -> dict:
        return {
            "step": self._step,
            "evictions_pred_guided": self._evictions_pred,
            "evictions_lru_fallback": self._evictions_lru,
            "prefetch_scheduled": self._prefetch_scheduled,
        }

    def _urgency(self, layer_id: int) -> float:
        adjusted = layer_id + 1
        mode = self.config.urgency_decay
        if mode == "inverse":
            return 1.0 / adjusted
        elif mode == "linear":
            return max(0.01, 1.0 - (adjusted - 1) * 0.03)
        elif mode == "exp":
            return 0.95 ** (adjusted - 1)
        return 1.0 / adjusted
