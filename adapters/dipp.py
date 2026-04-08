"""
DIPP — Draft-Informed Prioritized Preloading
===============================================
Solves P2 (Flat-Priority Prefetch): PCIe bandwidth (~79 experts in 30ms)
cannot satisfy full miss demand (~156 experts). SP-MoE uses FIFO which
may load far-layer experts before near-layer ones that block execution.

Core idea: Multi-dimensional Value function for bandwidth-constrained
           priority scheduling of expert preloading.

    V_ℓ(e) = demand(e) × miss(e) × urgency(ℓ)

    demand  = number of tokens needing this expert
    miss    = 1 if cache miss, 0 if already cached
    urgency = 1/ℓ for layer ℓ (early layers execute first → more urgent)

Schedule: global priority queue sorted by V, greedily fill until
          bandwidth budget is exhausted.

Progressive preloading: don't wait for all K draft tokens — start
    prefetching after each draft token is generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DIPPConfig:
    """Configuration for DIPP preloader."""
    # Maximum experts to prefetch per draft round (PCIe bandwidth constraint)
    max_prefetch_experts: int = 79
    # Enable progressive preloading (prefetch after each draft token)
    enable_progressive: bool = True
    # Urgency decay function: "inverse" (1/ℓ), "linear", "exp"
    urgency_decay: str = "inverse"
    # Expert size in bytes (for bandwidth accounting)
    expert_size_bytes: int = 9_437_184  # ~9.44 MB
    # Layer numbering starts at 1 (for urgency computation)
    layer_offset: int = 1


@dataclass
class DIPPStats:
    """Running statistics for DIPP."""
    total_schedules: int = 0
    total_experts_scheduled: int = 0
    total_experts_over_budget: int = 0  # demand that was cut by budget
    total_draft_tokens: int = 0
    progressive_rounds: int = 0

    @property
    def avg_scheduled_per_round(self) -> float:
        return self.total_experts_scheduled / max(1, self.total_schedules)

    @property
    def avg_budget_utilization(self) -> float:
        """Fraction of budget actually used."""
        if self.total_schedules == 0:
            return 0.0
        total_capacity = self.total_schedules * 79  # approx
        return self.total_experts_scheduled / max(1, total_capacity)


class DraftInformedPrioritizedPreloader:
    """
    K×L deep lookahead prioritized expert preloading.

    Architecture:
      Draft model generates K tokens, each producing router predictions
      for all L MoE layers. DIPP collects these predictions, computes
      a Value for each predicted miss expert, and schedules PCIe transfers
      in Value-descending order up to the bandwidth budget.

    Usage:
        dipp = DraftInformedPrioritizedPreloader(config)

        # Option 1: Full schedule (after all K drafts)
        schedule = dipp.compute_schedule(all_predictions, cache_state)

        # Option 2: Progressive (after each draft token)
        for k in range(K):
            new_prefetches = dipp.on_draft_token(k, router_preds, cache_state)
    """

    def __init__(self, config: Optional[DIPPConfig] = None):
        self.config = config or DIPPConfig()
        self.stats = DIPPStats()
        # Accumulator for progressive mode: layer → {token_pos → [expert_ids]}
        self._accumulated_predictions: dict[int, dict[int, list[int]]] = {}
        # Experts already scheduled in this draft round
        self._already_scheduled: set[tuple[int, int]] = set()

    def compute_value(
        self,
        layer_id: int,
        expert_id: int,
        predictions: dict[int, dict[int, list[int]]],
        cache_state: dict[int, set[int]],
    ) -> float:
        """
        Compute the prefetch Value for a single expert.

        Args:
            layer_id: MoE layer index.
            expert_id: Expert index.
            predictions: {layer_id: {token_pos: [expert_ids]}}.
            cache_state: {layer_id: set(cached_expert_ids)}.

        Returns:
            Value score. 0.0 if already cached or not needed.
        """
        # Miss check
        if expert_id in cache_state.get(layer_id, set()):
            return 0.0

        # Demand: how many tokens need this expert
        layer_preds = predictions.get(layer_id, {})
        demand = sum(
            1 for tok_experts in layer_preds.values()
            if expert_id in tok_experts
        )
        if demand == 0:
            return 0.0

        # Urgency
        urgency = self._urgency(layer_id)

        return demand * urgency

    def compute_schedule(
        self,
        predictions: dict[int, dict[int, list[int]]],
        cache_state: dict[int, set[int]],
    ) -> list[tuple[int, int, float]]:
        """
        Compute full prefetch schedule given K×L predictions.

        Args:
            predictions: {layer_id: {token_pos: [expert_ids]}}.
            cache_state: {layer_id: set(cached_expert_ids)}.

        Returns:
            List of (layer_id, expert_id, value) sorted by value descending,
            truncated to bandwidth budget.
        """
        self.stats.total_schedules += 1

        # Collect all miss experts and their values
        candidates: list[tuple[float, int, int]] = []  # (value, layer, expert)

        for layer_id, token_preds in predictions.items():
            # Collect unique expert IDs across all tokens for this layer
            unique_experts: set[int] = set()
            for expert_list in token_preds.values():
                unique_experts.update(expert_list)

            cached = cache_state.get(layer_id, set())
            for eid in unique_experts:
                if eid in cached:
                    continue
                value = self.compute_value(layer_id, eid, predictions, cache_state)
                if value > 0:
                    candidates.append((value, layer_id, eid))

        # Sort by value descending
        candidates.sort(reverse=True)

        # Apply bandwidth budget
        budget = self.config.max_prefetch_experts
        selected = candidates[:budget]
        over_budget = len(candidates) - len(selected)

        self.stats.total_experts_scheduled += len(selected)
        self.stats.total_experts_over_budget += max(0, over_budget)

        return [(layer_id, eid, val) for val, layer_id, eid in selected]

    def on_draft_token(
        self,
        draft_token_idx: int,
        router_predictions: dict[int, list[int]],
        cache_state: dict[int, set[int]],
    ) -> list[tuple[int, int, float]]:
        """
        Progressive preloading: process one draft token's predictions.

        Args:
            draft_token_idx: Index of this draft token (0-based).
            router_predictions: {layer_id: [expert_ids]} for this token.
            cache_state: Current cache state.

        Returns:
            Newly scheduled (layer_id, expert_id, value) entries for this
            incremental round.
        """
        self.stats.total_draft_tokens += 1

        # Accumulate predictions
        for layer_id, expert_ids in router_predictions.items():
            if layer_id not in self._accumulated_predictions:
                self._accumulated_predictions[layer_id] = {}
            self._accumulated_predictions[layer_id][draft_token_idx] = expert_ids

        # Recompute full schedule with accumulated predictions
        full_schedule = self.compute_schedule(
            self._accumulated_predictions, cache_state
        )

        # Filter out already-scheduled experts
        new_entries = []
        for layer_id, eid, val in full_schedule:
            if (layer_id, eid) not in self._already_scheduled:
                self._already_scheduled.add((layer_id, eid))
                new_entries.append((layer_id, eid, val))

        self.stats.progressive_rounds += 1
        return new_entries

    def reset_round(self) -> None:
        """Reset state for a new draft round."""
        self._accumulated_predictions.clear()
        self._already_scheduled.clear()

    def get_stats(self) -> DIPPStats:
        return self.stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _urgency(self, layer_id: int) -> float:
        """Compute urgency weight for a layer."""
        adjusted = layer_id + self.config.layer_offset
        mode = self.config.urgency_decay
        if mode == "inverse":
            return 1.0 / adjusted
        elif mode == "linear":
            return max(0.01, 1.0 - (adjusted - 1) * 0.03)
        elif mode == "exp":
            return 0.95 ** (adjusted - 1)
        return 1.0 / adjusted
