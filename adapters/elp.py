"""
ELP — Expert Lifecycle Partitioning
=====================================
Solves P4 (Cascade Eviction): SD burst access (W_sd=27 > S=17) triggers
cascade eviction propagating ~8 steps, adding ~443ms latency.

Core idea: Partition cache into Pin Zone (persistent experts, never evicted
by burst overflow) and Flex Zone (transient experts, fast turnover).
Cascade eviction is confined to Flex Zone → propagation drops from ~8 to ~2 steps.

Inspired by OS (Linux active/inactive page lists) and DB (hot/cold buffer
pool partitioning), but first applied to MoE expert caches.

Performance: Uses flat per-layer arrays instead of dict[(layer,expert)]
to eliminate tuple allocation and dict lookup overhead in the hot path.
Promotion checks are deferred to periodic rebalance to avoid per-access overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ELPConfig:
    """Configuration for Expert Lifecycle Partitioning."""
    # Fraction of total cache slots allocated to Pin Zone
    pin_ratio: float = 0.7
    # Access count threshold for promotion to Pin Zone
    promotion_threshold: int = 5
    # Steps without access before demotion from Pin Zone
    demotion_window: int = 50
    # How often to run rebalance (every N steps)
    rebalance_interval: int = 10
    # Hard limits on pin_ratio
    min_pin_ratio: float = 0.3
    max_pin_ratio: float = 0.8


@dataclass
class ExpertLifecycleStats:
    """Per-expert lifecycle tracking (used for compatibility/reporting)."""
    access_count: int = 0
    last_access_step: int = 0
    zone: str = "flex"  # "pin" | "flex"
    promoted_at_step: int = 0


@dataclass
class PartitionStats:
    """Snapshot of partition state for a layer."""
    pin_count: int = 0
    pin_capacity: int = 0
    flex_count: int = 0
    flex_capacity: int = 0
    promotions: int = 0
    demotions: int = 0


class ExpertLifecyclePartition:
    """
    Pin/Flex zone partitioned expert cache management.

    Pin Zone: persistent experts that are accessed nearly every step.
              These are NEVER evicted by Flex Zone overflow.
    Flex Zone: transient experts with sporadic access.
               Eviction (by SACR or LRU) only occurs here.

    Optimized: Uses per-layer arrays for access tracking.
    Promotion checks are batched in rebalance() instead of per-access.
    """

    def __init__(
        self,
        config: Optional[ELPConfig] = None,
        total_slots: int = 17,
        num_experts: int = 128,
    ):
        self.config = config or ELPConfig()
        self.total_slots = total_slots
        self.num_experts = num_experts

        # Per-layer flat arrays for access tracking
        self._access_count: dict[int, list[int]] = {}
        self._last_access: dict[int, list[int]] = {}
        # Per-layer zone membership: True = pin, False = flex
        self._is_pinned: dict[int, list[bool]] = {}
        # Per-layer zone sets (for fast candidate listing)
        self._pin_zone: dict[int, set[int]] = {}
        self._flex_zone: dict[int, set[int]] = {}

        # Counters for stats
        self._promotions: dict[int, int] = {}
        self._demotions: dict[int, int] = {}
        self._current_step = 0

    @property
    def pin_capacity(self) -> int:
        """Maximum number of experts in Pin Zone."""
        return max(1, int(self.total_slots * self.config.pin_ratio))

    @property
    def flex_capacity(self) -> int:
        """Maximum number of experts in Flex Zone."""
        return max(1, self.total_slots - self.pin_capacity)

    def _ensure_layer(self, layer_id: int) -> None:
        if layer_id not in self._access_count:
            n = self.num_experts
            self._access_count[layer_id] = [0] * n
            self._last_access[layer_id] = [0] * n
            self._is_pinned[layer_id] = [False] * n
            self._pin_zone[layer_id] = set()
            self._flex_zone[layer_id] = set()

    def access(
        self, layer_id: int, expert_id: int, step: Optional[int] = None
    ) -> None:
        """Record an expert access. O(1) with zero allocation.
        Promotion is deferred to rebalance()."""
        if step is not None:
            self._current_step = step
        ac = self._access_count.get(layer_id)
        if ac is None:
            self._ensure_layer(layer_id)
            ac = self._access_count[layer_id]
        ac[expert_id] += 1
        self._last_access[layer_id][expert_id] = self._current_step
        # Register in flex zone if not tracked yet (first access)
        if not self._is_pinned[layer_id][expert_id]:
            self._flex_zone[layer_id].add(expert_id)

    def access_batch(self, layer_id: int, expert_ids: list[int], step: int) -> None:
        """Record accesses for multiple experts. Avoids per-call overhead."""
        self._current_step = step
        ac = self._access_count.get(layer_id)
        if ac is None:
            self._ensure_layer(layer_id)
            ac = self._access_count[layer_id]
        la = self._last_access[layer_id]
        ip = self._is_pinned[layer_id]
        fz = self._flex_zone[layer_id]
        for eid in expert_ids:
            ac[eid] += 1
            la[eid] = step
            if not ip[eid]:
                fz.add(eid)

    def classify(self, layer_id: int, expert_id: int) -> str:
        """Return 'pin', 'flex', or 'uncached'."""
        ac = self._access_count.get(layer_id)
        if ac is None or ac[expert_id] == 0:
            return "uncached"
        if self._is_pinned[layer_id][expert_id]:
            return "pin"
        return "flex"

    def is_pinned(self, layer_id: int, expert_id: int) -> bool:
        ip = self._is_pinned.get(layer_id)
        return ip is not None and ip[expert_id]

    def get_flex_candidates(self, layer_id: int) -> list[int]:
        """Return all expert IDs in Flex Zone for this layer (eviction candidates)."""
        fz = self._flex_zone.get(layer_id)
        if fz is None:
            return []
        return list(fz)

    def get_pin_set(self, layer_id: int) -> set[int]:
        """Return set of pinned expert IDs for this layer."""
        ps = self._pin_zone.get(layer_id)
        if ps is None:
            return set()
        return set(ps)

    def remove_expert(self, layer_id: int, expert_id: int) -> None:
        """Remove an expert from partition tracking (after eviction)."""
        ac = self._access_count.get(layer_id)
        if ac is not None:
            ac[expert_id] = 0
            self._last_access[layer_id][expert_id] = 0
            self._is_pinned[layer_id][expert_id] = False
        pz = self._pin_zone.get(layer_id)
        if pz is not None:
            pz.discard(expert_id)
        fz = self._flex_zone.get(layer_id)
        if fz is not None:
            fz.discard(expert_id)

    def rebalance(self, layer_id: int) -> None:
        """
        Periodic rebalance: promote frequent flex experts, demote stale pin experts.
        This is now the ONLY place promotion and demotion happen (deferred from access).
        """
        self._ensure_layer(layer_id)
        pin_set = self._pin_zone[layer_id]
        flex_set = self._flex_zone[layer_id]
        ac = self._access_count[layer_id]
        la = self._last_access[layer_id]
        ip = self._is_pinned[layer_id]
        threshold = self.config.promotion_threshold
        demotion_window = self.config.demotion_window
        cur_step = self._current_step
        pin_cap = self.pin_capacity

        # Phase 1: Demote stale Pin experts
        to_demote = []
        for eid in list(pin_set):
            if ac[eid] == 0:
                to_demote.append(eid)
                continue
            if cur_step - la[eid] > demotion_window:
                to_demote.append(eid)
        for eid in to_demote:
            ip[eid] = False
            pin_set.discard(eid)
            flex_set.add(eid)
            self._demotions[layer_id] = self._demotions.get(layer_id, 0) + 1

        # Phase 2: Promote frequent Flex experts (if room in pin zone)
        if len(pin_set) < pin_cap:
            # Find flex experts above promotion threshold AND recently active
            promotable = []
            for eid in list(flex_set):
                if ac[eid] >= threshold and cur_step - la[eid] <= demotion_window:
                    promotable.append((ac[eid], eid))
            promotable.sort(reverse=True)
            for _, eid in promotable:
                if len(pin_set) >= pin_cap:
                    break
                ip[eid] = True
                flex_set.discard(eid)
                pin_set.add(eid)
                self._promotions[layer_id] = self._promotions.get(layer_id, 0) + 1
        elif len(pin_set) == pin_cap:
            # Pin zone full: try to swap in stronger flex experts
            # Find weakest pinned expert
            weakest_eid = -1
            weakest_count = float("inf")
            for eid in pin_set:
                if ac[eid] < weakest_count:
                    weakest_count = ac[eid]
                    weakest_eid = eid
            # Find strongest flex expert above threshold and recently active
            for eid in flex_set:
                if ac[eid] >= threshold and ac[eid] > weakest_count and cur_step - la[eid] <= demotion_window:
                    # Swap: demote weakest, promote this one
                    ip[weakest_eid] = False
                    pin_set.discard(weakest_eid)
                    flex_set.add(weakest_eid)
                    self._demotions[layer_id] = self._demotions.get(layer_id, 0) + 1

                    ip[eid] = True
                    flex_set.discard(eid)
                    pin_set.add(eid)
                    self._promotions[layer_id] = self._promotions.get(layer_id, 0) + 1
                    break

    def get_partition_stats(self, layer_id: int) -> PartitionStats:
        self._ensure_layer(layer_id)
        return PartitionStats(
            pin_count=len(self._pin_zone[layer_id]),
            pin_capacity=self.pin_capacity,
            flex_count=len(self._flex_zone[layer_id]),
            flex_capacity=self.flex_capacity,
            promotions=self._promotions.get(layer_id, 0),
            demotions=self._demotions.get(layer_id, 0),
        )

    def advance_step(self) -> int:
        self._current_step += 1
        return self._current_step
