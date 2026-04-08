"""
Accept/Reject Tracker — Shared Infrastructure for SACR and ELP
================================================================
Tracks per-expert accept/reject attribution from SD verify phases.

After each SD verify step, the tracker receives:
  1. Which tokens were accepted vs rejected
  2. Which experts each token activated (per layer)

From this, it computes and maintains a per-expert AcceptRatio (EMA-smoothed),
which is the primary signal for SACR's cache replacement decisions and
supports ELP's persistent vs transient classification.

Thread Safety: per-layer locking for concurrent verify callbacks.
Memory: O(L × E_active) ≈ O(48 × 128) = ~6K entries, negligible.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AcceptRejectTrackerConfig:
    """Configuration for the AcceptRejectTracker."""
    # EMA decay factor for AcceptRatio smoothing (0 = no smoothing, 1 = no update)
    ema_alpha: float = 0.15
    # Minimum total accesses before AcceptRatio is considered reliable
    min_observations: int = 3


@dataclass
class ExpertAccessStats:
    """Per-expert accept/reject statistics."""
    accepted_count: int = 0
    rejected_count: int = 0
    total_count: int = 0
    ema_accept_ratio: float = 0.5  # neutral initial
    last_access_step: int = 0

    @property
    def raw_accept_ratio(self) -> float:
        if self.total_count == 0:
            return 0.5
        return self.accepted_count / self.total_count

    @property
    def is_reliable(self) -> bool:
        """Whether we have enough observations for a meaningful AcceptRatio."""
        return self.total_count >= 3


class AcceptRejectTracker:
    """
    Tracks per-expert accept/reject attribution from SD verify phases.

    Usage:
        tracker = AcceptRejectTracker()

        # After each SD verify step:
        tracker.record_verify_result(
            layer_id=5,
            token_expert_map={0: [3, 7], 1: [3, 12], 2: [5, 8], 3: [9, 10]},
            accepted_mask=[True, True, False, False],
            step_id=42,
        )

        # Query:
        ratio = tracker.get_accept_ratio(layer_id=5, expert_id=3)  # high (both tokens accepted)
        ratio = tracker.get_accept_ratio(layer_id=5, expert_id=9)  # low (token rejected)
    """

    def __init__(self, config: Optional[AcceptRejectTrackerConfig] = None):
        self.config = config or AcceptRejectTrackerConfig()
        # (layer_id, expert_id) → ExpertAccessStats
        self._stats: dict[tuple[int, int], ExpertAccessStats] = {}
        # Per-layer locks for thread safety
        self._locks: dict[int, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._global_step = 0

    def _get_lock(self, layer_id: int) -> threading.Lock:
        if layer_id not in self._locks:
            with self._global_lock:
                if layer_id not in self._locks:
                    self._locks[layer_id] = threading.Lock()
        return self._locks[layer_id]

    def record_verify_result(
        self,
        layer_id: int,
        token_expert_map: dict[int, list[int]],
        accepted_mask: list[bool],
        step_id: Optional[int] = None,
    ) -> None:
        """
        Record expert access attribution from one SD verify step for one layer.

        Args:
            layer_id: The MoE layer index.
            token_expert_map: {token_position: [expert_ids]} — which experts
                each token activated in this layer during verify.
            accepted_mask: Per-token accept/reject result from SD verify.
                Must align with token_expert_map keys (sorted by position).
            step_id: Optional global step counter for recency tracking.
        """
        if step_id is not None:
            self._global_step = max(self._global_step, step_id)
        current_step = self._global_step

        lock = self._get_lock(layer_id)
        with lock:
            # Map token positions to their accept/reject status
            sorted_positions = sorted(token_expert_map.keys())
            for i, pos in enumerate(sorted_positions):
                if i >= len(accepted_mask):
                    break
                is_accepted = accepted_mask[i]
                expert_ids = token_expert_map[pos]

                for eid in expert_ids:
                    key = (layer_id, eid)
                    if key not in self._stats:
                        self._stats[key] = ExpertAccessStats()
                    stats = self._stats[key]

                    stats.total_count += 1
                    if is_accepted:
                        stats.accepted_count += 1
                    else:
                        stats.rejected_count += 1
                    stats.last_access_step = current_step

                    # EMA update
                    instant_ratio = 1.0 if is_accepted else 0.0
                    alpha = self.config.ema_alpha
                    stats.ema_accept_ratio = (
                        (1 - alpha) * stats.ema_accept_ratio + alpha * instant_ratio
                    )

    def get_accept_ratio(
        self, layer_id: int, expert_id: int, use_ema: bool = True
    ) -> float:
        """
        Get the AcceptRatio for an expert.

        Returns:
            EMA-smoothed AcceptRatio if use_ema=True and enough data,
            otherwise raw ratio. Returns 0.5 (neutral) if no data.
        """
        key = (layer_id, expert_id)
        stats = self._stats.get(key)
        if stats is None:
            return 0.5  # neutral for unseen experts
        if use_ema:
            return stats.ema_accept_ratio
        return stats.raw_accept_ratio

    def get_accept_count(self, layer_id: int, expert_id: int) -> int:
        key = (layer_id, expert_id)
        stats = self._stats.get(key)
        return stats.accepted_count if stats else 0

    def get_total_count(self, layer_id: int, expert_id: int) -> int:
        key = (layer_id, expert_id)
        stats = self._stats.get(key)
        return stats.total_count if stats else 0

    def get_expert_stats(
        self, layer_id: int
    ) -> dict[int, ExpertAccessStats]:
        """Get all tracked experts' stats for a given layer."""
        result = {}
        for (lid, eid), stats in self._stats.items():
            if lid == layer_id:
                result[eid] = stats
        return result

    def get_last_access_step(self, layer_id: int, expert_id: int) -> int:
        key = (layer_id, expert_id)
        stats = self._stats.get(key)
        return stats.last_access_step if stats else 0

    def is_reliable(self, layer_id: int, expert_id: int) -> bool:
        """Whether AcceptRatio has enough observations to be meaningful."""
        key = (layer_id, expert_id)
        stats = self._stats.get(key)
        if stats is None:
            return False
        return stats.total_count >= self.config.min_observations

    @property
    def global_step(self) -> int:
        return self._global_step

    def advance_step(self) -> int:
        """Manually advance the global step counter. Returns new step."""
        self._global_step += 1
        return self._global_step

    def reset(self) -> None:
        """Clear all tracked statistics."""
        with self._global_lock:
            self._stats.clear()
            self._global_step = 0
