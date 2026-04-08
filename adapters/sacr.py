"""
SACR — Speculation-Aware Cache Replacement
============================================
Solves P1 (Cache Pollution): rejected-token experts pollute cache,
occupying ~35% of space with low reuse probability.

Core idea: Introduce SD-specific AcceptRatio signal into eviction scoring.

    Score(e) = α · Recency(e) + β · Frequency(e) + γ · AcceptRatio(e)

Evict argmin Score(e). Low AcceptRatio → rejected-token expert → evict first.

When AcceptRatio data is insufficient (< min_observations), SACR degrades
gracefully to weighted LRU+LFU (γ effectively 0).

Performance: Uses flat per-layer arrays instead of dict[(layer,expert)]
to eliminate tuple allocation and dict lookup overhead in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from adapters.accept_reject_tracker import AcceptRejectTracker


@dataclass
class SACRConfig:
    """Configuration for SACR eviction policy."""
    alpha: float = 0.3   # recency weight
    beta: float = 0.2    # frequency weight
    gamma: float = 0.5   # accept_ratio weight (highest — most precise signal)
    # Window size for recency normalization (in steps)
    recency_window: int = 100
    # Fallback weights when AcceptRatio is unreliable
    fallback_alpha: float = 0.6
    fallback_beta: float = 0.4
    # Adaptive γ: scale γ by (1 - hit_rate) so that when cache is abundant
    # (high hit rate), SACR doesn't over-penalize rejected-token experts.
    # Without this, SACR degrades vs LRU at large cache sizes.
    adaptive_gamma: bool = True
    # Minimum γ floor (prevent complete disabling)
    adaptive_gamma_min: float = 0.05


@dataclass
class CacheMeta:
    """Per-expert cache metadata maintained by SACR."""
    last_access_step: int = 0
    access_count: int = 0
    first_access_step: int = 0


class SACREvictionPolicy:
    """
    Speculation-Aware Cache Replacement policy.

    Uses AcceptRatio from AcceptRejectTracker as the primary signal
    to distinguish high-value (accepted-token) experts from low-value
    (rejected-token) experts in eviction decisions.

    Optimized with flat per-layer arrays for O(1) access recording
    with zero allocation overhead.
    """

    def __init__(
        self,
        config: Optional[SACRConfig] = None,
        tracker: Optional[AcceptRejectTracker] = None,
        num_experts: int = 128,
    ):
        self.config = config or SACRConfig()
        self.tracker = tracker
        self.num_experts = num_experts
        self._current_step = 0
        # Per-layer flat arrays: layer_id → list[int] of size num_experts
        # Lazily initialized on first access to each layer
        self._access_count: dict[int, list[int]] = {}
        self._last_access: dict[int, list[int]] = {}
        # Track max access count per layer for normalization
        self._max_access_val: dict[int, int] = {}
        # Per-layer hit rate for adaptive γ (set externally or via update)
        self._layer_hit_rate: dict[int, float] = {}
        # Pre-compute config values used in hot path
        self._inv_recency_window = 1.0 / max(self.config.recency_window, 1)
        self._cfg_alpha = self.config.alpha
        self._cfg_beta = self.config.beta
        self._cfg_gamma = self.config.gamma
        self._cfg_fb_alpha = self.config.fallback_alpha
        self._cfg_fb_beta = self.config.fallback_beta
        self._cfg_adaptive = self.config.adaptive_gamma
        self._cfg_gamma_min = self.config.adaptive_gamma_min

    def _ensure_layer(self, layer_id: int) -> None:
        if layer_id not in self._access_count:
            n = self.num_experts
            self._access_count[layer_id] = [0] * n
            self._last_access[layer_id] = [0] * n
            self._max_access_val[layer_id] = 0

    def record_access(self, layer_id: int, expert_id: int, step: Optional[int] = None) -> None:
        """Record an expert access. O(1) with zero allocation."""
        if step is not None:
            self._current_step = step
        ac = self._access_count.get(layer_id)
        if ac is None:
            self._ensure_layer(layer_id)
            ac = self._access_count[layer_id]
        ac[expert_id] += 1
        self._last_access[layer_id][expert_id] = self._current_step
        c = ac[expert_id]
        if c > self._max_access_val[layer_id]:
            self._max_access_val[layer_id] = c

    def record_access_batch(self, layer_id: int, expert_ids: list[int], step: int) -> None:
        """Record accesses for multiple experts in one call. Avoids per-call overhead."""
        self._current_step = step
        ac = self._access_count.get(layer_id)
        if ac is None:
            self._ensure_layer(layer_id)
            ac = self._access_count[layer_id]
        la = self._last_access[layer_id]
        mx = self._max_access_val[layer_id]
        for eid in expert_ids:
            ac[eid] += 1
            la[eid] = step
            if ac[eid] > mx:
                mx = ac[eid]
        self._max_access_val[layer_id] = mx

    def score(self, layer_id: int, expert_id: int) -> float:
        """
        Compute retention score for an expert. Higher = keep longer.
        Returns 0.0 for experts with no metadata (should be evicted first).
        """
        ac = self._access_count.get(layer_id)
        if ac is None or ac[expert_id] == 0:
            return 0.0

        # Recency: normalized [0, 1], higher = more recent
        age = self._current_step - self._last_access[layer_id][expert_id]
        recency = max(0.0, 1.0 - age * self._inv_recency_window)

        # Frequency: normalized [0, 1], higher = more frequent
        max_count = self._max_access_val.get(layer_id, 1)
        if max_count < 1:
            max_count = 1
        frequency = ac[expert_id] / max_count
        if frequency > 1.0:
            frequency = 1.0

        # AcceptRatio: [0, 1] from tracker
        tracker = self.tracker
        has_reliable_ar = (
            tracker is not None
            and tracker.is_reliable(layer_id, expert_id)
        )

        if has_reliable_ar:
            accept_ratio = tracker.get_accept_ratio(layer_id, expert_id)
            alpha = self._cfg_alpha
            beta = self._cfg_beta
            gamma = self._cfg_gamma
            if self._cfg_adaptive:
                hr = self._layer_hit_rate.get(layer_id, 0.0)
                if hr > 0.6:
                    gamma = self._cfg_gamma_min
                else:
                    pressure = 1.0 - hr / 0.6
                    if pressure < 0.0:
                        pressure = 0.0
                    gamma = gamma * pressure
                    if gamma < self._cfg_gamma_min:
                        gamma = self._cfg_gamma_min
                gamma_deficit = self._cfg_gamma - gamma
                alpha = self._cfg_alpha + gamma_deficit * 0.6
                beta = self._cfg_beta + gamma_deficit * 0.4
        else:
            accept_ratio = 0.0
            alpha = self._cfg_fb_alpha
            beta = self._cfg_fb_beta
            gamma = 0.0

        return alpha * recency + beta * frequency + gamma * accept_ratio

    def select_victim(
        self, layer_id: int, candidates: list[int]
    ) -> int:
        """Select the eviction victim from candidates: argmin score(e)."""
        if not candidates:
            raise ValueError("Cannot select victim from empty candidates list")

        best_victim = candidates[0]
        best_score = self.score(layer_id, best_victim)

        for i in range(1, len(candidates)):
            eid = candidates[i]
            s = self.score(layer_id, eid)
            if s < best_score:
                best_score = s
                best_victim = eid

        return best_victim

    def remove_expert(self, layer_id: int, expert_id: int) -> None:
        """Reset metadata for an evicted expert."""
        ac = self._access_count.get(layer_id)
        if ac is not None:
            ac[expert_id] = 0
            self._last_access[layer_id][expert_id] = 0

    def advance_step(self) -> int:
        """Advance the internal step counter. Returns new step."""
        self._current_step += 1
        return self._current_step

    @property
    def current_step(self) -> int:
        return self._current_step

    def update_hit_rate(self, layer_id: int, hit_rate: float) -> None:
        """Update per-layer hit rate for adaptive γ scaling."""
        self._layer_hit_rate[layer_id] = hit_rate

    def get_meta(self, layer_id: int, expert_id: int) -> Optional[CacheMeta]:
        ac = self._access_count.get(layer_id)
        if ac is None or ac[expert_id] == 0:
            return None
        return CacheMeta(
            last_access_step=self._last_access[layer_id][expert_id],
            access_count=ac[expert_id],
        )
