from dataclasses import dataclass
from typing import Any, Dict, List

from controllers.interface import RuntimeState


@dataclass
class PrefetchPolicyConfig:
    hard_threshold: float = 0.60
    soft_threshold: float = 0.35
    io_cost_weight: float = 1.0
    shared_expert_bonus: float = 0.20
    reuse_bonus: float = 0.15
    depth_penalty: float = 0.12
    isolation_penalty: float = 0.18
    waste_penalty: float = 1.0
    expert_size_bytes: int = 9 * 1024 * 1024


class AcceptanceAwarePrefetchPolicy:
    """Heuristic prefetch v1.

    score = p(expert_needed) * p(token_accepted) * benefit - cost
    """

    def __init__(self, config: PrefetchPolicyConfig | None = None):
        self.cfg = config or PrefetchPolicyConfig()

    def decide(self, expert_candidates: List[Dict[str, Any]], state: RuntimeState) -> Dict[str, Any]:
        hard: List[int] = []
        soft: List[int] = []
        defer: List[int] = []

        for c in expert_candidates:
            eid = int(c.get("expert_id", -1))
            p_need = float(c.get("p_expert_needed", 0.0))
            p_accept = float(c.get("p_token_accepted", state.acceptance_rate))
            benefit = float(c.get("benefit", 1.0))
            cost = float(c.get("cost", 0.0)) * self.cfg.io_cost_weight

            score = p_need * p_accept * benefit - cost
            if score >= self.cfg.hard_threshold:
                hard.append(eid)
            elif score >= self.cfg.soft_threshold:
                soft.append(eid)
            else:
                defer.append(eid)

        reason = "balanced"
        if state.acceptance_rate < 0.35:
            reason = "low_acceptance_high_io_cost"
        elif hard and not defer:
            reason = "high_confidence_prefetch"

        return {
            "hard": hard,
            "soft": soft,
            "defer": defer,
            "apply": True,
            "reason": reason,
            "wasted_prefetched_bytes": 0,
            "prefetch_hit": None,
            "prefetch_miss": None,
        }


class FrontierAwarePrefetchPolicy(AcceptanceAwarePrefetchPolicy):
    """Speculative-frontier-aware prefetch v2.

    v2 prefers experts shared by multiple speculative branches and defers deep,
    isolated experts whose prefetches are likely to become rejection waste.
    """

    def _candidate_acceptance(self, candidate: Dict[str, Any], state: RuntimeState) -> float:
        p_accept = float(candidate.get("p_token_accepted", state.acceptance_rate))
        branch_alive = candidate.get("p_branch_alive")
        if branch_alive is not None:
            p_accept *= float(branch_alive)
        return max(0.0, min(1.0, p_accept))

    def _shared_fraction(self, candidate: Dict[str, Any]) -> float:
        if "shared_fraction" in candidate:
            return max(0.0, min(1.0, float(candidate["shared_fraction"])))
        shared_count = max(1.0, float(candidate.get("shared_count", 1.0)))
        frontier_size = max(shared_count, float(candidate.get("frontier_size", shared_count)))
        return max(0.0, min(1.0, shared_count / frontier_size))

    def _depth_ratio(self, candidate: Dict[str, Any]) -> float:
        depth = max(1.0, float(candidate.get("avg_depth", candidate.get("depth", 1.0))))
        frontier_depth = max(depth, float(candidate.get("frontier_depth", depth)))
        return max(0.0, min(1.0, depth / frontier_depth))

    def _estimate_waste_bytes(
        self,
        candidate: Dict[str, Any],
        p_accept: float,
        shared_fraction: float,
        depth_ratio: float,
    ) -> int:
        size_bytes = int(candidate.get("size_bytes", self.cfg.expert_size_bytes))
        waste_prob = (1.0 - p_accept) * (1.0 - shared_fraction) * depth_ratio
        return int(size_bytes * max(0.0, min(1.0, waste_prob)))

    def decide(self, expert_candidates: List[Dict[str, Any]], state: RuntimeState) -> Dict[str, Any]:
        hard: List[int] = []
        soft: List[int] = []
        defer: List[int] = []
        wasted_prefetched_bytes = 0
        prefetch_hit = 0
        prefetch_miss = 0

        for candidate in expert_candidates:
            expert_id = int(candidate.get("expert_id", -1))
            p_need = float(candidate.get("p_expert_needed", 0.0))
            benefit = float(candidate.get("benefit", 1.0))
            cost = float(candidate.get("cost", 0.0)) * self.cfg.io_cost_weight
            expert_reuse = max(0.0, min(1.0, float(candidate.get("expert_reuse", 0.0))))
            p_accept = self._candidate_acceptance(candidate, state)
            shared_fraction = self._shared_fraction(candidate)
            depth_ratio = self._depth_ratio(candidate)

            base_score = p_need * p_accept * benefit - cost
            shared_bonus = self.cfg.shared_expert_bonus * shared_fraction
            reuse_bonus = self.cfg.reuse_bonus * expert_reuse
            depth_penalty = self.cfg.depth_penalty * depth_ratio
            isolation_penalty = self.cfg.isolation_penalty * (1.0 - shared_fraction) * depth_ratio
            estimated_waste = self._estimate_waste_bytes(candidate, p_accept, shared_fraction, depth_ratio)
            waste_penalty = self.cfg.waste_penalty * (estimated_waste / max(1, self.cfg.expert_size_bytes))

            score = (
                base_score
                + shared_bonus
                + reuse_bonus
                - depth_penalty
                - isolation_penalty
                - waste_penalty
            )

            if score >= self.cfg.hard_threshold:
                hard.append(expert_id)
                prefetch_hit += 1
                wasted_prefetched_bytes += estimated_waste
            elif score >= self.cfg.soft_threshold:
                soft.append(expert_id)
                prefetch_hit += 1
                wasted_prefetched_bytes += estimated_waste
            else:
                defer.append(expert_id)
                prefetch_miss += 1

        reason_parts: List[str] = []
        if state.acceptance_rate < 0.35:
            reason_parts.append("low_acceptance")
        if hard:
            reason_parts.append("shared_frontier_prefetch")
        if defer:
            reason_parts.append("deep_isolated_deferred")
        if not reason_parts:
            reason_parts.append("balanced")

        return {
            "hard": hard,
            "soft": soft,
            "defer": defer,
            "apply": True,
            "reason": "|".join(reason_parts),
            "wasted_prefetched_bytes": wasted_prefetched_bytes,
            "prefetch_hit": prefetch_hit,
            "prefetch_miss": prefetch_miss,
        }
