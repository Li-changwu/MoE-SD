from dataclasses import dataclass
from typing import Any, Dict, List

from controllers.interface import RuntimeState


@dataclass
class PrefetchPolicyConfig:
    hard_threshold: float = 0.60
    soft_threshold: float = 0.35
    io_cost_weight: float = 1.0


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
