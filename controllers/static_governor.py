from dataclasses import dataclass
from typing import Any, Dict, List

from controllers.interface import RuntimeState, SchedulerController


@dataclass
class StaticGovernorConfig:
    default_k: int = 4
    prefill_max_k: int = 2
    decode_max_k: int = 4
    high_pressure_ratio: float = 0.9
    medium_pressure_ratio: float = 0.8
    low_acceptance_threshold: float = 0.35
    medium_acceptance_threshold: float = 0.55
    kv_growth_fast_mb: float = 256.0


class StaticGovernor(SchedulerController):
    """Rule-based v0 governor for speculation K and coarse memory budget."""

    def __init__(self, config: StaticGovernorConfig | None = None):
        self.cfg = config or StaticGovernorConfig()
        self._last_kv_mb: Dict[str, float] = {}

    def _memory_pressure(self, state: RuntimeState) -> str:
        total = max(state.gpu_mem_total_mb, 1.0)
        ratio = state.gpu_mem_used_mb / total
        if ratio >= self.cfg.high_pressure_ratio:
            return "high"
        if ratio >= self.cfg.medium_pressure_ratio:
            return "medium"
        return "low"

    def _acceptance_bucket(self, state: RuntimeState) -> str:
        a = state.acceptance_rate
        if a < self.cfg.low_acceptance_threshold:
            return "low"
        if a < self.cfg.medium_acceptance_threshold:
            return "medium"
        return "high"

    def _kv_growth(self, state: RuntimeState) -> float:
        rid = state.request.request_id
        prev = self._last_kv_mb.get(rid, state.kv_cache_mb)
        growth = state.kv_cache_mb - prev
        self._last_kv_mb[rid] = state.kv_cache_mb
        return growth

    def decide_speculation_k(self, state: RuntimeState) -> Dict[str, Any]:
        pressure = self._memory_pressure(state)
        acceptance = self._acceptance_bucket(state)
        phase = state.request.phase.value

        if phase == "prefill":
            k = min(self.cfg.prefill_max_k, self.cfg.default_k)
        else:
            k = min(self.cfg.decode_max_k, self.cfg.default_k)

        reason: List[str] = [f"phase={phase}"]

        if pressure == "high":
            k = min(k, 1)
            reason.append("high_memory_pressure")
        elif pressure == "medium":
            k = min(k, 2)
            reason.append("medium_memory_pressure")

        if acceptance == "low":
            k = min(k, 1)
            reason.append("low_acceptance")
        elif acceptance == "medium":
            k = min(k, 2)
            reason.append("medium_acceptance")

        return {"k": int(max(k, 1)), "apply": True, "reason": "|".join(reason)}

    def decide_memory_partition(self, state: RuntimeState) -> Dict[str, Any]:
        total = max(state.gpu_mem_total_mb, 1.0)
        pressure = self._memory_pressure(state)
        kv_growth = self._kv_growth(state)

        kv_reserve = 0.55 * total
        expert_budget = 0.25 * total
        speculative_budget = 0.20 * total
        reason: List[str] = [f"pressure={pressure}"]

        if pressure == "medium":
            kv_reserve = 0.60 * total
            speculative_budget = 0.15 * total
            reason.append("shift_to_kv")
        elif pressure == "high":
            kv_reserve = 0.68 * total
            speculative_budget = 0.10 * total
            expert_budget = 0.22 * total
            reason.append("high_pressure_shrink_spec")

        if kv_growth >= self.cfg.kv_growth_fast_mb:
            kv_reserve += 0.05 * total
            speculative_budget -= 0.05 * total
            reason.append("fast_kv_growth")

        if state.acceptance_rate < self.cfg.low_acceptance_threshold:
            speculative_budget -= 0.03 * total
            reason.append("low_acceptance_shrink_spec")

        speculative_budget = max(0.0, speculative_budget)
        expert_budget = max(0.0, total - kv_reserve - speculative_budget)

        return {
            "expert_budget_mb": round(expert_budget, 2),
            "speculative_budget_mb": round(speculative_budget, 2),
            "kv_reserve_mb": round(kv_reserve, 2),
            "apply": True,
            "reason": "|".join(reason),
        }

    def decide_prefetch(self, expert_candidates: List[Dict[str, Any]], state: RuntimeState) -> Dict[str, Any]:
        # v0 keeps prefetch conservative. Real prefetch logic will be in ISSUE-012.
        hard = []
        soft = []
        defer = [c.get("expert_id") for c in expert_candidates]
        return {
            "hard": hard,
            "soft": soft,
            "defer": defer,
            "apply": False,
            "reason": "static_v0_prefetch_disabled",
        }
