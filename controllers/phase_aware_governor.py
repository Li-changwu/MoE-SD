from dataclasses import dataclass
from typing import Any, Dict, List

from controllers.interface import RuntimeState, SchedulerController


@dataclass
class PhaseAwareGovernorConfig:
    prefill_max_k: int = 1
    decode_max_k: int = 4
    prefill_memory_bias_kv: float = 0.70
    decode_memory_bias_kv: float = 0.55
    high_pressure_ratio: float = 0.9
    medium_pressure_ratio: float = 0.8
    low_acceptance_threshold: float = 0.35


class PhaseAwareGovernor(SchedulerController):
    """Phase-aware v1 governor: prefill favors TTFT, decode favors TPOT/throughput."""

    def __init__(self, config: PhaseAwareGovernorConfig | None = None):
        self.cfg = config or PhaseAwareGovernorConfig()

    def _pressure(self, s: RuntimeState) -> str:
        ratio = s.gpu_mem_used_mb / max(s.gpu_mem_total_mb, 1.0)
        if ratio >= self.cfg.high_pressure_ratio:
            return "high"
        if ratio >= self.cfg.medium_pressure_ratio:
            return "medium"
        return "low"

    def decide_speculation_k(self, state: RuntimeState) -> Dict[str, Any]:
        phase = state.request.phase.value
        pressure = self._pressure(state)
        if phase == "prefill":
            k = self.cfg.prefill_max_k
            reason = ["prefill_conservative"]
        else:
            k = self.cfg.decode_max_k
            reason = ["decode_aggressive"]

        if pressure == "high":
            k = min(k, 1)
            reason.append("high_pressure")
        elif pressure == "medium":
            k = min(k, 2)
            reason.append("medium_pressure")

        if state.acceptance_rate < self.cfg.low_acceptance_threshold:
            k = min(k, 1)
            reason.append("low_acceptance")

        return {"k": int(max(1, k)), "apply": True, "reason": "|".join(reason), "phase": phase}

    def decide_memory_partition(self, state: RuntimeState) -> Dict[str, Any]:
        phase = state.request.phase.value
        total = max(state.gpu_mem_total_mb, 1.0)

        if phase == "prefill":
            kv_ratio = self.cfg.prefill_memory_bias_kv
            reason = ["prefill_bias_kv"]
        else:
            kv_ratio = self.cfg.decode_memory_bias_kv
            reason = ["decode_balance"]

        pressure = self._pressure(state)
        if pressure == "high":
            kv_ratio += 0.08
            reason.append("high_pressure_more_kv")
        elif pressure == "medium":
            kv_ratio += 0.04
            reason.append("medium_pressure_more_kv")

        kv_ratio = min(max(kv_ratio, 0.4), 0.85)
        kv_reserve = kv_ratio * total
        speculative_budget = max(0.08 * total, 0.18 * total if phase == "decode" else 0.10 * total)
        expert_budget = max(0.0, total - kv_reserve - speculative_budget)

        return {
            "expert_budget_mb": round(expert_budget, 2),
            "speculative_budget_mb": round(speculative_budget, 2),
            "kv_reserve_mb": round(kv_reserve, 2),
            "apply": True,
            "reason": "|".join(reason),
            "phase": phase,
        }

    def decide_prefetch(self, expert_candidates: List[Dict[str, Any]], state: RuntimeState) -> Dict[str, Any]:
        # v1 still keeps prefetch disabled; ISSUE-012 introduces real policy.
        return {
            "hard": [],
            "soft": [],
            "defer": [c.get("expert_id") for c in expert_candidates],
            "apply": False,
            "reason": "phase_aware_v1_prefetch_disabled",
            "phase": state.request.phase.value,
        }
