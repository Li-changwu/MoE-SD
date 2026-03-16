from dataclasses import dataclass
from typing import Any, Dict

from controllers.interface import RuntimeState


@dataclass
class MemoryPartitionConfig:
    update_interval_steps: int = 8
    pressure_medium_ratio: float = 0.8
    pressure_high_ratio: float = 0.9
    smooth_alpha: float = 0.5


class DynamicMemoryPartitionController:
    """Rule-based dynamic budget controller (v2)."""

    def __init__(self, config: MemoryPartitionConfig | None = None):
        self.cfg = config or MemoryPartitionConfig()
        self._last: Dict[str, float] = {
            "expert_budget_mb": 0.0,
            "speculative_budget_mb": 0.0,
            "kv_reserve_mb": 0.0,
        }

    def _pressure(self, state: RuntimeState) -> str:
        ratio = state.gpu_mem_used_mb / max(state.gpu_mem_total_mb, 1.0)
        if ratio >= self.cfg.pressure_high_ratio:
            return "high"
        if ratio >= self.cfg.pressure_medium_ratio:
            return "medium"
        return "low"

    def _smooth(self, k: str, target: float) -> float:
        prev = self._last.get(k, target)
        out = self.cfg.smooth_alpha * target + (1 - self.cfg.smooth_alpha) * prev
        self._last[k] = out
        return out

    def decide(self, state: RuntimeState, step_id: int) -> Dict[str, Any]:
        total = max(state.gpu_mem_total_mb, 1.0)
        pressure = self._pressure(state)

        expert_ratio = 0.28
        spec_ratio = 0.18
        kv_ratio = 0.54
        reason = [f"pressure={pressure}"]

        if pressure == "medium":
            kv_ratio = 0.60
            spec_ratio = 0.14
            expert_ratio = 0.26
            reason.append("medium_pressure_shift")
        elif pressure == "high":
            kv_ratio = 0.68
            spec_ratio = 0.10
            expert_ratio = 0.22
            reason.append("high_pressure_shift")

        if state.acceptance_rate < 0.35:
            spec_ratio -= 0.03
            kv_ratio += 0.02
            reason.append("low_acceptance_shrink_spec")

        norm = max(expert_ratio + spec_ratio + kv_ratio, 1e-6)
        expert_ratio, spec_ratio, kv_ratio = expert_ratio / norm, spec_ratio / norm, kv_ratio / norm

        expert_budget = self._smooth("expert_budget_mb", expert_ratio * total)
        speculative_budget = self._smooth("speculative_budget_mb", spec_ratio * total)
        kv_reserve = self._smooth("kv_reserve_mb", kv_ratio * total)

        apply = (step_id % self.cfg.update_interval_steps) == 0

        return {
            "expert_budget_mb": round(expert_budget, 2),
            "speculative_budget_mb": round(speculative_budget, 2),
            "kv_reserve_mb": round(kv_reserve, 2),
            "apply": apply,
            "reason": "|".join(reason),
        }
