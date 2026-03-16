from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    UNKNOWN = "unknown"


@dataclass
class RequestState:
    request_id: str
    prompt_len: int
    output_len: int
    request_rate: float
    phase: Phase


@dataclass
class RuntimeState:
    request: RequestState
    step_id: int
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float
    kv_cache_mb: float
    acceptance_rate: float


def build_decision_trace(
    *,
    request_id: str,
    phase: str,
    step_id: int,
    component: str,
    decision: Dict[str, Any],
    reason: str,
    policy_name: str,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "phase": phase,
        "step_id": step_id,
        "component": component,
        "decision": decision,
        "reason": reason,
        "policy_name": policy_name,
    }


class SchedulerController:
    """Frozen controller interface for all scheduling policies."""

    def on_request_arrival(self, req_meta: Dict[str, Any]) -> None:
        pass

    def on_prefill_begin(self, state: RuntimeState) -> None:
        pass

    def on_prefill_end(self, state: RuntimeState) -> None:
        pass

    def on_decode_step(self, step_meta: Dict[str, Any]) -> None:
        pass

    def decide_speculation_k(self, state: RuntimeState) -> Dict[str, Any]:
        raise NotImplementedError

    def decide_memory_partition(self, state: RuntimeState) -> Dict[str, Any]:
        raise NotImplementedError

    def decide_prefetch(self, expert_candidates: List[Dict[str, Any]], state: RuntimeState) -> Dict[str, Any]:
        raise NotImplementedError


class NoOpController(SchedulerController):
    """Safe default controller that preserves native behavior."""

    def decide_speculation_k(self, state: RuntimeState) -> Dict[str, Any]:
        return {"k": None, "apply": False, "reason": "controller_disabled"}

    def decide_memory_partition(self, state: RuntimeState) -> Dict[str, Any]:
        return {
            "expert_budget_mb": None,
            "speculative_budget_mb": None,
            "kv_reserve_mb": None,
            "apply": False,
            "reason": "controller_disabled",
        }

    def decide_prefetch(self, expert_candidates: List[Dict[str, Any]], state: RuntimeState) -> Dict[str, Any]:
        return {
            "hard": [],
            "soft": [],
            "defer": [c.get("expert_id") for c in expert_candidates],
            "apply": False,
            "reason": "controller_disabled",
        }
