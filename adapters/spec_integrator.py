"""Speculative Decoding Integrator for EAGLE-3"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class SpeculationConfig:
    spec_method: str = "eagle3"
    max_spec_tokens: int = 8
    min_spec_tokens: int = 1
    adaptive: bool = True

class SpecIntegrator:
    """Integrates controller speculation decisions with vLLM EAGLE-3."""
    
    def __init__(self, controller, vllm_worker, config=None):
        self.controller = controller
        self.vllm_worker = vllm_worker
        self.config = config or SpeculationConfig()
        self._current_k = None
        self._acceptance_history = []
        
    def decide_and_apply_speculation(self, state) -> Dict[str, Any]:
        try:
            decision = self.controller.decide_speculation_k(state)
            if not decision.get("apply", False):
                return {"applied": False, "k": 0, **decision}
            k = decision.get("k", self.config.max_spec_tokens)
            k = max(self.config.min_spec_tokens, min(k, self.config.max_spec_tokens))
            self._apply_spec_k(k)
            return {"applied": True, "k": k, "method": self.config.spec_method}
        except Exception as e:
            logger.error(f"Speculation decision failed: {e}")
            return {"applied": False, "k": 0, "reason": str(e)}
            
    def _apply_spec_k(self, k: int) -> None:
        if hasattr(self.vllm_worker, 'update_speculative_tokens'):
            self.vllm_worker.update_speculative_tokens(k)
        self._current_k = k
        
    def report_acceptance(self, accepted: int, proposed: int, request_id: str) -> None:
        rate = accepted / proposed if proposed > 0 else 0.0
        self._acceptance_history.append(rate)
        
    @property
    def mean_acceptance(self) -> float:
        return sum(self._acceptance_history) / len(self._acceptance_history) if self._acceptance_history else 0.0
        
    def get_statistics(self) -> Dict[str, Any]:
        return {"current_k": self._current_k, "mean_acceptance": self.mean_acceptance, "method": self.config.spec_method}
