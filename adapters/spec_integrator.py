"""
Speculative Decoding Integrator for EAGLE-3
=============================================
Bridges the controller's K-decisions with the vLLM SpecDecodeWorker,
feeding acceptance observations back and coordinating expert-cache
prefetch from draft-model routing predictions.

Architecture::

    Controller ──decide_speculation_k()──> SpecIntegrator
                                              │
                     ┌────────────────────────┤
                     ▼                        ▼
           vLLM SpecDecodeWorker        ExpertPrefetch
           (set num_speculative_tokens)  (prefetch from draft routing)
                     │
                     ▼
              acceptance stats  ──report_acceptance()──> Controller

The integrator tracks per-request and global acceptance with an
exponential moving average and a sliding window, allowing the
controller to react to both short-term drops and long-term trends.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SpeculationConfig:
    """Configuration for the speculation integrator."""
    spec_method: str = "eagle3"
    max_spec_tokens: int = 8
    min_spec_tokens: int = 1
    adaptive: bool = True
    # EMA decay for acceptance rate tracking (0 = no smoothing, 1 = no update)
    ema_alpha: float = 0.1
    # How many recent verifications to keep for windowed stats
    window_size: int = 128
    # Minimum verifications before allowing K adaptation
    warmup_rounds: int = 8
    # Enable expert-cache prefetch from draft routing signals
    enable_draft_prefetch: bool = True


# ---------------------------------------------------------------------------
# Per-request tracking
# ---------------------------------------------------------------------------
@dataclass
class RequestAcceptanceState:
    """Acceptance tracking for a single in-flight request."""
    request_id: str
    accepted_total: int = 0
    proposed_total: int = 0
    rounds: int = 0
    last_k: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def rate(self) -> float:
        return self.accepted_total / self.proposed_total if self.proposed_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Sliding window acceptance tracker
# ---------------------------------------------------------------------------
class AcceptanceTracker:
    """Thread-safe sliding-window + EMA acceptance tracker."""

    def __init__(self, window_size: int = 128, ema_alpha: float = 0.1):
        self._window: Deque[Tuple[int, int]] = deque(maxlen=window_size)
        self._ema_rate: float = 0.5  # neutral start
        self._alpha = ema_alpha
        self._lock = threading.Lock()
        self._total_accepted = 0
        self._total_proposed = 0

    def record(self, accepted: int, proposed: int) -> None:
        with self._lock:
            self._window.append((accepted, proposed))
            self._total_accepted += accepted
            self._total_proposed += proposed
            # EMA update
            inst_rate = accepted / proposed if proposed > 0 else 0.0
            self._ema_rate = (1 - self._alpha) * self._ema_rate + self._alpha * inst_rate

    @property
    def ema_rate(self) -> float:
        return self._ema_rate

    @property
    def window_rate(self) -> float:
        with self._lock:
            total_p = sum(p for _, p in self._window)
            if total_p == 0:
                return 0.0
            return sum(a for a, _ in self._window) / total_p

    @property
    def global_rate(self) -> float:
        return self._total_accepted / self._total_proposed if self._total_proposed > 0 else 0.0

    @property
    def count(self) -> int:
        return len(self._window)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ema_rate": round(self.ema_rate, 4),
            "window_rate": round(self.window_rate, 4),
            "global_rate": round(self.global_rate, 4),
            "total_accepted": self._total_accepted,
            "total_proposed": self._total_proposed,
            "window_samples": self.count,
        }


# ---------------------------------------------------------------------------
# SpecIntegrator
# ---------------------------------------------------------------------------
class SpecIntegrator:
    """
    Integrates controller speculation decisions with vLLM EAGLE-3.

    Lifecycle::

        integrator = SpecIntegrator(controller, vllm_worker, config)
        integrator.attach()       # hook into SpecDecodeWorker
        ...                        # normal vLLM operation
        integrator.detach()       # unhook
    """

    def __init__(
        self,
        controller: Any,
        vllm_worker: Any = None,
        config: Optional[SpeculationConfig] = None,
    ):
        self.controller = controller
        self.vllm_worker = vllm_worker
        self.config = config or SpeculationConfig()

        # --- tracking ---
        self._tracker = AcceptanceTracker(
            window_size=self.config.window_size,
            ema_alpha=self.config.ema_alpha,
        )
        self._request_states: Dict[str, RequestAcceptanceState] = {}
        self._current_k: Optional[int] = None
        self._k_history: Deque[int] = deque(maxlen=1000)
        self._attached = False
        self._orig_execute: Optional[Callable] = None
        self._prefetch_callback: Optional[Callable] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Attach / detach (monkey-patch SpecDecodeWorker)
    # ------------------------------------------------------------------ #
    def attach(self) -> bool:
        """
        Hook into the vLLM worker's spec-decode pipeline.

        Returns True if successfully attached.
        """
        if self._attached:
            logger.warning("SpecIntegrator already attached")
            return True

        if self.vllm_worker is None:
            logger.info("No vLLM worker provided; SpecIntegrator in standalone mode")
            return False

        # Find the spec decode worker
        spec_worker = self._locate_spec_worker(self.vllm_worker)
        if spec_worker is None:
            logger.warning("SpecDecodeWorker not found; cannot attach")
            return False

        if not hasattr(spec_worker, "execute_model"):
            logger.warning("SpecDecodeWorker has no execute_model; cannot attach")
            return False

        import functools
        self._orig_execute = spec_worker.execute_model
        self._spec_worker = spec_worker

        @functools.wraps(self._orig_execute)
        def _hooked_execute(*args, **kwargs):
            return self._wrapped_execute(self._orig_execute, *args, **kwargs)

        spec_worker.execute_model = _hooked_execute  # type: ignore[assignment]
        self._attached = True
        logger.info("SpecIntegrator attached to %s", type(spec_worker).__name__)
        return True

    def detach(self) -> None:
        """Restore original SpecDecodeWorker."""
        if not self._attached:
            return
        if hasattr(self, "_spec_worker") and self._orig_execute is not None:
            self._spec_worker.execute_model = self._orig_execute
        self._attached = False
        self._orig_execute = None
        logger.info("SpecIntegrator detached")

    # ------------------------------------------------------------------ #
    #  Core decision + application
    # ------------------------------------------------------------------ #
    def decide_and_apply_speculation(self, state: Any) -> Dict[str, Any]:
        """
        Ask the controller for K, clamp it, and push it to the spec worker.

        Returns a dict summarizing the decision and whether it was applied.
        """
        if not self.config.adaptive:
            return {"applied": False, "k": self._current_k, "reason": "adaptive_disabled"}

        # Warm-up guard: don't change K until we have enough observations
        if self._tracker.count < self.config.warmup_rounds:
            return {
                "applied": False,
                "k": self._current_k,
                "reason": f"warmup ({self._tracker.count}/{self.config.warmup_rounds})",
            }

        try:
            decision = self.controller.decide_speculation_k(state)
            if not decision.get("apply", False):
                return {"applied": False, "k": self._current_k, **decision}

            k = decision.get("k", self.config.max_spec_tokens)
            k = max(self.config.min_spec_tokens, min(k, self.config.max_spec_tokens))
            self._apply_spec_k(k)
            return {
                "applied": True,
                "k": k,
                "method": self.config.spec_method,
                "reason": decision.get("reason", ""),
            }
        except Exception as e:
            logger.error("Speculation decision failed: %s", e)
            return {"applied": False, "k": self._current_k, "reason": str(e)}

    def _apply_spec_k(self, k: int) -> None:
        """Push K to the vLLM spec-decode subsystem."""
        self._current_k = k
        self._k_history.append(k)

        if self.vllm_worker is None:
            return

        # Path 1: worker has update_speculative_tokens
        if hasattr(self.vllm_worker, "update_speculative_tokens"):
            self.vllm_worker.update_speculative_tokens(k)
            return

        # Path 2: spec_config attribute on engine / worker
        for attr_name in ("spec_config", "speculative_config", "speculation_config"):
            cfg = getattr(self.vllm_worker, attr_name, None)
            if cfg is not None and hasattr(cfg, "num_speculative_tokens"):
                cfg.num_speculative_tokens = k
                return

        # Path 3: /dev/shm IPC for sidecar approach
        try:
            import os
            tmp = "/dev/shm/moe_sd_k.tmp"
            target = "/dev/shm/moe_sd_k"
            with open(tmp, "w") as f:
                f.write(str(k))
            os.replace(tmp, target)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    #  Acceptance reporting
    # ------------------------------------------------------------------ #
    def report_acceptance(
        self, accepted: int, proposed: int, request_id: str = ""
    ) -> None:
        """
        Record acceptance from one verification round.

        Called either manually or automatically by the execute_model hook.
        """
        self._tracker.record(accepted, proposed)

        # Per-request tracking
        if request_id:
            with self._lock:
                rs = self._request_states.get(request_id)
                if rs is None:
                    rs = RequestAcceptanceState(request_id=request_id)
                    self._request_states[request_id] = rs
                rs.accepted_total += accepted
                rs.proposed_total += proposed
                rs.rounds += 1
                rs.last_k = self._current_k or 0

    def finish_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Mark a request complete and return its stats."""
        with self._lock:
            rs = self._request_states.pop(request_id, None)
        if rs is None:
            return None
        return {
            "request_id": request_id,
            "acceptance_rate": round(rs.rate, 4),
            "total_accepted": rs.accepted_total,
            "total_proposed": rs.proposed_total,
            "rounds": rs.rounds,
            "last_k": rs.last_k,
        }

    # ------------------------------------------------------------------ #
    #  Prefetch integration
    # ------------------------------------------------------------------ #
    def set_prefetch_callback(self, callback: Callable) -> None:
        """
        Register a callback for expert prefetch from draft routing.

        The callback signature: ``callback(layer_id: int, expert_ids: List[int])``
        """
        self._prefetch_callback = callback

    def on_draft_routing(self, layer_id: int, expert_ids: List[int]) -> None:
        """
        Called when the draft model produces routing for a layer.

        Forwards to the prefetch callback (e.g., ExpertWeightCache.prefetch).
        """
        if self._prefetch_callback is not None and self.config.enable_draft_prefetch:
            try:
                self._prefetch_callback(layer_id, expert_ids)
            except Exception as e:
                logger.debug("Prefetch callback failed: %s", e)

    # ------------------------------------------------------------------ #
    #  Wrapped execute_model
    # ------------------------------------------------------------------ #
    def _wrapped_execute(self, orig_fn: Callable, *args, **kwargs):
        """
        Wrapper around SpecDecodeWorker.execute_model.

        1. Applies controller K decision (pre-execute).
        2. Runs the original execute_model.
        3. Extracts acceptance stats (post-execute).
        """
        # Pre-execute: apply K
        try:
            state = self._build_state_from_worker()
            if state is not None:
                self.decide_and_apply_speculation(state)
        except Exception as e:
            logger.debug("Pre-execute K decision skipped: %s", e)

        # Run real execute
        result = orig_fn(*args, **kwargs)

        # Post-execute: extract acceptance
        try:
            accepted, proposed = self._extract_acceptance_from_result(result)
            self.report_acceptance(accepted, proposed)
        except Exception:
            pass

        return result

    def _build_state_from_worker(self) -> Optional[Any]:
        """Attempt to build RuntimeState from worker introspection."""
        if RuntimeState is None:
            return None
        try:
            request = RequestState(
                request_id="aggregate",
                prompt_len=0,
                output_len=0,
                request_rate=0.0,
                phase=Phase.DECODE,
            )
            return RuntimeState(
                request=request,
                step_id=0,
                gpu_mem_used_mb=0.0,
                gpu_mem_total_mb=48000.0,
                kv_cache_mb=0.0,
                acceptance_rate=self._tracker.ema_rate,
            )
        except Exception:
            return None

    @staticmethod
    def _extract_acceptance_from_result(result: Any) -> Tuple[int, int]:
        """Extract (accepted, proposed) from SpecDecodeWorker output."""
        if hasattr(result, "num_accepted_tokens"):
            return int(result.num_accepted_tokens), int(result.num_draft_tokens)
        if isinstance(result, dict):
            return int(result.get("accepted", 0)), int(result.get("proposed", 0))
        if isinstance(result, (list, tuple)):
            proposed = len(result)
            accepted = sum(1 for r in result if r is not None)
            return accepted, proposed
        raise ValueError(f"Cannot extract acceptance from {type(result)}")

    @staticmethod
    def _locate_spec_worker(worker: Any) -> Optional[Any]:
        """Walk the vLLM worker tree to find SpecDecodeWorker."""
        # Direct
        if type(worker).__name__ == "SpecDecodeWorker":
            return worker
        # Nested
        for attr in ("spec_worker", "spec_decode_worker", "driver_worker"):
            child = getattr(worker, attr, None)
            if child is not None and hasattr(child, "execute_model"):
                return child
        return None

    # ------------------------------------------------------------------ #
    #  Properties & statistics
    # ------------------------------------------------------------------ #
    @property
    def mean_acceptance(self) -> float:
        return self._tracker.ema_rate

    @property
    def window_acceptance(self) -> float:
        return self._tracker.window_rate

    @property
    def current_k(self) -> Optional[int]:
        return self._current_k

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "current_k": self._current_k,
            "method": self.config.spec_method,
            "adaptive": self.config.adaptive,
            "attached": self._attached,
            "active_requests": len(self._request_states),
            "k_changes": len(self._k_history),
        }
        stats.update(self._tracker.to_dict())
        return stats
