"""
vLLM Scheduler Hooks for MoE-SD Controller Integration
=======================================================
Monkey-patches vLLM engine ``step()`` to inject SpecMoE controller
decisions (K tuning, memory partitioning, expert prefetch) at every
decode step, and collects acceptance-rate / KV-cache observations
back into the controller's feedback loop.

Hook architecture:
  1. ``LLMEngine.step()`` is wrapped to call ``on_step_begin`` /
     ``on_step_end`` around the real step.
  2. ``SpecDecodeWorker`` (if present) is wrapped so that after
     verification we capture accepted / rejected counts.
  3. ``Scheduler``'s KV-cache occupancy is sampled every N steps
     for memory-partition enforcement.

All hooks are installed / uninstalled atomically; failure at any
point leaves the original vLLM code untouched.
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional vLLM imports – these may not be available in dev containers.
# ---------------------------------------------------------------------------
try:
    from vllm.engine.llm_engine import LLMEngine  # type: ignore
    HAS_VLLM_ENGINE = True
except ImportError:
    HAS_VLLM_ENGINE = False
    LLMEngine = None  # type: ignore

try:
    from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker  # type: ignore
    HAS_SPEC_DECODE = True
except ImportError:
    HAS_SPEC_DECODE = False
    SpecDecodeWorker = None  # type: ignore

# Import our controller interface
try:
    from controllers.interface import Phase, RequestState, RuntimeState
except ImportError:
    # Fallback stubs so the module is importable outside MoE-SD tree
    Phase = None  # type: ignore
    RequestState = None  # type: ignore
    RuntimeState = None  # type: ignore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class StepTrace:
    """Trace entry for one engine step."""
    step_id: int
    timestamp: float
    k_decided: Optional[int] = None
    acceptance_rate: float = 0.0
    accepted_tokens: int = 0
    proposed_tokens: int = 0
    kv_usage: float = 0.0
    gpu_mem_used_mb: float = 0.0
    decision_reason: str = ""


@dataclass
class AcceptanceWindow:
    """Sliding window for acceptance-rate estimation."""
    window_size: int = 64
    _accepted: Deque[int] = field(default_factory=deque)
    _proposed: Deque[int] = field(default_factory=deque)

    def record(self, accepted: int, proposed: int) -> None:
        self._accepted.append(accepted)
        self._proposed.append(proposed)
        while len(self._accepted) > self.window_size:
            self._accepted.popleft()
            self._proposed.popleft()

    @property
    def rate(self) -> float:
        total_p = sum(self._proposed)
        if total_p == 0:
            return 0.0
        return sum(self._accepted) / total_p

    @property
    def count(self) -> int:
        return len(self._accepted)


# ---------------------------------------------------------------------------
# SchedulerHookManager
# ---------------------------------------------------------------------------
class SchedulerHookManager:
    """
    Manages hooks into vLLM scheduler / engine lifecycle events.

    Usage::

        mgr = SchedulerHookManager(controller)
        mgr.install(engine)       # monkey-patches engine.step
        ...                        # normal vLLM operation
        mgr.uninstall(engine)     # restores original step
    """

    def __init__(
        self,
        controller: Any,
        *,
        trace_capacity: int = 10_000,
        kv_sample_interval: int = 4,
        gpu_total_mb: float = 48_000.0,
    ):
        self.controller = controller
        self._installed = False
        self._lock = threading.Lock()

        # --- observation state ---
        self._step_id = 0
        self._acceptance = AcceptanceWindow()
        self._traces: Deque[StepTrace] = deque(maxlen=trace_capacity)
        self._kv_sample_interval = kv_sample_interval
        self._gpu_total_mb = gpu_total_mb
        self._last_kv_usage = 0.0

        # --- saved originals (for uninstall) ---
        self._orig_engine_step: Optional[Callable] = None
        self._orig_spec_execute: Optional[Callable] = None
        self._engine_ref: Optional[Any] = None
        self._spec_worker_ref: Optional[Any] = None

        # --- current K ---
        self._current_k: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Install / uninstall
    # ------------------------------------------------------------------ #
    def install(self, engine: Any) -> None:
        """
        Install hooks on a vLLM ``LLMEngine`` (or ``AsyncLLMEngine``).

        Wraps ``engine.step`` and, if spec-decode is configured, also wraps
        the ``SpecDecodeWorker.execute_model`` method to capture acceptance
        rates.
        """
        with self._lock:
            if self._installed:
                logger.warning("SchedulerHookManager already installed")
                return

            # ----- engine.step hook -----
            real_engine = engine
            # AsyncLLMEngine wraps an LLMEngine
            if hasattr(engine, "engine"):
                real_engine = engine.engine

            if not hasattr(real_engine, "step"):
                raise AttributeError(
                    f"{type(real_engine).__name__} has no 'step' method"
                )

            self._orig_engine_step = real_engine.step
            self._engine_ref = real_engine

            @functools.wraps(self._orig_engine_step)
            def _hooked_step(*args, **kwargs):
                return self._wrapped_step(self._orig_engine_step, *args, **kwargs)

            real_engine.step = _hooked_step  # type: ignore[assignment]

            # ----- spec decode worker hook (optional) -----
            spec_worker = self._find_spec_worker(real_engine)
            if spec_worker is not None and hasattr(spec_worker, "execute_model"):
                self._orig_spec_execute = spec_worker.execute_model
                self._spec_worker_ref = spec_worker

                @functools.wraps(self._orig_spec_execute)
                def _hooked_execute(*a, **kw):
                    return self._wrapped_spec_execute(
                        self._orig_spec_execute, *a, **kw
                    )

                spec_worker.execute_model = _hooked_execute  # type: ignore[assignment]

            self._installed = True
            logger.info(
                "SchedulerHookManager installed (engine=%s, spec_worker=%s)",
                type(real_engine).__name__,
                type(spec_worker).__name__ if spec_worker else "None",
            )

    def uninstall(self, engine: Any = None) -> None:
        """Restore original methods."""
        with self._lock:
            if not self._installed:
                return

            if self._engine_ref is not None and self._orig_engine_step is not None:
                self._engine_ref.step = self._orig_engine_step
                self._orig_engine_step = None

            if (
                self._spec_worker_ref is not None
                and self._orig_spec_execute is not None
            ):
                self._spec_worker_ref.execute_model = self._orig_spec_execute
                self._orig_spec_execute = None

            self._engine_ref = None
            self._spec_worker_ref = None
            self._installed = False
            logger.info("SchedulerHookManager uninstalled")

    # ------------------------------------------------------------------ #
    #  Wrapped methods
    # ------------------------------------------------------------------ #
    def _wrapped_step(self, orig_step: Callable, *args, **kwargs):
        """Called in place of ``LLMEngine.step``."""
        self._step_id += 1
        trace = StepTrace(step_id=self._step_id, timestamp=time.time())

        # ---- PRE-STEP: ask controller for K ----
        try:
            state = self._build_runtime_state()
            decision = self.controller.decide_speculation_k(state)
            k = decision.get("k")
            if decision.get("apply", False) and k is not None:
                self._apply_k(k)
                trace.k_decided = k
                trace.decision_reason = decision.get("reason", "")
                self._current_k = k
        except Exception as e:
            logger.debug("Controller decision skipped: %s", e)

        # ---- REAL STEP ----
        result = orig_step(*args, **kwargs)

        # ---- POST-STEP: sample KV cache ----
        if self._step_id % self._kv_sample_interval == 0:
            self._sample_kv_cache()

        trace.acceptance_rate = self._acceptance.rate
        trace.kv_usage = self._last_kv_usage
        self._traces.append(trace)
        return result

    def _wrapped_spec_execute(self, orig_execute: Callable, *args, **kwargs):
        """Wraps ``SpecDecodeWorker.execute_model`` to capture acceptance."""
        result = orig_execute(*args, **kwargs)

        # vLLM SpecDecodeWorker returns SamplerOutput or similar;
        # extraction depends on version. Try common attribute names.
        try:
            accepted, proposed = self._extract_acceptance(result)
            self._acceptance.record(accepted, proposed)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_spec_worker(engine: Any) -> Optional[Any]:
        """Locate the SpecDecodeWorker inside a vLLM engine."""
        # Path 1: engine.model_executor.driver_worker.spec_worker
        try:
            executor = getattr(engine, "model_executor", None)
            if executor is None:
                return None
            driver = getattr(executor, "driver_worker", None)
            if driver is None:
                # Try workers list
                workers = getattr(executor, "workers", [])
                for w in workers:
                    if type(w).__name__ == "SpecDecodeWorker":
                        return w
                return None
            # Direct attribute
            spec = getattr(driver, "spec_worker", None)
            if spec is not None:
                return spec
            # Check if driver IS the spec worker
            if type(driver).__name__ == "SpecDecodeWorker":
                return driver
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_acceptance(result: Any) -> Tuple[int, int]:
        """
        Extract (accepted, proposed) counts from spec-decode output.

        vLLM structures vary across versions; we try multiple paths.
        """
        # Path 1: result has .num_accepted_tokens / .num_draft_tokens
        if hasattr(result, "num_accepted_tokens"):
            return int(result.num_accepted_tokens), int(result.num_draft_tokens)

        # Path 2: list of SamplerOutput – count non-None entries
        if isinstance(result, (list, tuple)):
            proposed = len(result)
            accepted = sum(1 for r in result if r is not None)
            return accepted, proposed

        # Path 3: dict with 'accepted' / 'proposed'
        if isinstance(result, dict):
            return int(result.get("accepted", 0)), int(result.get("proposed", 0))

        raise ValueError(f"Cannot extract acceptance from {type(result)}")

    def _build_runtime_state(self) -> "RuntimeState":
        """Construct a RuntimeState for the controller."""
        if RuntimeState is None:
            raise RuntimeError("controllers.interface not available")

        request = RequestState(
            request_id="aggregate",
            prompt_len=0,
            output_len=0,
            request_rate=0.0,
            phase=Phase.DECODE,
        )
        return RuntimeState(
            request=request,
            step_id=self._step_id,
            gpu_mem_used_mb=self._gpu_total_mb * max(0.5, self._last_kv_usage),
            gpu_mem_total_mb=self._gpu_total_mb,
            kv_cache_mb=self._last_kv_usage * self._gpu_total_mb * 0.4,
            acceptance_rate=self._acceptance.rate,
        )

    def _apply_k(self, k: int) -> None:
        """Push a new K to the spec decode config."""
        engine = self._engine_ref
        if engine is None:
            return

        # Path 1: engine.scheduler_config or engine.speculative_config
        spec_cfg = getattr(engine, "speculative_config", None)
        if spec_cfg is not None and hasattr(spec_cfg, "num_speculative_tokens"):
            spec_cfg.num_speculative_tokens = k
            return

        # Path 2: write to /dev/shm for metrics-sidecar based approach
        try:
            import os
            tmp = "/dev/shm/moe_sd_k.tmp"
            target = "/dev/shm/moe_sd_k"
            with open(tmp, "w") as f:
                f.write(str(k))
            os.replace(tmp, target)
        except OSError:
            pass

    def _sample_kv_cache(self) -> None:
        """Read KV-cache occupancy from the scheduler."""
        engine = self._engine_ref
        if engine is None:
            return
        try:
            scheduler = getattr(engine, "scheduler", None)
            if scheduler is None:
                schedulers = getattr(engine, "scheduler", [])
                if isinstance(schedulers, list) and schedulers:
                    scheduler = schedulers[0]
            if scheduler is None:
                return
            block_manager = getattr(scheduler, "block_manager", None)
            if block_manager is None:
                return
            usage = getattr(block_manager, "gpu_cache_usage", None)
            if callable(usage):
                self._last_kv_usage = float(usage())
            elif isinstance(usage, (int, float)):
                self._last_kv_usage = float(usage)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Public observation API
    # ------------------------------------------------------------------ #
    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def step_count(self) -> int:
        return self._step_id

    @property
    def current_k(self) -> Optional[int]:
        return self._current_k

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance.rate

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "installed": self._installed,
            "step_count": self._step_id,
            "current_k": self._current_k,
            "acceptance_rate": round(self._acceptance.rate, 4),
            "acceptance_samples": self._acceptance.count,
            "kv_usage": round(self._last_kv_usage, 4),
            "trace_count": len(self._traces),
        }

    def get_recent_traces(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return last *n* step traces as dicts."""
        import dataclasses

        recent = list(self._traces)[-n:]
        return [dataclasses.asdict(t) for t in recent]
