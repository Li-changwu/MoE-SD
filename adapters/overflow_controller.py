"""
v3.1 Overflow Controller — C1 through C6
==========================================
A lightweight, step-level overlay on BriskMoE that detects cache overflow
risk from speculative decoding's working-set explosion, and responds with:
  - C3: Adaptive K reduction (slow variable, next-step)
  - C4: Selective prefetch for overflow layers (fast variable, current-step)
  - C5: Rescue reservation / pin (fast variable, current-step)

Design invariant: <0.05 ms overhead on non-overflow steps.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from adapters.elmm_plugin import ELMMManager

logger = logging.getLogger(__name__)

# ── Physical constants (measured on A6000, PCIe Gen4 x16) ────────────
EXPERT_BYTES = 9.44e6        # 9.44 MB per expert (BF16)
PCIE_BW = 25e9               # 25 GB/s practical
H2D_PER_EXPERT_MS = EXPERT_BYTES / PCIE_BW * 1000  # ~0.378 ms
SLOW_PATH_OVERHEAD_MS = 0.5  # remap rebuild + eviction + dispatch

# ── C4 hard budget caps ──────────────────────────────────────────────
MAX_OVERLAY_PREFETCH_PER_STEP = 16
MAX_OVERLAY_LAYERS_PER_STEP = 6


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class WorkingSetInfo:
    """Per-layer working-set estimate from draft routing."""
    union_size: int
    miss_count: int          # cold misses (excluding prefetch in-flight)
    miss_ratio: float
    miss_experts: list[int]
    prefetch_inflight: int = 0


@dataclass
class OverflowReport:
    """Output of C2: per-step overflow risk assessment."""
    overflow_layers: list[tuple[str, float]]  # [(layer_name, stall_ms)]
    total_expected_stall_ms: float
    severity: float                           # = total_expected_stall_ms
    recommended_action: str                   # "none"|"prefetch"|"lower_k"|"rescue"

    @property
    def overflow_layer_names(self) -> set[str]:
        return {name for name, _ in self.overflow_layers}

    # Convenience: empty report for non-overflow steps
    NONE: ClassVar["OverflowReport"]  # set below


OverflowReport.NONE = OverflowReport(
    overflow_layers=[],
    total_expected_stall_ms=0.0,
    severity=0.0,
    recommended_action="none",
)


@dataclass
class StepFeedback:
    """Output of C6: per-step actual execution feedback."""
    total_misses: int = 0
    total_evictions: int = 0
    ready_ratio: float = 1.0       # fraction of overflow layers that avoided slow path
    slow_path_layers: set[str] = field(default_factory=set)


# =====================================================================
# C1: Working-Set Estimator
# =====================================================================

class WorkingSetEstimator:
    """Estimate per-layer expert working-set for the upcoming verify pass.

    Frequency: 1x per decode step.  Cost: pure CPU, O(K × L × top_k).
    """

    def __init__(self) -> None:
        self._layer_idx_to_name: dict[int, str] = {}

    def configure(self, layer_caches: dict[str, Any]) -> None:
        """Build layer_index → layer_name mapping from ELMM caches."""
        for name in layer_caches:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        self._layer_idx_to_name[int(parts[i + 1])] = name
                    except ValueError:
                        pass

    def estimate(
        self,
        draft_routing: dict[int, list[int]],
        cache_state: dict[str, set[int]],
        prefetch_pending: dict[str, set[int]] | None = None,
    ) -> dict[str, WorkingSetInfo]:
        pending = prefetch_pending or {}
        results: dict[str, WorkingSetInfo] = {}
        for layer_idx, expert_ids in draft_routing.items():
            layer_name = self._layer_idx_to_name.get(layer_idx)
            if layer_name is None:
                continue
            union = set(expert_ids)
            cached = cache_state.get(layer_name, set())
            arriving = pending.get(layer_name, set())
            misses = union - cached - arriving
            results[layer_name] = WorkingSetInfo(
                union_size=len(union),
                miss_count=len(misses),
                miss_ratio=len(misses) / max(1, len(union)),
                miss_experts=list(misses),
                prefetch_inflight=len(union & arriving),
            )
        return results


# =====================================================================
# C2: Overflow Risk Detector
# =====================================================================

class OverflowRiskDetector:
    """Detect per-layer overflow risk as expected stall cost (ms).

    severity ≈ expected_stall_ms, NOT miss_ratio.
    """

    def __init__(
        self,
        stall_threshold_ms: float = 1.0,
        pressure_ema_alpha: float = 0.1,
    ) -> None:
        self.stall_threshold_ms = stall_threshold_ms
        self.pressure_ema_alpha = pressure_ema_alpha
        self.tail_latency_ema: dict[str, float] = {}
        # Fed by C6
        self._last_feedback: StepFeedback = StepFeedback()

    def detect(
        self,
        ws_info: dict[str, WorkingSetInfo],
        recent_evictions: dict[str, int],
        recent_slow_path: dict[str, bool],
    ) -> OverflowReport:
        overflow_layers: list[tuple[str, float]] = []
        total_stall = 0.0

        for layer, info in ws_info.items():
            cold_miss = info.miss_count
            h2d_ms = cold_miss * H2D_PER_EXPERT_MS
            evict_penalty = recent_evictions.get(layer, 0) * 0.05
            slow_flag = SLOW_PATH_OVERHEAD_MS if recent_slow_path.get(layer) else 0.0

            expected_stall = h2d_ms + evict_penalty + slow_flag

            # Update EMA
            prev = self.tail_latency_ema.get(layer, 0.0)
            self.tail_latency_ema[layer] = (
                (1 - self.pressure_ema_alpha) * prev
                + self.pressure_ema_alpha * expected_stall
            )

            if expected_stall > self.stall_threshold_ms:
                overflow_layers.append((layer, expected_stall))
                total_stall += expected_stall

        # Select action based on total expected stall
        if not overflow_layers:
            action = "none"
        elif total_stall < 3.0:
            action = "prefetch"
        elif total_stall < 8.0:
            action = "lower_k"
        else:
            action = "rescue"

        return OverflowReport(
            overflow_layers=overflow_layers,
            total_expected_stall_ms=total_stall,
            severity=total_stall,
            recommended_action=action,
        )

    def update_feedback(self, feedback: StepFeedback) -> None:
        self._last_feedback = feedback


# =====================================================================
# C3: Adaptive K Governor
# =====================================================================

class AdaptiveKGovernor:
    """Step-level adaptive speculation depth controller.

    IMPORTANT: C3 is a SLOW-VARIABLE controller.
    adjust() output takes effect on the NEXT step, not the current one.
    For current-step rescue, rely on C4 (prefetch) and C5 (reservation).
    """

    def __init__(self, K_max: int = 4, K_min: int = 1) -> None:
        self.K = K_max
        self.K_max = K_max
        self.K_min = K_min
        self._calm_steps = 0
        # Stats
        self.total_adjustments = 0
        self.total_reductions = 0

    def adjust(self, overflow_report: OverflowReport) -> int:
        """Adjust K based on overflow report. Returns new K for NEXT step."""
        if overflow_report.recommended_action in ("lower_k", "rescue"):
            new_k = max(self.K_min, self.K - 1)
            if new_k != self.K:
                self.total_reductions += 1
                self.total_adjustments += 1
            self.K = new_k
            self._calm_steps = 0
        elif overflow_report.recommended_action == "none" and self.K < self.K_max:
            self._calm_steps += 1
            if self._calm_steps >= 3:
                self.K = min(self.K_max, self.K + 1)
                self._calm_steps = 0
                self.total_adjustments += 1
        else:
            self._calm_steps = 0
        return self.K


# =====================================================================
# C4: Selective Prefetch
# =====================================================================

class SelectivePrefetch:
    """Enhanced DIPP: prioritize overflow-risk layers with HARD budget.

    Does NOT replace existing DIPP. Only adds overflow-priority prefetch
    when C2 detects overflow risk. Retreats to zero extra prefetch
    when overflow severity drops to "none".
    """

    def __init__(self, elmm: "ELMMManager") -> None:
        self._elmm = elmm
        # Stats
        self.total_extra_prefetches = 0
        self.total_activations = 0

    def execute(
        self,
        overflow_report: OverflowReport,
        ws_info: dict[str, WorkingSetInfo],
    ) -> int:
        """Issue priority prefetches for overflow layers.

        Returns number of experts actually prefetched.
        """
        if overflow_report.recommended_action == "none":
            return 0

        self.total_activations += 1

        # Sort by expected stall descending
        sorted_layers = sorted(
            overflow_report.overflow_layers,
            key=lambda x: x[1],
            reverse=True,
        )

        budget_remaining = MAX_OVERLAY_PREFETCH_PER_STEP
        layers_used = 0
        total_prefetched = 0

        for layer_name, _stall_ms in sorted_layers:
            if budget_remaining <= 0 or layers_used >= MAX_OVERLAY_LAYERS_PER_STEP:
                break
            info = ws_info.get(layer_name)
            if not info or not info.miss_experts:
                continue
            experts_to_prefetch = info.miss_experts[:budget_remaining]
            if experts_to_prefetch:
                self._elmm.prefetch_experts(layer_name, experts_to_prefetch)
                n = len(experts_to_prefetch)
                budget_remaining -= n
                total_prefetched += n
                layers_used += 1

        self.total_extra_prefetches += total_prefetched
        return total_prefetched


# =====================================================================
# C5: Rescue Reservation (Phase 1: Reservation-only)
# =====================================================================

class RescueReservation:
    """Pin critical experts before verify forward to prevent cascading eviction.

    Phase 1: Reservation-only — no kernel changes, no extra allocation.
    Only pins existing cache slots to prevent eviction during slow path.
    """

    def __init__(self, elmm: "ELMMManager") -> None:
        self._elmm = elmm
        self._pinned: list[tuple[str, int]] = []  # (layer_name, expert_id) to unpin
        # Stats
        self.total_pins = 0
        self.total_activations = 0

    def pre_reserve(
        self,
        overflow_report: OverflowReport,
        ws_info: dict[str, WorkingSetInfo],
    ) -> int:
        """Pin experts that are about to be needed, preventing eviction.

        Returns number of experts pinned.
        """
        if overflow_report.recommended_action != "rescue":
            return 0

        self.total_activations += 1
        pinned = 0

        for layer_name, _stall_ms in overflow_report.overflow_layers:
            cache = self._elmm._layer_caches.get(layer_name)
            if cache is None:
                continue
            info = ws_info.get(layer_name)
            if info is None:
                continue
            # Pin currently-cached experts that this step needs
            # (prevent eviction by other layers' loading)
            for eid in info.miss_experts:
                # We can only pin experts that are already cached
                # The point is: protect recently loaded experts from
                # being immediately evicted by other layers
                if cache.contains(eid):
                    # Move to end of LRU (protect from eviction)
                    if eid in cache._slot_map:
                        cache._slot_map.move_to_end(eid)
                        self._pinned.append((layer_name, eid))
                        pinned += 1

        self.total_pins += pinned
        return pinned

    def release(self) -> None:
        """Release all pins after verify forward completes."""
        # In Phase 1 (LRU move-to-end), no explicit unpin needed.
        # The LRU promotion already happened.
        self._pinned.clear()


# =====================================================================
# C6: Step Feedback Collector
# =====================================================================

class StepFeedbackCollector:
    """Lightweight per-step feedback for closing the C2/C3 control loop.

    Collects miss/eviction/slow-path data per layer within a step,
    then summarizes into a StepFeedback at step boundary.
    """

    def __init__(self) -> None:
        self._step_miss_acc: dict[str, int] = {}
        self._step_evict_acc: dict[str, int] = {}
        self._step_slow_path: dict[str, bool] = {}
        self._prev_evictions: dict[str, int] = {}
        self._current_overflow_report: OverflowReport = OverflowReport.NONE
        # Stats
        self.total_steps = 0
        self.total_overflow_steps = 0
        self.avg_ready_ratio = 0.0
        self._ready_ratio_sum = 0.0

    def begin_step(self, overflow_report: OverflowReport) -> None:
        """Call at the start of each verify forward pass."""
        self._step_miss_acc.clear()
        self._step_evict_acc.clear()
        self._step_slow_path.clear()
        self._current_overflow_report = overflow_report

    def record_layer(
        self,
        layer_name: str,
        step_hits: int,
        step_misses: int,
        cache_evictions: int,
    ) -> None:
        """Call after each layer's forward in _elmm_forward_impl."""
        self._step_miss_acc[layer_name] = step_misses
        prev = self._prev_evictions.get(layer_name, cache_evictions)
        delta_evict = cache_evictions - prev
        self._step_evict_acc[layer_name] = max(0, delta_evict)
        self._prev_evictions[layer_name] = cache_evictions
        self._step_slow_path[layer_name] = step_misses > 0

    def finalize_step(self) -> StepFeedback:
        """Call after the last offloaded layer. Returns step summary."""
        self.total_steps += 1

        total_misses = sum(self._step_miss_acc.values())
        total_evictions = sum(self._step_evict_acc.values())

        # Compute ready_ratio
        overflow_names = self._current_overflow_report.overflow_layer_names
        if overflow_names:
            self.total_overflow_steps += 1
            avoided = sum(
                1 for l in overflow_names
                if not self._step_slow_path.get(l, False)
            )
            ready_ratio = avoided / len(overflow_names)
        else:
            ready_ratio = 1.0

        self._ready_ratio_sum += ready_ratio
        self.avg_ready_ratio = self._ready_ratio_sum / self.total_steps

        return StepFeedback(
            total_misses=total_misses,
            total_evictions=total_evictions,
            ready_ratio=ready_ratio,
            slow_path_layers=set(
                l for l, v in self._step_slow_path.items() if v
            ),
        )


# =====================================================================
# Top-level Controller (orchestrates C1-C6)
# =====================================================================

class OverflowController:
    """Step-level overflow controller that orchestrates C1-C6.

    Usage:
        controller = OverflowController(elmm_manager)
        controller.configure()

        # In draft_prefetch_hook, after draft completes:
        report = controller.on_draft_complete(draft_routing)
        # report.recommended_action -> adjust K on proposer

        # In _elmm_forward_impl, per-layer:
        controller.on_layer_complete(layer_name, hits, misses, evictions)

        # After last offloaded layer:
        controller.on_step_complete()
    """

    def __init__(
        self,
        elmm: "ELMMManager",
        K_max: int = 4,
        K_min: int = 1,
        stall_threshold_ms: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._elmm = elmm

        # C1-C6 modules
        self.estimator = WorkingSetEstimator()           # C1
        self.detector = OverflowRiskDetector(            # C2
            stall_threshold_ms=stall_threshold_ms,
        )
        self.governor = AdaptiveKGovernor(K_max=K_max, K_min=K_min)  # C3
        self.prefetcher = SelectivePrefetch(elmm)        # C4
        self.rescue = RescueReservation(elmm)            # C5
        self.feedback = StepFeedbackCollector()           # C6

        # State
        self._last_report: OverflowReport = OverflowReport.NONE
        self._last_ws_info: dict[str, WorkingSetInfo] = {}
        self._configured = False

        # Stats
        self.total_on_draft_calls = 0
        self.total_on_draft_ms = 0.0

    def configure(self) -> None:
        """Initialize from ELMM state. Call after ELMM install."""
        self.estimator.configure(self._elmm._layer_caches)
        self._configured = True
        print(f"[OverflowController] Configured: K=[{self.governor.K_min},{self.governor.K_max}], "
              f"stall_thresh={self.detector.stall_threshold_ms}ms, "
              f"layers={len(self._elmm._layer_caches)}",
              file=sys.stderr, flush=True)

    def on_draft_complete(
        self,
        draft_routing: dict[int, list[int]],
    ) -> OverflowReport:
        """Called after draft forward. Runs C1→C2→C3→C4→C5.

        Returns the overflow report so the caller can adjust K.
        """
        if not self.enabled or not self._configured:
            return OverflowReport.NONE

        t0 = time.perf_counter()
        self.total_on_draft_calls += 1

        # ── C1: Working-set estimate ──
        cache_state: dict[str, set[int]] = {}
        for name, cache in self._elmm._layer_caches.items():
            cache_state[name] = set(cache._slot_map.keys())

        ws_info = self.estimator.estimate(
            draft_routing, cache_state,
        )
        self._last_ws_info = ws_info

        # ── C2: Overflow risk detection ──
        recent_evictions: dict[str, int] = {}
        recent_slow: dict[str, bool] = {}
        last_fb = self.detector._last_feedback
        for layer in ws_info:
            recent_evictions[layer] = self.feedback._step_evict_acc.get(layer, 0)
            recent_slow[layer] = layer in last_fb.slow_path_layers

        report = self.detector.detect(ws_info, recent_evictions, recent_slow)
        self._last_report = report

        # ── C3: Adaptive K (for NEXT step) ──
        self.governor.adjust(report)

        # ── C4: Selective prefetch (current-step fast variable) ──
        self.prefetcher.execute(report, ws_info)

        # ── C5: Rescue reservation (current-step fast variable) ──
        self.rescue.pre_reserve(report, ws_info)

        # ── C6: Begin step tracking ──
        self.feedback.begin_step(report)

        elapsed = (time.perf_counter() - t0) * 1000
        self.total_on_draft_ms += elapsed

        return report

    def on_layer_complete(
        self,
        layer_name: str,
        step_hits: int,
        step_misses: int,
        cache_evictions: int,
    ) -> None:
        """Called after each layer's _elmm_forward_impl. Feeds C6."""
        if not self.enabled:
            return
        self.feedback.record_layer(
            layer_name, step_hits, step_misses, cache_evictions,
        )

    def on_step_complete(self) -> StepFeedback:
        """Called after the last offloaded layer. Closes feedback loop."""
        if not self.enabled:
            return StepFeedback()

        # Release C5 pins
        self.rescue.release()

        # Finalize C6 feedback
        fb = self.feedback.finalize_step()
        self.detector.update_feedback(fb)
        return fb

    def get_recommended_K(self) -> int:
        """Current K recommendation from C3."""
        return self.governor.K

    def get_stats(self) -> dict:
        """Return controller stats for logging."""
        return {
            "total_steps": self.feedback.total_steps,
            "overflow_steps": self.feedback.total_overflow_steps,
            "overflow_rate": (
                self.feedback.total_overflow_steps / max(1, self.feedback.total_steps)
            ),
            "avg_ready_ratio": self.feedback.avg_ready_ratio,
            "current_K": self.governor.K,
            "K_reductions": self.governor.total_reductions,
            "K_adjustments": self.governor.total_adjustments,
            "c4_activations": self.prefetcher.total_activations,
            "c4_extra_prefetches": self.prefetcher.total_extra_prefetches,
            "c5_activations": self.rescue.total_activations,
            "c5_pins": self.rescue.total_pins,
            "avg_on_draft_ms": (
                self.total_on_draft_ms / max(1, self.total_on_draft_calls)
            ),
            "last_action": self._last_report.recommended_action,
            "last_severity_ms": self._last_report.total_expected_stall_ms,
        }
