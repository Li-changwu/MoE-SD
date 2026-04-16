"""
Unit tests for v3.1 Overflow Controller (C1-C6).
"""
import unittest
from unittest.mock import MagicMock, patch
from collections import OrderedDict

from adapters.overflow_controller import (
    WorkingSetEstimator,
    WorkingSetInfo,
    OverflowRiskDetector,
    OverflowReport,
    AdaptiveKGovernor,
    SelectivePrefetch,
    RescueReservation,
    StepFeedbackCollector,
    StepFeedback,
    OverflowController,
    H2D_PER_EXPERT_MS,
)


class TestWorkingSetEstimator(unittest.TestCase):
    """C1: Working-set estimation from draft routing."""

    def setUp(self):
        self.estimator = WorkingSetEstimator()
        # Simulate 3 layers
        self.estimator._layer_idx_to_name = {
            0: "model.layers.0.mlp.experts",
            1: "model.layers.1.mlp.experts",
            2: "model.layers.2.mlp.experts",
        }

    def test_all_cached(self):
        """No misses when all experts are cached."""
        draft_routing = {0: [1, 2, 3], 1: [4, 5, 6]}
        cache_state = {
            "model.layers.0.mlp.experts": {1, 2, 3, 4, 5},
            "model.layers.1.mlp.experts": {4, 5, 6, 7, 8},
        }
        result = self.estimator.estimate(draft_routing, cache_state)
        self.assertEqual(result["model.layers.0.mlp.experts"].miss_count, 0)
        self.assertEqual(result["model.layers.1.mlp.experts"].miss_count, 0)

    def test_all_miss(self):
        """All experts miss when cache is empty."""
        draft_routing = {0: [10, 20, 30]}
        cache_state = {"model.layers.0.mlp.experts": set()}
        result = self.estimator.estimate(draft_routing, cache_state)
        info = result["model.layers.0.mlp.experts"]
        self.assertEqual(info.miss_count, 3)
        self.assertAlmostEqual(info.miss_ratio, 1.0)
        self.assertEqual(set(info.miss_experts), {10, 20, 30})

    def test_partial_miss(self):
        """Mixed hit/miss scenario."""
        draft_routing = {0: [1, 2, 3, 4, 5]}
        cache_state = {"model.layers.0.mlp.experts": {1, 2, 3}}
        result = self.estimator.estimate(draft_routing, cache_state)
        info = result["model.layers.0.mlp.experts"]
        self.assertEqual(info.miss_count, 2)
        self.assertEqual(info.union_size, 5)
        self.assertAlmostEqual(info.miss_ratio, 0.4)

    def test_prefetch_pending_excluded(self):
        """Experts in-flight via DIPP are not counted as misses."""
        draft_routing = {0: [1, 2, 3, 4, 5]}
        cache_state = {"model.layers.0.mlp.experts": {1, 2}}
        pending = {"model.layers.0.mlp.experts": {3, 4}}
        result = self.estimator.estimate(draft_routing, cache_state, pending)
        info = result["model.layers.0.mlp.experts"]
        self.assertEqual(info.miss_count, 1)  # only expert 5
        self.assertEqual(info.prefetch_inflight, 2)  # experts 3, 4

    def test_unknown_layer_skipped(self):
        """Layer indices not in mapping are silently skipped."""
        draft_routing = {99: [1, 2, 3]}
        cache_state = {}
        result = self.estimator.estimate(draft_routing, cache_state)
        self.assertEqual(len(result), 0)

    def test_duplicate_experts(self):
        """Duplicate expert IDs in routing are deduplicated."""
        draft_routing = {0: [1, 1, 2, 2, 3]}
        cache_state = {"model.layers.0.mlp.experts": {1}}
        result = self.estimator.estimate(draft_routing, cache_state)
        info = result["model.layers.0.mlp.experts"]
        self.assertEqual(info.union_size, 3)  # {1, 2, 3}
        self.assertEqual(info.miss_count, 2)  # {2, 3}


class TestOverflowRiskDetector(unittest.TestCase):
    """C2: Overflow risk detection based on expected stall cost."""

    def setUp(self):
        self.detector = OverflowRiskDetector(stall_threshold_ms=1.0)

    def test_no_overflow(self):
        """Zero misses → action = none."""
        ws_info = {
            "layer0": WorkingSetInfo(8, 0, 0.0, [], 0),
        }
        report = self.detector.detect(ws_info, {}, {})
        self.assertEqual(report.recommended_action, "none")
        self.assertEqual(len(report.overflow_layers), 0)

    def test_small_miss_prefetch(self):
        """A few cold misses → action = prefetch."""
        ws_info = {
            "layer0": WorkingSetInfo(8, 4, 0.5, [10, 11, 12, 13], 0),
        }
        report = self.detector.detect(ws_info, {}, {})
        # 4 × 0.378 ≈ 1.51ms per layer → total < 3ms
        self.assertEqual(report.recommended_action, "prefetch")
        self.assertEqual(len(report.overflow_layers), 1)

    def test_moderate_miss_lower_k(self):
        """Many misses across layers → action = lower_k."""
        ws_info = {}
        for i in range(5):
            name = f"layer{i}"
            ws_info[name] = WorkingSetInfo(8, 3, 0.375, [10, 11, 12], 0)
        report = self.detector.detect(ws_info, {}, {})
        # 5 layers × 3 × 0.378 ≈ 5.67ms → 3-8ms range
        self.assertEqual(report.recommended_action, "lower_k")

    def test_severe_rescue(self):
        """Many misses + eviction pressure → action = rescue."""
        ws_info = {}
        evictions = {}
        for i in range(10):
            name = f"layer{i}"
            ws_info[name] = WorkingSetInfo(8, 5, 0.625, list(range(5)), 0)
            evictions[name] = 10
        report = self.detector.detect(ws_info, evictions, {})
        # 10 layers × (5×0.378 + 10×0.05) = 10 × 2.39 = 23.9ms → rescue
        self.assertEqual(report.recommended_action, "rescue")

    def test_slow_path_penalty(self):
        """Recent slow path adds penalty."""
        ws_info = {
            "layer0": WorkingSetInfo(8, 3, 0.375, [10, 11, 12], 0),
        }
        # Without slow path
        report1 = self.detector.detect(ws_info, {}, {})
        stall1 = report1.total_expected_stall_ms

        self.detector.tail_latency_ema.clear()
        # With slow path flag
        report2 = self.detector.detect(ws_info, {}, {"layer0": True})
        stall2 = report2.total_expected_stall_ms
        self.assertGreater(stall2, stall1)

    def test_ema_updates(self):
        """EMA tracks historical stall."""
        ws_info = {
            "layer0": WorkingSetInfo(8, 5, 0.625, list(range(5)), 0),
        }
        self.detector.detect(ws_info, {}, {})
        ema1 = self.detector.tail_latency_ema["layer0"]
        self.assertGreater(ema1, 0)

        # Second call with no miss
        ws_zero = {
            "layer0": WorkingSetInfo(8, 0, 0.0, [], 0),
        }
        self.detector.detect(ws_zero, {}, {})
        ema2 = self.detector.tail_latency_ema["layer0"]
        self.assertLess(ema2, ema1)  # EMA decays


class TestAdaptiveKGovernor(unittest.TestCase):
    """C3: Adaptive speculation depth control."""

    def setUp(self):
        self.gov = AdaptiveKGovernor(K_max=4, K_min=1)

    def test_initial_K(self):
        self.assertEqual(self.gov.K, 4)

    def test_lower_k_on_overflow(self):
        """K decreases by 1 on lower_k action."""
        report = OverflowReport([], 5.0, 5.0, "lower_k")
        k = self.gov.adjust(report)
        self.assertEqual(k, 3)
        k = self.gov.adjust(report)
        self.assertEqual(k, 2)

    def test_K_floor(self):
        """K doesn't go below K_min."""
        report = OverflowReport([], 10.0, 10.0, "rescue")
        for _ in range(10):
            self.gov.adjust(report)
        self.assertEqual(self.gov.K, 1)

    def test_slow_recovery(self):
        """K increases only after 3 consecutive calm steps."""
        # First drive K down
        self.gov.K = 2
        self.gov._calm_steps = 0

        none_report = OverflowReport.NONE
        self.gov.adjust(none_report)  # calm_steps = 1
        self.assertEqual(self.gov.K, 2)
        self.gov.adjust(none_report)  # calm_steps = 2
        self.assertEqual(self.gov.K, 2)
        self.gov.adjust(none_report)  # calm_steps = 3 → K+1
        self.assertEqual(self.gov.K, 3)

    def test_calm_reset_on_overflow(self):
        """Calm counter resets when overflow triggers."""
        self.gov.K = 3
        self.gov._calm_steps = 2

        report = OverflowReport([], 5.0, 5.0, "lower_k")
        self.gov.adjust(report)
        self.assertEqual(self.gov._calm_steps, 0)
        self.assertEqual(self.gov.K, 2)

    def test_K_ceiling(self):
        """K doesn't exceed K_max."""
        self.gov.K = 4
        none_report = OverflowReport.NONE
        for _ in range(10):
            self.gov.adjust(none_report)
        self.assertEqual(self.gov.K, 4)


class TestSelectivePrefetch(unittest.TestCase):
    """C4: Selective prefetch with hard budget."""

    def setUp(self):
        self.elmm = MagicMock()
        self.prefetcher = SelectivePrefetch(self.elmm)

    def test_no_action_on_none(self):
        """No prefetch when action is none."""
        report = OverflowReport.NONE
        count = self.prefetcher.execute(report, {})
        self.assertEqual(count, 0)
        self.elmm.prefetch_experts.assert_not_called()

    def test_prefetch_overflow_layers(self):
        """Prefetch miss experts for overflow layers."""
        ws_info = {
            "layer0": WorkingSetInfo(8, 3, 0.375, [10, 11, 12], 0),
        }
        report = OverflowReport(
            overflow_layers=[("layer0", 2.0)],
            total_expected_stall_ms=2.0,
            severity=2.0,
            recommended_action="prefetch",
        )
        count = self.prefetcher.execute(report, ws_info)
        self.assertEqual(count, 3)
        self.elmm.prefetch_experts.assert_called_once_with("layer0", [10, 11, 12])

    def test_budget_cap(self):
        """Per-step budget of 16 experts is enforced."""
        ws_info = {}
        overflow = []
        for i in range(10):
            name = f"layer{i}"
            ws_info[name] = WorkingSetInfo(8, 5, 0.625, list(range(i*10, i*10+5)), 0)
            overflow.append((name, 5.0))
        report = OverflowReport(overflow, 50.0, 50.0, "rescue")
        count = self.prefetcher.execute(report, ws_info)
        self.assertLessEqual(count, 16)

    def test_layer_cap(self):
        """Max 6 layers per step."""
        ws_info = {}
        overflow = []
        for i in range(10):
            name = f"layer{i}"
            ws_info[name] = WorkingSetInfo(8, 1, 0.125, [i*10], 0)
            overflow.append((name, 2.0))
        report = OverflowReport(overflow, 20.0, 20.0, "rescue")
        self.prefetcher.execute(report, ws_info)
        # Should only call prefetch for at most 6 layers
        self.assertLessEqual(self.elmm.prefetch_experts.call_count, 6)


class TestRescueReservation(unittest.TestCase):
    """C5: Reservation-only rescue path."""

    def _make_cache(self, cached_experts):
        cache = MagicMock()
        cache._slot_map = OrderedDict((e, i) for i, e in enumerate(cached_experts))
        cache.contains = lambda eid: eid in cache._slot_map
        return cache

    def setUp(self):
        self.elmm = MagicMock()
        cache = self._make_cache([1, 2, 3, 4, 5])
        self.elmm._layer_caches = {"layer0": cache}
        self.rescue = RescueReservation(self.elmm)

    def test_no_action_unless_rescue(self):
        """Skip when action is not rescue."""
        report = OverflowReport(
            overflow_layers=[("layer0", 2.0)],
            total_expected_stall_ms=2.0,
            severity=2.0,
            recommended_action="prefetch",  # not rescue
        )
        ws_info = {"layer0": WorkingSetInfo(8, 2, 0.25, [1, 6], 0)}
        pinned = self.rescue.pre_reserve(report, ws_info)
        self.assertEqual(pinned, 0)

    def test_pin_cached_experts(self):
        """Pin experts that are in cache and in miss list."""
        report = OverflowReport(
            overflow_layers=[("layer0", 10.0)],
            total_expected_stall_ms=10.0,
            severity=10.0,
            recommended_action="rescue",
        )
        # miss_experts [2, 3, 99] — 2,3 are cached, 99 is not
        ws_info = {"layer0": WorkingSetInfo(8, 3, 0.375, [2, 3, 99], 0)}
        pinned = self.rescue.pre_reserve(report, ws_info)
        self.assertEqual(pinned, 2)

    def test_release_clears_state(self):
        """Release clears pinned tracking."""
        self.rescue._pinned = [("layer0", 1), ("layer0", 2)]
        self.rescue.release()
        self.assertEqual(len(self.rescue._pinned), 0)


class TestStepFeedbackCollector(unittest.TestCase):
    """C6: Step feedback collection."""

    def setUp(self):
        self.c6 = StepFeedbackCollector()

    def test_basic_flow(self):
        """Record layers and finalize step."""
        self.c6.begin_step(OverflowReport.NONE)
        self.c6.record_layer("layer0", 5, 3, 10)
        self.c6.record_layer("layer1", 8, 0, 5)
        fb = self.c6.finalize_step()
        self.assertEqual(fb.total_misses, 3)
        self.assertEqual(fb.ready_ratio, 1.0)  # no overflow predicted

    def test_ready_ratio_with_overflow(self):
        """ready_ratio counts overflow layers that avoided slow path."""
        overflow = OverflowReport(
            overflow_layers=[("layer0", 2.0), ("layer1", 3.0)],
            total_expected_stall_ms=5.0,
            severity=5.0,
            recommended_action="lower_k",
        )
        self.c6.begin_step(overflow)
        self.c6.record_layer("layer0", 5, 3, 10)  # misses > 0 → slow path
        self.c6.record_layer("layer1", 8, 0, 5)    # no miss → avoided
        fb = self.c6.finalize_step()
        self.assertAlmostEqual(fb.ready_ratio, 0.5)  # 1 of 2

    def test_eviction_delta(self):
        """Eviction delta computed correctly across steps."""
        self.c6.begin_step(OverflowReport.NONE)
        self.c6.record_layer("layer0", 5, 0, 10)  # evictions=10 (first)
        fb1 = self.c6.finalize_step()

        self.c6.begin_step(OverflowReport.NONE)
        self.c6.record_layer("layer0", 5, 0, 15)  # evictions=15 (delta=5)
        fb2 = self.c6.finalize_step()
        self.assertEqual(fb2.total_evictions, 5)


class TestOverflowController(unittest.TestCase):
    """Integration: OverflowController orchestrates C1-C6."""

    def _make_elmm(self):
        elmm = MagicMock()
        # Simulate 3 layer caches
        caches = {}
        for i in range(3):
            name = f"model.layers.{i}.mlp.experts"
            cache = MagicMock()
            cache._slot_map = OrderedDict(
                (e, s) for s, e in enumerate(range(i*10, i*10+8))
            )
            cache.contains = lambda eid, c=cache: eid in c._slot_map
            cache._evictions = 0
            caches[name] = cache
        elmm._layer_caches = caches
        elmm._layer_name_to_id = {
            f"model.layers.{i}.mlp.experts": i for i in range(3)
        }
        elmm._last_expert_set = {
            f"model.layers.{i}.mlp.experts": set(range(i*10, i*10+8))
            for i in range(3)
        }
        elmm._ordered_layers = list(caches.keys())
        return elmm

    def test_end_to_end_no_overflow(self):
        """Full cycle with no overflow → action=none, K unchanged."""
        elmm = self._make_elmm()
        ctrl = OverflowController(elmm, K_max=4, K_min=1)
        ctrl.configure()

        # All experts are cached → no miss
        draft_routing = {0: [0, 1, 2], 1: [10, 11], 2: [20, 21]}
        report = ctrl.on_draft_complete(draft_routing)

        self.assertEqual(report.recommended_action, "none")
        self.assertEqual(ctrl.get_recommended_K(), 4)
        elmm.prefetch_experts.assert_not_called()

    def test_end_to_end_with_overflow(self):
        """Full cycle with overflow → K reduced, prefetch issued."""
        elmm = self._make_elmm()
        ctrl = OverflowController(elmm, K_max=4, K_min=1, stall_threshold_ms=0.1)
        ctrl.configure()

        # Request experts NOT in cache → misses
        draft_routing = {0: [50, 51, 52, 53, 54, 55, 56, 57]}
        report = ctrl.on_draft_complete(draft_routing)

        self.assertNotEqual(report.recommended_action, "none")
        self.assertLess(ctrl.get_recommended_K(), 4)

    def test_feedback_loop(self):
        """C6 feedback reaches C2 for next step."""
        elmm = self._make_elmm()
        ctrl = OverflowController(elmm, K_max=4, K_min=1)
        ctrl.configure()

        # Step 1: trigger overflow
        draft_routing = {0: [50, 51, 52]}
        ctrl.on_draft_complete(draft_routing)

        # Simulate layer completions
        for name in elmm._layer_caches:
            cache = elmm._layer_caches[name]
            ctrl.on_layer_complete(name, 5, 3, cache._evictions)

        fb = ctrl.on_step_complete()
        self.assertIsInstance(fb, StepFeedback)
        self.assertEqual(ctrl.feedback.total_steps, 1)

    def test_stats_tracking(self):
        """Controller stats are populated."""
        elmm = self._make_elmm()
        ctrl = OverflowController(elmm, K_max=4, K_min=1)
        ctrl.configure()

        stats = ctrl.get_stats()
        self.assertIn("total_steps", stats)
        self.assertIn("overflow_rate", stats)
        self.assertIn("current_K", stats)
        self.assertEqual(stats["current_K"], 4)

    def test_disabled_controller(self):
        """Disabled controller is a no-op."""
        elmm = self._make_elmm()
        ctrl = OverflowController(elmm, enabled=False)
        ctrl.configure()

        report = ctrl.on_draft_complete({0: [50, 51]})
        self.assertEqual(report.recommended_action, "none")

    def test_overhead_budget(self):
        """on_draft_complete should be fast (<1ms on CPU)."""
        import time
        elmm = self._make_elmm()
        ctrl = OverflowController(elmm, K_max=4, K_min=1)
        ctrl.configure()

        draft_routing = {i: list(range(i*10, i*10+8)) for i in range(3)}

        # Warmup
        ctrl.on_draft_complete(draft_routing)

        # Measure
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            ctrl.on_draft_complete(draft_routing)
        elapsed = (time.perf_counter() - t0) / N * 1000
        # Should be well under 0.1ms (pure CPU dict ops)
        self.assertLess(elapsed, 1.0, f"on_draft_complete took {elapsed:.3f}ms")


if __name__ == "__main__":
    unittest.main()
