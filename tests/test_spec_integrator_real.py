#!/usr/bin/env python3
"""Tests for the real SpecIntegrator (adapters/spec_integrator.py)."""
import sys, os, types, unittest, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.spec_integrator import (
    AcceptanceTracker,
    SpecIntegrator,
    SpeculationConfig,
    RequestAcceptanceState,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
class MockController:
    def __init__(self, k=3, apply=True):
        self._k = k
        self._apply = apply
        self.calls = 0

    def decide_speculation_k(self, state):
        self.calls += 1
        return {"k": self._k, "apply": self._apply, "reason": "test"}


class MockWorker:
    """Fake vLLM worker."""
    def __init__(self):
        self.speculative_tokens = 3
        self._execute_calls = 0

    def update_speculative_tokens(self, k):
        self.speculative_tokens = k


class MockSpecDecodeWorker:
    """Fake SpecDecodeWorker with execute_model."""
    def __init__(self):
        self.exec_count = 0

    def execute_model(self, *args, **kwargs):
        self.exec_count += 1
        return types.SimpleNamespace(num_accepted_tokens=2, num_draft_tokens=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAcceptanceTracker(unittest.TestCase):
    def test_empty(self):
        t = AcceptanceTracker()
        self.assertEqual(t.count, 0)
        self.assertAlmostEqual(t.ema_rate, 0.5, places=2)  # neutral start

    def test_record_updates_ema(self):
        t = AcceptanceTracker(ema_alpha=0.5)
        t.record(3, 5)  # instant rate = 0.6
        # EMA = 0.5 * 0.5 + 0.5 * 0.6 = 0.55
        self.assertAlmostEqual(t.ema_rate, 0.55, places=2)

    def test_window_rate(self):
        t = AcceptanceTracker(window_size=10)
        t.record(3, 5)
        t.record(7, 10)
        # window: (3+7)/(5+10) = 10/15 ≈ 0.667
        self.assertAlmostEqual(t.window_rate, 10 / 15, places=3)

    def test_global_rate(self):
        t = AcceptanceTracker()
        t.record(1, 4)
        t.record(3, 4)
        self.assertAlmostEqual(t.global_rate, 4 / 8, places=3)

    def test_to_dict(self):
        t = AcceptanceTracker()
        t.record(5, 10)
        d = t.to_dict()
        self.assertIn("ema_rate", d)
        self.assertIn("window_rate", d)
        self.assertIn("total_accepted", d)
        self.assertEqual(d["total_accepted"], 5)

    def test_thread_safety(self):
        t = AcceptanceTracker()
        errors = []

        def _writer():
            for _ in range(100):
                t.record(1, 2)

        threads = [threading.Thread(target=_writer) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # 4 threads × 100 records = 400 total proposed
        self.assertEqual(t._total_proposed, 800)


class TestSpecIntegrator(unittest.TestCase):

    def test_standalone_mode(self):
        ctrl = MockController()
        si = SpecIntegrator(ctrl, vllm_worker=None)
        ok = si.attach()
        self.assertFalse(ok)  # no worker → standalone

    def test_report_acceptance(self):
        ctrl = MockController()
        si = SpecIntegrator(ctrl)
        si.report_acceptance(3, 5, request_id="r1")
        si.report_acceptance(2, 5, request_id="r1")
        self.assertAlmostEqual(si.mean_acceptance, 0.5, delta=0.1)

    def test_per_request_tracking(self):
        ctrl = MockController()
        si = SpecIntegrator(ctrl)
        si.report_acceptance(2, 3, request_id="r1")
        si.report_acceptance(1, 3, request_id="r1")
        stats = si.finish_request("r1")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["total_accepted"], 3)
        self.assertEqual(stats["total_proposed"], 6)
        self.assertEqual(stats["rounds"], 2)

    def test_finish_unknown_request(self):
        ctrl = MockController()
        si = SpecIntegrator(ctrl)
        self.assertIsNone(si.finish_request("nonexistent"))

    def test_decide_and_apply_with_worker(self):
        ctrl = MockController(k=5, apply=True)
        worker = MockWorker()
        cfg = SpeculationConfig(warmup_rounds=0)
        si = SpecIntegrator(ctrl, vllm_worker=worker, config=cfg)
        # Feed enough samples to pass warmup (warmup_rounds=0)
        result = si.decide_and_apply_speculation(None)
        self.assertTrue(result["applied"])
        self.assertEqual(result["k"], 5)
        self.assertEqual(worker.speculative_tokens, 5)

    def test_warmup_blocks_adaptation(self):
        ctrl = MockController(k=5, apply=True)
        cfg = SpeculationConfig(warmup_rounds=10)
        si = SpecIntegrator(ctrl, config=cfg)
        result = si.decide_and_apply_speculation(None)
        # Not enough samples → blocked
        self.assertFalse(result["applied"])
        self.assertIn("warmup", result["reason"])

    def test_adaptive_disabled(self):
        ctrl = MockController(k=5, apply=True)
        cfg = SpeculationConfig(adaptive=False)
        si = SpecIntegrator(ctrl, config=cfg)
        result = si.decide_and_apply_speculation(None)
        self.assertFalse(result["applied"])
        self.assertIn("adaptive_disabled", result["reason"])

    def test_k_clamping(self):
        ctrl = MockController(k=100, apply=True)
        worker = MockWorker()
        cfg = SpeculationConfig(max_spec_tokens=8, min_spec_tokens=1, warmup_rounds=0)
        si = SpecIntegrator(ctrl, vllm_worker=worker, config=cfg)
        result = si.decide_and_apply_speculation(None)
        self.assertEqual(result["k"], 8)  # clamped to max

    def test_prefetch_callback(self):
        ctrl = MockController()
        si = SpecIntegrator(ctrl)
        calls = []
        si.set_prefetch_callback(lambda lid, eids: calls.append((lid, eids)))
        si.on_draft_routing(layer_id=5, expert_ids=[1, 2, 3])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (5, [1, 2, 3]))

    def test_get_statistics(self):
        ctrl = MockController()
        si = SpecIntegrator(ctrl)
        si.report_acceptance(5, 10)
        stats = si.get_statistics()
        self.assertIn("method", stats)
        self.assertIn("ema_rate", stats)
        self.assertIn("total_accepted", stats)
        self.assertEqual(stats["total_accepted"], 5)

    def test_attach_detach_spec_worker(self):
        ctrl = MockController()
        worker = MockSpecDecodeWorker()
        # SpecIntegrator uses _locate_spec_worker which checks type name
        si = SpecIntegrator(ctrl, vllm_worker=worker)
        ok = si.attach()
        if ok:
            # execute_model should be hooked
            worker.execute_model()
            self.assertTrue(si._attached)
            si.detach()
            self.assertFalse(si._attached)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SpecIntegrator (Real) Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAcceptanceTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecIntegrator))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {result.testsRun}")

    if result.failures:
        for test, tb in result.failures:
            print(f"\nFAIL: {test}")
            print(tb)
    if result.errors:
        for test, tb in result.errors:
            print(f"\nERROR: {test}")
            print(tb)

    sys.exit(0 if failed == 0 else 1)
