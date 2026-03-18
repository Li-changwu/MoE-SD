#!/usr/bin/env python3
"""Tests for the real SchedulerHookManager (adapters/vllm_hooks.py)."""
import sys, os, time, types, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.vllm_hooks import AcceptanceWindow, SchedulerHookManager, StepTrace


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
class MockController:
    """Minimal controller stub."""
    def __init__(self, k=3, apply=True):
        self._k = k
        self._apply = apply
        self.calls = 0

    def decide_speculation_k(self, state):
        self.calls += 1
        return {"k": self._k, "apply": self._apply, "reason": "test"}

    def decide_memory_partition(self, state):
        return {"apply": False}

    def decide_prefetch(self, candidates, state):
        return {"apply": False}


class MockEngine:
    """Fake vLLM LLMEngine."""
    def __init__(self):
        self.step_count = 0
        self.speculative_config = types.SimpleNamespace(num_speculative_tokens=3)

    def step(self):
        self.step_count += 1
        return self.step_count


class MockAsyncEngine:
    """Fake AsyncLLMEngine wrapping a MockEngine."""
    def __init__(self):
        self.engine = MockEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAcceptanceWindow(unittest.TestCase):
    def test_empty(self):
        w = AcceptanceWindow()
        self.assertEqual(w.rate, 0.0)
        self.assertEqual(w.count, 0)

    def test_single_record(self):
        w = AcceptanceWindow()
        w.record(3, 5)
        self.assertAlmostEqual(w.rate, 0.6, places=2)

    def test_window_overflow(self):
        w = AcceptanceWindow(window_size=3)
        for _ in range(5):
            w.record(1, 2)
        self.assertEqual(w.count, 3)
        self.assertAlmostEqual(w.rate, 0.5, places=2)


class TestSchedulerHookManager(unittest.TestCase):

    def test_install_wraps_step(self):
        ctrl = MockController(k=4)
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()
        orig_step = engine.step

        mgr.install(engine)
        self.assertTrue(mgr.installed)
        self.assertIsNot(engine.step, orig_step)

        # Calling step still works and invokes controller
        result = engine.step()
        self.assertEqual(result, 1)  # original step returns count
        self.assertEqual(ctrl.calls, 1)
        self.assertEqual(mgr.step_count, 1)

    def test_uninstall_restores(self):
        ctrl = MockController()
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()

        mgr.install(engine)
        mgr.uninstall(engine)
        self.assertFalse(mgr.installed)
        # After uninstall, step should work normally without controller
        result = engine.step()
        self.assertEqual(result, 1)
        # Controller should NOT have been called (hook removed)
        self.assertEqual(ctrl.calls, 0)

    def test_async_engine_unwrap(self):
        ctrl = MockController(k=2)
        mgr = SchedulerHookManager(ctrl)
        async_engine = MockAsyncEngine()
        inner = async_engine.engine

        mgr.install(async_engine)
        self.assertTrue(mgr.installed)
        # The hook should be on the inner engine
        inner.step()
        self.assertEqual(mgr.step_count, 1)

        mgr.uninstall()

    def test_apply_k_via_speculative_config(self):
        ctrl = MockController(k=6, apply=True)
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()
        self.assertEqual(engine.speculative_config.num_speculative_tokens, 3)

        mgr.install(engine)
        engine.step()  # triggers controller → apply K=6
        self.assertEqual(engine.speculative_config.num_speculative_tokens, 6)
        self.assertEqual(mgr.current_k, 6)
        mgr.uninstall()

    def test_controller_apply_false(self):
        ctrl = MockController(k=5, apply=False)
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()

        mgr.install(engine)
        engine.step()
        # K should not be applied
        self.assertEqual(engine.speculative_config.num_speculative_tokens, 3)
        self.assertIsNone(mgr.current_k)
        mgr.uninstall()

    def test_get_statistics(self):
        ctrl = MockController(k=3)
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()
        mgr.install(engine)
        engine.step()
        engine.step()

        stats = mgr.get_statistics()
        self.assertTrue(stats["installed"])
        self.assertEqual(stats["step_count"], 2)
        self.assertEqual(stats["current_k"], 3)
        self.assertIn("acceptance_rate", stats)
        mgr.uninstall()

    def test_double_install_warns(self):
        ctrl = MockController()
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()
        mgr.install(engine)
        # Second install should not raise
        mgr.install(engine)
        self.assertTrue(mgr.installed)
        mgr.uninstall()

    def test_no_step_attribute_raises(self):
        ctrl = MockController()
        mgr = SchedulerHookManager(ctrl)
        with self.assertRaises(AttributeError):
            mgr.install(object())

    def test_get_recent_traces(self):
        ctrl = MockController(k=2)
        mgr = SchedulerHookManager(ctrl)
        engine = MockEngine()
        mgr.install(engine)
        for _ in range(5):
            engine.step()

        traces = mgr.get_recent_traces(n=3)
        self.assertEqual(len(traces), 3)
        self.assertIn("step_id", traces[0])
        self.assertIn("timestamp", traces[0])
        mgr.uninstall()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Hooks (Real) Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAcceptanceWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulerHookManager))

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
