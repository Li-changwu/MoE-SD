#!/usr/bin/env python3
"""Tests for the vllm_moe_sd_scheduler CLI (cli.py)."""
import json, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_moe_sd_scheduler.cli import _build_vllm_serve_cmd, load_controller
from vllm_moe_sd_scheduler.config import SchedulerConfig
from vllm_moe_sd_scheduler.feature_flags import FeatureFlags


class TestLoadController(unittest.TestCase):
    def test_noop(self):
        ctrl = load_controller("noop")
        self.assertEqual(type(ctrl).__name__, "NoOpController")

    def test_unknown_falls_back(self):
        ctrl = load_controller("does_not_exist_xyz")
        self.assertEqual(type(ctrl).__name__, "NoOpController")


class TestBuildVllmServeCmd(unittest.TestCase):
    def _make_config(self, **overrides):
        d = {
            "model": "/path/to/model",
            "workload_profile": "online_short",
            "policy_name": "static",
        }
        d.update(overrides)
        return SchedulerConfig.from_dict(d)

    def test_basic_cmd(self):
        cfg = self._make_config()
        cmd = _build_vllm_serve_cmd(cfg)
        self.assertIn("--model", cmd)
        self.assertIn("/path/to/model", cmd)
        self.assertIn("--port", cmd)

    def test_speculative_model_included(self):
        # Attach vllm_args via monkey-patch since config doesn't have it natively
        cfg = self._make_config()
        cfg.vllm_args = {"speculative_model": "/path/to/eagle", "num_speculative_tokens": 5}
        cmd = _build_vllm_serve_cmd(cfg)
        self.assertIn("--speculative-model", cmd)
        self.assertIn("/path/to/eagle", cmd)
        self.assertIn("--num-speculative-tokens", cmd)
        self.assertIn("5", cmd)

    def test_no_speculative_model(self):
        cfg = self._make_config()
        cmd = _build_vllm_serve_cmd(cfg)
        self.assertNotIn("--speculative-model", cmd)
        self.assertNotIn("--num-speculative-tokens", cmd)

    def test_host_and_port(self):
        cfg = self._make_config()
        cmd = _build_vllm_serve_cmd(cfg)
        self.assertIn("--host", cmd)
        self.assertIn("0.0.0.0", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("8000", cmd)

    def test_custom_args(self):
        cfg = self._make_config()
        cfg.vllm_args = {"max_model_len": 8192, "tensor_parallel_size": 2}
        cmd = _build_vllm_serve_cmd(cfg)
        self.assertIn("--max-model-len", cmd)
        self.assertIn("8192", cmd)
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertIn("2", cmd)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CLI Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLoadController))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildVllmServeCmd))

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
