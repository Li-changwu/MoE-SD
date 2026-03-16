import unittest

from controllers.interface import (
    NoOpController,
    Phase,
    RequestState,
    RuntimeState,
    build_decision_trace,
)


class TestControllerInterface(unittest.TestCase):
    def setUp(self):
        req = RequestState(
            request_id="req-1",
            prompt_len=128,
            output_len=64,
            request_rate=1.0,
            phase=Phase.DECODE,
        )
        self.state = RuntimeState(
            request=req,
            step_id=3,
            gpu_mem_used_mb=12345.0,
            gpu_mem_total_mb=46000.0,
            kv_cache_mb=2048.0,
            acceptance_rate=0.6,
        )
        self.controller = NoOpController()

    def test_noop_speculation_decision(self):
        out = self.controller.decide_speculation_k(self.state)
        self.assertFalse(out["apply"])
        self.assertIn("reason", out)

    def test_noop_memory_decision(self):
        out = self.controller.decide_memory_partition(self.state)
        self.assertFalse(out["apply"])
        self.assertIn("kv_reserve_mb", out)

    def test_noop_prefetch_decision(self):
        candidates = [{"expert_id": 1}, {"expert_id": 2}]
        out = self.controller.decide_prefetch(candidates, self.state)
        self.assertFalse(out["apply"])
        self.assertEqual(out["defer"], [1, 2])

    def test_decision_trace_builder(self):
        trace = build_decision_trace(
            request_id="req-1",
            phase="decode",
            step_id=3,
            component="speculation_k",
            decision={"k": 2, "apply": True},
            reason="acceptance_high",
            policy_name="static_v0",
        )
        self.assertEqual(trace["request_id"], "req-1")
        self.assertEqual(trace["component"], "speculation_k")
        self.assertIn("decision", trace)


if __name__ == "__main__":
    unittest.main()
