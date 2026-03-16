import unittest

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.memory_partition_controller import DynamicMemoryPartitionController


class TestMemoryPartitionController(unittest.TestCase):
    def mk_state(self, used: float, acc: float) -> RuntimeState:
        req = RequestState("r1", 256, 128, 1.0, Phase.DECODE)
        return RuntimeState(req, 8, used, 46000, 2048, acc)

    def test_outputs_have_required_fields(self):
        c = DynamicMemoryPartitionController()
        out = c.decide(self.mk_state(42000, 0.3), step_id=8)
        self.assertIn("expert_budget_mb", out)
        self.assertIn("speculative_budget_mb", out)
        self.assertIn("kv_reserve_mb", out)

    def test_apply_respects_interval(self):
        c = DynamicMemoryPartitionController()
        out1 = c.decide(self.mk_state(30000, 0.8), step_id=1)
        out2 = c.decide(self.mk_state(30000, 0.8), step_id=8)
        self.assertFalse(out1["apply"])
        self.assertTrue(out2["apply"])


if __name__ == "__main__":
    unittest.main()
