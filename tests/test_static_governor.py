import unittest

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.static_governor import StaticGovernor


class TestStaticGovernor(unittest.TestCase):
    def mk_state(self, phase: Phase, used: float, total: float, acc: float, kv: float) -> RuntimeState:
        req = RequestState(
            request_id="r1",
            prompt_len=512,
            output_len=128,
            request_rate=2.0,
            phase=phase,
        )
        return RuntimeState(
            request=req,
            step_id=1,
            gpu_mem_used_mb=used,
            gpu_mem_total_mb=total,
            kv_cache_mb=kv,
            acceptance_rate=acc,
        )

    def test_reduce_k_under_high_pressure(self):
        g = StaticGovernor()
        st = self.mk_state(Phase.DECODE, used=42000, total=46000, acc=0.9, kv=1000)
        out = g.decide_speculation_k(st)
        self.assertEqual(out["k"], 1)

    def test_reduce_k_under_low_acceptance(self):
        g = StaticGovernor()
        st = self.mk_state(Phase.DECODE, used=20000, total=46000, acc=0.2, kv=1000)
        out = g.decide_speculation_k(st)
        self.assertEqual(out["k"], 1)

    def test_prefill_more_conservative(self):
        g = StaticGovernor()
        st = self.mk_state(Phase.PREFILL, used=20000, total=46000, acc=0.8, kv=1000)
        out = g.decide_speculation_k(st)
        self.assertLessEqual(out["k"], 2)

    def test_memory_partition_keys(self):
        g = StaticGovernor()
        st = self.mk_state(Phase.DECODE, used=38000, total=46000, acc=0.3, kv=2000)
        out = g.decide_memory_partition(st)
        self.assertIn("expert_budget_mb", out)
        self.assertIn("speculative_budget_mb", out)
        self.assertIn("kv_reserve_mb", out)


if __name__ == "__main__":
    unittest.main()
