import unittest

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.phase_aware_governor import PhaseAwareGovernor


class TestPhaseAwareGovernor(unittest.TestCase):
    def mk_state(self, phase: Phase, used: float, total: float, acc: float) -> RuntimeState:
        req = RequestState(
            request_id="req-pa",
            prompt_len=1024,
            output_len=256,
            request_rate=2.0,
            phase=phase,
        )
        return RuntimeState(
            request=req,
            step_id=1,
            gpu_mem_used_mb=used,
            gpu_mem_total_mb=total,
            kv_cache_mb=2048,
            acceptance_rate=acc,
        )

    def test_prefill_k_more_conservative(self):
        g = PhaseAwareGovernor()
        pre = g.decide_speculation_k(self.mk_state(Phase.PREFILL, 20000, 46000, 0.8))
        dec = g.decide_speculation_k(self.mk_state(Phase.DECODE, 20000, 46000, 0.8))
        self.assertLessEqual(pre["k"], dec["k"])

    def test_low_acceptance_reduces_k(self):
        g = PhaseAwareGovernor()
        out = g.decide_speculation_k(self.mk_state(Phase.DECODE, 20000, 46000, 0.2))
        self.assertEqual(out["k"], 1)

    def test_prefill_has_higher_kv_bias(self):
        g = PhaseAwareGovernor()
        pre = g.decide_memory_partition(self.mk_state(Phase.PREFILL, 20000, 46000, 0.8))
        dec = g.decide_memory_partition(self.mk_state(Phase.DECODE, 20000, 46000, 0.8))
        self.assertGreater(pre["kv_reserve_mb"], dec["kv_reserve_mb"])


if __name__ == "__main__":
    unittest.main()
