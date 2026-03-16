import unittest

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.prefetch_policy import AcceptanceAwarePrefetchPolicy


class TestPrefetchPolicy(unittest.TestCase):
    def mk_state(self, acc: float) -> RuntimeState:
        req = RequestState("r1", 256, 128, 1.0, Phase.DECODE)
        return RuntimeState(req, 1, 20000, 46000, 1024, acc)

    def test_tiered_outputs(self):
        p = AcceptanceAwarePrefetchPolicy()
        st = self.mk_state(0.8)
        cands = [
            {"expert_id": 1, "p_expert_needed": 0.9, "p_token_accepted": 0.9, "benefit": 1.0, "cost": 0.1},
            {"expert_id": 2, "p_expert_needed": 0.5, "p_token_accepted": 0.5, "benefit": 1.0, "cost": 0.15},
            {"expert_id": 3, "p_expert_needed": 0.2, "p_token_accepted": 0.2, "benefit": 1.0, "cost": 0.2},
        ]
        out = p.decide(cands, st)
        self.assertIn(1, out["hard"])
        self.assertIn(3, out["defer"])


if __name__ == "__main__":
    unittest.main()
