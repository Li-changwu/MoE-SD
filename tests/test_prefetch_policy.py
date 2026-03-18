import unittest

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.prefetch_policy import (
    AcceptanceAwarePrefetchPolicy,
    FrontierAwarePrefetchPolicy,
)


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

    def test_frontier_policy_prefers_shared_experts(self):
        p = FrontierAwarePrefetchPolicy()
        st = self.mk_state(0.45)
        cands = [
            {
                "expert_id": 7,
                "p_expert_needed": 0.92,
                "p_token_accepted": 0.45,
                "benefit": 1.2,
                "cost": 0.10,
                "shared_count": 3,
                "frontier_size": 4,
                "avg_depth": 1.5,
                "frontier_depth": 3,
                "expert_reuse": 0.9,
            },
            {
                "expert_id": 42,
                "p_expert_needed": 0.85,
                "p_token_accepted": 0.45,
                "benefit": 1.1,
                "cost": 0.10,
                "shared_count": 1,
                "frontier_size": 4,
                "avg_depth": 3.0,
                "frontier_depth": 3,
                "expert_reuse": 0.1,
            },
        ]
        out = p.decide(cands, st)
        self.assertIn(7, out["hard"] + out["soft"])
        self.assertIn(42, out["defer"])
        self.assertGreater(out["wasted_prefetched_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
