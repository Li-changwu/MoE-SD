"""Tests for MoE-Infinity request-level prefetch planner."""

import torch

from adapters.moe_infinity_prefetch import MoEInfinityConfig, SequenceTracePrefetcher


def test_sequence_trace_prefetch_learns_from_history():
    p = SequenceTracePrefetcher(
        num_layers=4,
        config=MoEInfinityConfig(history_size=8, horizon=2, max_prefetch_k=3, min_similarity=0.0),
    )

    # Request 1: layer0 -> layer1 follows (1,2) -> (5,6)
    p.observe_layer(0, torch.tensor([[1, 2], [1, 2]], dtype=torch.long))
    p.observe_layer(1, torch.tensor([[5, 6], [5, 6]], dtype=torch.long))

    # Start request 2 by wrapping to layer0 (finalizes request 1)
    plan = p.plan_prefetch(
        current_layer_id=0,
        topk_ids=torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
        targets=[
            {"layer_id": 1, "cached": set(), "num_experts": 16, "width_cap": 3},
        ],
    )

    assert 1 in plan
    assert len(plan[1]) > 0
    assert any(e in {5, 6} for e in plan[1])


def test_sequence_trace_prefetch_respects_cached_and_k():
    p = SequenceTracePrefetcher(
        num_layers=4,
        config=MoEInfinityConfig(history_size=8, horizon=2, max_prefetch_k=2, min_similarity=0.0),
    )

    p.observe_layer(0, torch.tensor([[3, 4]], dtype=torch.long))
    p.observe_layer(1, torch.tensor([[7, 8]], dtype=torch.long))

    plan = p.plan_prefetch(
        current_layer_id=0,
        topk_ids=torch.tensor([[3, 4]], dtype=torch.long),
        targets=[
            {"layer_id": 1, "cached": {7}, "num_experts": 16, "width_cap": 4},
        ],
    )

    assert 1 in plan
    assert 7 not in plan[1]
    assert len(plan[1]) <= 2
