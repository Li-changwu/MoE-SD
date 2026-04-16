"""Tests for AdapMoE gating/prefetch/cache allocation adapters."""

import torch

from adapters.adapmoe_cache_alloc import LayerRuntimeStat, allocate_slots_dp
from adapters.adapmoe_gating import AdapMoEGatingConfig, SensitivityAdaptiveGating
from adapters.adapmoe_prefetch import AdapMoEConfig, AdaptiveGatingPrefetcher


def test_adapmoe_gating_collapses_high_confidence_tokens():
    gating = SensitivityAdaptiveGating(
        num_layers=4,
        config=AdapMoEGatingConfig(target_single_ratio=0.6, warmup_steps=0),
    )

    topk_weights = torch.tensor(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.55, 0.45],
            [0.52, 0.48],
        ],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long)

    new_w, new_ids, ratio = gating.apply(
        layer_id=2,
        layer_rank=2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    assert ratio > 0.0
    # At least one token should be forced to single-expert routing.
    assert bool((new_w[:, 1] == 0).any())
    assert bool((new_ids[:, 1] == new_ids[:, 0]).any())


def test_adapmoe_prefetch_learns_transition_and_supports_horizon():
    prefetcher = AdaptiveGatingPrefetcher(
        config=AdapMoEConfig(min_prefetch_k=1, max_prefetch_k=3, horizon=2)
    )

    # Step 1 traces: layer 0 -> layer 1 transition dominated by 3/4.
    prefetcher.observe_layer(0, torch.tensor([[1, 1], [2, 2]], dtype=torch.long))
    prefetcher.observe_layer(1, torch.tensor([[3, 3], [4, 4]], dtype=torch.long))

    # Step 2 starts from layer 0 again and asks for multi-layer planning.
    topk_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    router_logits = torch.tensor([[5.0, 1.0], [4.0, 1.0]], dtype=torch.float32)
    hidden_states = torch.randn(2, 8)

    plan = prefetcher.plan_prefetch(
        current_layer_id=0,
        topk_ids=topk_ids,
        router_logits=router_logits,
        hidden_states=hidden_states,
        targets=[
            {"layer_id": 1, "cached": set(), "num_experts": 8, "width_cap": 3},
            {"layer_id": 2, "cached": set(), "num_experts": 8, "width_cap": 2},
        ],
    )

    assert 1 in plan
    assert len(plan[1]) >= 1
    # Learned transition should prioritize experts seen in layer 1 trace.
    assert any(e in {3, 4} for e in plan[1])


def test_adapmoe_cache_alloc_dp_respects_budget_and_minimum():
    stats = [
        LayerRuntimeStat("l0", capacity=8, miss_rate=0.8, prefetch_acc=0.2, single_ratio=0.3),
        LayerRuntimeStat("l1", capacity=8, miss_rate=0.3, prefetch_acc=0.7, single_ratio=0.7),
        LayerRuntimeStat("l2", capacity=8, miss_rate=0.6, prefetch_acc=0.4, single_ratio=0.4),
    ]

    alloc = allocate_slots_dp(stats=stats, total_slots=12, min_slots_per_layer=2)

    assert set(alloc.keys()) == {"l0", "l1", "l2"}
    assert sum(alloc.values()) == 12
    assert all(v >= 2 for v in alloc.values())
    assert all(v <= 8 for v in alloc.values())
