"""
Tests for the SpecMoE Engine — End-to-End Integration
======================================================
Verifies that all components wire together correctly.
"""

import sys
import torch

sys.path.insert(0, "/root/MoE-SD")


def test_engine_initialization():
    """Engine should initialize all components based on config."""
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig

    config = SpecMoEConfig(
        num_experts=16,
        top_k=4,
        hidden_size=32,
        moe_intermediate_size=16,
        num_moe_layers=4,
        enable_dedup=True,
        enable_sdd=True,
        enable_cache=True,
        enable_trace=False,
        hook_vllm=False,  # No vLLM in test env
        gpu_cache_budget_gb=0.01,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()

    assert engine._initialized
    assert engine._dispatcher is not None
    assert engine._sdd is not None
    assert engine._cache is not None
    assert engine._prefetch is not None

    stats = engine.get_statistics()
    assert stats["initialized"]
    assert stats["verify_rounds"] == 0
    print(f"  PASS: engine initialization")


def test_manual_dispatch_without_sdd():
    """Manual dispatch should work without SDD."""
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig
    import torch.nn.functional as F

    config = SpecMoEConfig(
        num_experts=8,
        top_k=2,
        hidden_size=16,
        moe_intermediate_size=8,
        num_moe_layers=2,
        enable_dedup=True,
        enable_sdd=False,
        enable_cache=False,
        hook_vllm=False,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()

    torch.manual_seed(42)
    T, D, E, N = 4, 16, 8, 8
    hidden = torch.randn(T, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01
    logits = torch.randn(T, E)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    output = engine.dispatch_moe(hidden, w1, w2, weights, ids, layer_id=0, is_verify=True)

    assert output.shape == (T, D)
    assert output.abs().max() > 0
    print(f"  PASS: manual dispatch (output_norm={output.norm():.4f})")


def test_manual_dispatch_with_sdd():
    """Manual dispatch with SDD should freeze divergent tokens."""
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig
    import torch.nn.functional as F

    config = SpecMoEConfig(
        num_experts=8,
        top_k=2,
        hidden_size=16,
        moe_intermediate_size=8,
        num_moe_layers=4,
        enable_dedup=True,
        enable_sdd=True,
        sdd_min_check_layer=0,
        sdd_consecutive_threshold=2,
        sdd_method="entropy",
        enable_cache=False,
        hook_vllm=False,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()

    # Initialize SDD for 3 draft tokens
    engine._sdd.init_verify_round(num_draft_tokens=3)

    torch.manual_seed(42)
    T, D, E, N = 4, 16, 8, 8
    hidden = torch.randn(T, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01

    # Concentrated routing → high entropy divergence → will freeze
    logits = torch.zeros(T, E)
    logits[:, 0] = 10.0  # Concentrated on expert 0
    weights, ids = torch.topk(F.softmax(logits, dim=-1), 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Run through multiple layers to trigger SDD freeze
    for layer in range(4):
        output = engine.dispatch_moe(hidden, w1, w2, weights, ids, layer_id=layer, is_verify=True)

    sdd_stats = engine._sdd.get_statistics()
    print(f"  PASS: SDD dispatch (checks={sdd_stats['total_checks']}, freezes={sdd_stats['total_freezes']})")


def test_verify_lifecycle():
    """Test begin_verify → dispatch → end_verify cycle."""
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig
    import torch.nn.functional as F

    config = SpecMoEConfig(
        num_experts=8,
        top_k=2,
        hidden_size=16,
        moe_intermediate_size=8,
        num_moe_layers=2,
        enable_dedup=True,
        enable_sdd=False,
        enable_cache=False,
        hook_vllm=False,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()

    # Begin verify with K=3
    K = 3
    engine.begin_verify(K=K)

    # Simulate dispatch
    T = K + 1
    torch.manual_seed(0)
    hidden = torch.randn(T, 16)
    w1 = torch.randn(8, 16, 16) * 0.01
    w2 = torch.randn(8, 16, 8) * 0.01
    logits = torch.randn(T, 8)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    engine.dispatch_moe(hidden, w1, w2, weights, ids, layer_id=0, is_verify=True)

    # End verify: 2 of 3 draft tokens accepted
    engine.end_verify(accepted_tokens=2, proposed_tokens=3)

    stats = engine.get_statistics()
    assert stats["verify_rounds"] == 1
    assert stats["acceptance_rate"] == round(2 / 3, 4)
    assert stats["mean_accepted_length"] == 2.0
    assert stats["tokens_generated"] == 3  # 2 accepted + 1 bonus
    print(f"  PASS: verify lifecycle (acceptance={stats['acceptance_rate']}, tokens={stats['tokens_generated']})")


def test_multiple_verify_rounds():
    """Statistics should accumulate across multiple verify rounds."""
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig

    config = SpecMoEConfig(
        num_experts=8, top_k=2, hidden_size=16, moe_intermediate_size=8,
        num_moe_layers=2, enable_dedup=True, enable_sdd=False,
        enable_cache=False, hook_vllm=False,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()

    # Round 1: K=3, accept 2
    engine.begin_verify(K=3)
    engine.end_verify(accepted_tokens=2, proposed_tokens=3)

    # Round 2: K=4, accept 4
    engine.begin_verify(K=4)
    engine.end_verify(accepted_tokens=4, proposed_tokens=4)

    # Round 3: K=3, accept 0
    engine.begin_verify(K=3)
    engine.end_verify(accepted_tokens=0, proposed_tokens=3)

    stats = engine.get_statistics()
    assert stats["verify_rounds"] == 3
    assert stats["tokens_generated"] == (2 + 1) + (4 + 1) + (0 + 1)  # 9
    expected_rate = round(6 / 10, 4)
    assert stats["acceptance_rate"] == expected_rate
    print(f"  PASS: multi-round stats (rounds=3, tokens=9, rate={expected_rate})")


def test_engine_shutdown():
    """Shutdown should cleanly release resources."""
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig

    config = SpecMoEConfig(
        num_experts=8, top_k=2, hidden_size=16, moe_intermediate_size=8,
        num_moe_layers=2, hook_vllm=False,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()
    assert engine._initialized

    engine.shutdown()
    assert not engine._initialized
    print(f"  PASS: engine shutdown")


def test_create_specmoe_engine_helper():
    """Quick-start helper should create a working engine."""
    from adapters.specmoe_engine import create_specmoe_engine

    engine = create_specmoe_engine(
        num_experts=8,
        top_k=2,
        hidden_size=16,
        moe_intermediate_size=8,
        num_layers=2,
        gpu_cache_gb=0.01,
        enable_sdd=True,
        enable_trace=False,
        model=None,
    )

    assert engine._initialized
    engine.shutdown()
    print(f"  PASS: create_specmoe_engine helper")


def test_save_statistics():
    """Statistics should be saveable to JSON."""
    import json
    import tempfile
    from pathlib import Path
    from adapters.specmoe_engine import SpecMoEEngine, SpecMoEConfig

    config = SpecMoEConfig(
        num_experts=8, top_k=2, hidden_size=16, moe_intermediate_size=8,
        num_moe_layers=2, hook_vllm=False,
    )

    engine = SpecMoEEngine(config)
    engine.initialize()

    engine.begin_verify(K=3)
    engine.end_verify(accepted_tokens=2, proposed_tokens=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "stats.json")
        engine.save_statistics(path)

        with open(path) as f:
            data = json.load(f)

        assert data["verify_rounds"] == 1
        assert "dispatcher" in data
        assert "sdd" in data

    engine.shutdown()
    print(f"  PASS: save statistics to JSON")


def run_all():
    print("=" * 60)
    print("SpecMoE Engine Integration Tests")
    print("=" * 60)

    tests = [
        test_engine_initialization,
        test_manual_dispatch_without_sdd,
        test_manual_dispatch_with_sdd,
        test_verify_lifecycle,
        test_multiple_verify_rounds,
        test_engine_shutdown,
        test_create_specmoe_engine_helper,
        test_save_statistics,
    ]

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            print(f"\n[TEST] {name}")
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
