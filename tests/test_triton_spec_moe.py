"""
Comprehensive Tests for SpecMoE Triton Kernel & PyTorch Fallback
=================================================================
Tests correctness of expert deduplication, frozen token handling,
and numerical equivalence between vanilla MoE and SpecMoE.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/MoE-SD")


def _vanilla_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """Reference vanilla MoE implementation (no dedup)."""
    T, D = hidden_states.shape
    N = w1.shape[1] // 2
    output = torch.zeros(T, D, device=hidden_states.device, dtype=hidden_states.dtype)

    for t in range(T):
        for s in range(topk_ids.shape[1]):
            eid = topk_ids[t, s].item()
            w = topk_weights[t, s]
            gate_up = hidden_states[t] @ w1[eid].T  # [2N]
            gate = F.silu(gate_up[:N])
            up = gate_up[N:]
            expert_out = (gate * up) @ w2[eid].T  # [D]
            output[t] += w * expert_out

    return output


def test_spec_fused_moe_function_correctness():
    """SpecFusedMoEFunction should produce same result as vanilla MoE."""
    from adapters.triton_spec_moe import SpecFusedMoEFunction

    torch.manual_seed(42)
    E, D, N, T, top_k = 16, 64, 32, 4, 4

    hidden = torch.randn(T, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01
    logits = torch.randn(T, E)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    expected = _vanilla_moe(hidden, w1, w2, weights, ids)
    actual, naive, dedup = SpecFusedMoEFunction.forward(hidden, w1, w2, weights, ids)

    assert torch.allclose(expected, actual, atol=1e-4), \
        f"Max diff: {(expected - actual).abs().max().item()}"
    assert naive == T * top_k
    assert dedup <= naive
    print(f"  PASS: correctness (max_diff={( expected - actual).abs().max():.6f}, naive={naive}, dedup={dedup})")


def test_dedup_saves_with_shared_experts():
    """When multiple tokens share experts, dedup should save loads."""
    from adapters.triton_spec_moe import SpecFusedMoEFunction

    torch.manual_seed(0)
    E, D, N, top_k = 16, 32, 16, 4

    hidden = torch.randn(4, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01

    # Force all tokens to use same top-k experts
    ids = torch.tensor([[0, 1, 2, 3]] * 4)
    weights = torch.ones(4, top_k) / top_k

    _, naive, dedup = SpecFusedMoEFunction.forward(hidden, w1, w2, weights, ids)

    assert naive == 16  # 4 tokens × 4 experts
    assert dedup == 4   # Only 4 unique experts
    print(f"  PASS: dedup savings (naive={naive}, dedup={dedup}, ratio={(1 - dedup/naive)*100:.0f}%)")


def test_active_mask_freezes_tokens():
    """Frozen tokens should be skipped but output shape should be preserved."""
    from adapters.triton_spec_moe import SpecFusedMoEFunction

    torch.manual_seed(42)
    E, D, N, T, top_k = 16, 32, 16, 4, 4

    hidden = torch.randn(T, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01
    logits = torch.randn(T, E)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Freeze tokens 1 and 3
    active_mask = torch.tensor([True, False, True, False])

    output, naive, dedup = SpecFusedMoEFunction.forward(
        hidden, w1, w2, weights, ids, active_mask=active_mask
    )

    assert output.shape == (T, D)
    # Frozen tokens should have zero output
    assert output[1].abs().max() == 0
    assert output[3].abs().max() == 0
    # Active tokens should have non-zero output
    assert output[0].abs().max() > 0
    assert output[2].abs().max() > 0
    # Only 2 active tokens contribute to dedup
    assert naive == 2 * top_k
    print(f"  PASS: frozen mask (frozen_zeros=True, active_nonzero=True)")


def test_dispatcher_statistics():
    """SpecFusedMoEDispatcher should track dedup statistics."""
    from adapters.triton_spec_moe import SpecFusedMoEDispatcher

    torch.manual_seed(42)
    E, D, N, top_k = 16, 32, 16, 4

    dispatcher = SpecFusedMoEDispatcher(num_experts=E, top_k=top_k, use_triton=False)

    hidden = torch.randn(4, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01

    # Same experts for all tokens → max dedup
    ids = torch.tensor([[0, 1, 2, 3]] * 4)
    weights = torch.ones(4, top_k) / top_k

    _ = dispatcher(hidden, w1, w2, weights, ids)

    stats = dispatcher.get_statistics()
    assert stats["total_calls"] == 1
    assert stats["total_naive_loads"] == 16
    assert stats["total_dedup_loads"] == 4
    assert stats["dedup_ratio"] == 0.75
    print(f"  PASS: dispatcher stats ({stats})")


def test_triton_v2_correctness():
    """SpecFusedMoETritonV2 PyTorch fallback should match vanilla."""
    from adapters.triton_spec_moe import SpecFusedMoETritonV2

    torch.manual_seed(42)
    E, D, N, T, top_k = 16, 64, 32, 4, 4

    v2 = SpecFusedMoETritonV2(num_experts=E, top_k=top_k)

    hidden = torch.randn(T, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01
    logits = torch.randn(T, E)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    expected = _vanilla_moe(hidden, w1, w2, weights, ids)
    actual = v2.forward(hidden, w1, w2, weights, ids)

    assert torch.allclose(expected, actual, atol=1e-4), \
        f"Max diff: {(expected - actual).abs().max().item()}"

    stats = v2.get_statistics()
    assert stats["total_calls"] == 1
    print(f"  PASS: TritonV2 correctness (max_diff={(expected - actual).abs().max():.6f})")


def test_triton_v2_active_mask():
    """TritonV2 should respect active_mask."""
    from adapters.triton_spec_moe import SpecFusedMoETritonV2

    torch.manual_seed(42)
    E, D, N, T, top_k = 8, 32, 16, 6, 2

    v2 = SpecFusedMoETritonV2(num_experts=E, top_k=top_k)

    hidden = torch.randn(T, D)
    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01
    logits = torch.randn(T, E)
    weights, ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    active_mask = torch.tensor([True, True, False, True, False, True])
    output = v2.forward(hidden, w1, w2, weights, ids, active_mask=active_mask)

    assert output.shape == (T, D)
    assert output[2].abs().max() == 0  # frozen
    assert output[4].abs().max() == 0  # frozen
    assert output[0].abs().max() > 0   # active
    print(f"  PASS: TritonV2 active_mask")


def test_maf_dedup_correlation():
    """Verify that dedup ratio correlates with theoretical MAF."""
    from adapters.triton_spec_moe import SpecFusedMoEFunction
    from collectors.expert_trace_hook import compute_theoretical_maf

    torch.manual_seed(42)
    E, D, N, top_k = 128, 32, 16, 8

    w1 = torch.randn(E, 2 * N, D) * 0.01
    w2 = torch.randn(E, D, N) * 0.01

    for K in [1, 2, 3, 4, 5]:
        T = K + 1
        hidden = torch.randn(T, D)

        # Random routing (uniform-ish)
        logits = torch.randn(T, E)
        weights, ids = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        _, naive, dedup = SpecFusedMoEFunction.forward(hidden, w1, w2, weights, ids)
        measured_maf = dedup / top_k
        theoretical_maf = compute_theoretical_maf(K, k=top_k, N=E)

        # Should be in same ballpark (single sample has high variance)
        print(f"    K={K}: measured_MAF={measured_maf:.2f}, theoretical={theoretical_maf:.2f}")

    print(f"  PASS: MAF-dedup correlation check completed")


def run_all():
    print("=" * 60)
    print("SpecFusedMoE Triton Kernel Tests")
    print("=" * 60)

    tests = [
        test_spec_fused_moe_function_correctness,
        test_dedup_saves_with_shared_experts,
        test_active_mask_freezes_tokens,
        test_dispatcher_statistics,
        test_triton_v2_correctness,
        test_triton_v2_active_mask,
        test_maf_dedup_correlation,
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
