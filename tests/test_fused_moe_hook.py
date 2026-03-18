"""
Tests for FusedMoE Hook
========================
Tests the hook mechanism without requiring actual vLLM installation.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/MoE-SD")


def test_hook_creation():
    """Hook should be creatable without vLLM."""
    from adapters.fused_moe_hook import FusedMoEHook

    hook = FusedMoEHook()
    assert not hook._installed
    assert not hook._verify_mode

    stats = hook.get_statistics()
    assert stats["installed"] is False
    assert stats["total_intercepts"] == 0
    print(f"  PASS: hook creation")


def test_hook_configure():
    """Configure should accept SpecMoE components."""
    from adapters.fused_moe_hook import FusedMoEHook
    from adapters.triton_spec_moe import SpecFusedMoEDispatcher
    from adapters.layer_early_terminator import SpeculationDivergenceDetector

    hook = FusedMoEHook()
    dispatcher = SpecFusedMoEDispatcher(num_experts=8, top_k=2, use_triton=False)
    sdd = SpeculationDivergenceDetector(num_layers=4)

    hook.configure(spec_moe=dispatcher, sdd=sdd)

    assert hook._spec_moe is dispatcher
    assert hook._sdd is sdd
    print(f"  PASS: hook configure")


def test_verify_context_manager():
    """verify_context should properly set/unset verify mode."""
    from adapters.fused_moe_hook import FusedMoEHook

    hook = FusedMoEHook()

    assert not hook._verify_mode

    with hook.verify_context(batch_size=4):
        assert hook._verify_mode
        assert hook._verify_batch_size == 4

    assert not hook._verify_mode
    print(f"  PASS: verify context manager")


def test_hook_install_without_vllm():
    """Install should return False when vLLM is not available."""
    from adapters.fused_moe_hook import FusedMoEHook

    hook = FusedMoEHook()
    result = hook.install()
    # vLLM not installed → should fail gracefully
    assert result is False
    assert not hook._installed
    print(f"  PASS: install without vLLM (graceful failure)")


def test_global_hook_singleton():
    """get_hook should return same instance."""
    from adapters.fused_moe_hook import get_hook

    hook1 = get_hook()
    hook2 = get_hook()
    assert hook1 is hook2
    print(f"  PASS: global hook singleton")


def test_hook_statistics():
    """Statistics should be tracked correctly."""
    from adapters.fused_moe_hook import FusedMoEHook
    from adapters.triton_spec_moe import SpecFusedMoEDispatcher

    hook = FusedMoEHook()
    dispatcher = SpecFusedMoEDispatcher(num_experts=8, top_k=2, use_triton=False)
    hook.configure(spec_moe=dispatcher)

    stats = hook.get_statistics()
    assert stats["total_intercepts"] == 0
    assert "spec_moe" in stats
    assert stats["spec_moe"]["total_calls"] == 0
    print(f"  PASS: hook statistics tracking")


def test_set_verify_mode_with_mask():
    """Setting verify mode with active_mask should work."""
    from adapters.fused_moe_hook import FusedMoEHook
    from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

    hook = FusedMoEHook()
    sdd = SpeculationDivergenceDetector(
        config=SDDConfig(min_check_layer=0),
        num_layers=4,
    )
    hook.configure(sdd=sdd)

    mask = torch.tensor([True, True, False, True])
    hook.set_verify_mode(enabled=True, batch_size=4, active_mask=mask)

    assert hook._verify_mode
    assert hook._active_mask is mask
    assert hook._verify_batch_size == 4

    hook.set_verify_mode(enabled=False)
    assert not hook._verify_mode
    print(f"  PASS: verify mode with mask")


def run_all():
    print("=" * 60)
    print("FusedMoE Hook Tests")
    print("=" * 60)

    tests = [
        test_hook_creation,
        test_hook_configure,
        test_verify_context_manager,
        test_hook_install_without_vllm,
        test_global_hook_singleton,
        test_hook_statistics,
        test_set_verify_mode_with_mask,
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
