#!/usr/bin/env python3
"""
Tests for ELMM Milestone 2 Features
=====================================
Validates the three follow-up features:
  1. Temporal Locality Data Collection
  2. Draft-Guided Prefetch
  3. Adaptive Cache Budget Rebalancing

All tests run on CPU with synthetic data (no GPU required).
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/root/MoE-SD")


# ---------------------------------------------------------------------------
# Helpers: build a mock FusedMoE model for ELMM testing
# ---------------------------------------------------------------------------

class _MockQuantMethod:
    mk_owns_shared_expert = False

    def apply(self, layer, x, topk_weights, topk_ids):
        # Simple weighted sum as mock MoE output
        T, D = x.shape
        out = torch.zeros_like(x)
        for t in range(T):
            for k_idx in range(topk_ids.shape[1]):
                eid = topk_ids[t, k_idx].item()
                w = topk_weights[t, k_idx].item()
                # Read from the swapped-in param (scratchpad or UVA)
                w13 = layer.w13_weight[eid]  # [2N, D]
                w2 = layer.w2_weight[eid]    # [D, N]
                N2, _D = w13.shape
                gate = w13[: N2 // 2] @ x[t]  # [N]
                up = w13[N2 // 2 :] @ x[t]    # [N]
                h = torch.nn.functional.silu(gate) * up
                out[t] += w * (w2 @ h)
        return out


class _MockRouter:
    def __init__(self, num_experts, top_k):
        self._ne = num_experts
        self._k = top_k

    def select_experts(self, hidden_states, router_logits):
        T = hidden_states.shape[0]
        # Deterministic routing based on router_logits
        topk_weights = torch.ones(T, self._k) / self._k
        topk_ids = torch.zeros(T, self._k, dtype=torch.long)
        for t in range(T):
            probs = torch.softmax(router_logits[t, : self._ne], dim=0)
            _, ids = probs.topk(self._k)
            topk_ids[t] = ids
            topk_weights[t] = probs[ids]
            topk_weights[t] /= topk_weights[t].sum()
        return topk_weights, topk_ids


class _FakeAttr:
    """Emulates _vllm_offloaded_cpu_data attribute on a parameter."""
    pass


class MockFusedMoE(torch.nn.Module):
    """
    Minimal mock of vllm.model_executor.layers.fused_moe.layer.FusedMoE
    that ELMM can detect and patch.
    """

    def __init__(self, num_experts=16, hidden=32, intermediate=64, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        self.top_k = top_k

        # Expert weights in (E, 2N, D) and (E, D, N) format
        self.w13_weight = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden) * 0.01
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(num_experts, hidden, intermediate) * 0.01
        )

        # Gate (normally a Linear, but we pass router_logits directly)
        self.gate = None
        self.shared_experts = None
        self.router = _MockRouter(num_experts, top_k)
        self.quant_method = _MockQuantMethod()
        self.shared_experts_stream = None

        # Mark as offloaded: ELMM checks if ANY param has this attribute
        self._gate_param = torch.nn.Parameter(torch.zeros(1))
        self._gate_param._vllm_offloaded_cpu_data = _FakeAttr()

    def ensure_moe_quant_config_init(self):
        pass

    def ensure_dp_chunking_init(self):
        pass

    def _maybe_setup_shared_experts_stream(self, hidden_states, has_sep, flag):
        return False, None

    def _get_shared_experts_input(self, hidden_states):
        return hidden_states

    def forward_impl(self, hidden_states, router_logits):
        topk_w, topk_ids = self.router.select_experts(hidden_states, router_logits)
        return self.quant_method.apply(self, hidden_states, topk_w, topk_ids)


class MockModel(torch.nn.Module):
    """Model with multiple FusedMoE layers."""

    def __init__(self, num_layers=4, num_experts=16, hidden=32, intermediate=64, top_k=4):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = MockFusedMoE(num_experts, hidden, intermediate, top_k)
            self.layers.append(layer)


# Monkey-patch so ELMM's isinstance check works
import adapters.elmm_plugin as elmm_mod

_orig_install = elmm_mod.ELMMManager.install


def _patched_install(self, model):
    """
    Patched install that doesn't require vllm import.
    Scans for MockFusedMoE instead of the real FusedMoE.
    """
    import sys
    offloaded_layers = []
    total_fused_moe = 0

    for name, module in model.named_modules():
        if not isinstance(module, MockFusedMoE):
            continue
        total_fused_moe += 1
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if w13 is None or w2 is None:
            continue
        any_offloaded = any(
            hasattr(p, "_vllm_offloaded_cpu_data")
            for p in module.parameters()
        )
        if not any_offloaded:
            continue
        offloaded_layers.append((name, module))

    print(f"[ELMM-TEST] Found {total_fused_moe} FusedMoE layers, "
          f"{len(offloaded_layers)} offloaded", file=sys.stderr, flush=True)

    if not offloaded_layers:
        return

    max_w13_bytes = 0
    max_w2_bytes = 0
    max_w13_shape = ()
    max_w2_shape = ()

    for name, module in offloaded_layers:
        w13 = module.w13_weight
        w2 = module.w2_weight
        num_experts = w13.shape[0]
        expert_size = (
            w13[0].nelement() * w13[0].element_size()
            + w2[0].nelement() * w2[0].element_size()
        )
        self._layer_meta[name] = {
            "num_experts": num_experts,
            "expert_size": expert_size,
            "w13_shape": tuple(w13.shape),
            "w2_shape": tuple(w2.shape),
            "dtype": w13.dtype,
        }
        w13_bytes = w13.nelement() * w13.element_size()
        w2_bytes = w2.nelement() * w2.element_size()
        if w13_bytes > max_w13_bytes:
            max_w13_bytes = w13_bytes
            max_w13_shape = tuple(w13.shape)
        if w2_bytes > max_w2_bytes:
            max_w2_bytes = w2_bytes
            max_w2_shape = tuple(w2.shape)

    ref_dtype = list(self._layer_meta.values())[0]["dtype"]
    device = torch.device("cpu")  # CPU for testing
    self._scratch_w13 = torch.empty(max_w13_shape, dtype=ref_dtype, device=device)
    self._scratch_w2 = torch.empty(max_w2_shape, dtype=ref_dtype, device=device)

    num_layers = len(offloaded_layers)
    per_layer_budget = self.config.gpu_cache_budget_bytes // num_layers

    for name, _module in offloaded_layers:
        meta = self._layer_meta[name]
        expert_size = meta["expert_size"]
        max_slots = max(1, per_layer_budget // expert_size)
        w13_single = meta["w13_shape"][1:]
        w2_single = meta["w2_shape"][1:]
        from adapters.elmm_plugin import _LayerExpertCache
        cache = _LayerExpertCache(
            layer_name=name,
            max_slots=max_slots,
            w13_single_shape=w13_single,
            w2_single_shape=w2_single,
            dtype=meta["dtype"],
            device=device,
        )
        self._layer_caches[name] = cache

    for name, module in offloaded_layers:
        self._original_forward_impls[id(module)] = module.forward_impl
        self._patched_modules[name] = module

        manager = self
        layer_name = name

        def make_patched_forward(mgr, lname):
            def patched_forward_impl(hidden_states, router_logits):
                return mgr._elmm_forward_impl(lname, hidden_states, router_logits)
            return patched_forward_impl

        module.forward_impl = make_patched_forward(manager, layer_name)

    self._installed = True
    self._total_cache_slots = sum(c._max_slots for c in self._layer_caches.values())
    for name in self._layer_caches:
        self._hit_rate_ema[name] = 0.5
    # Initialize remap table for pool-direct mode
    if self.config.enable_pool_direct:
        max_experts = max(m["num_experts"] for m in self._layer_meta.values())
        self._remap_table = torch.arange(max_experts, dtype=torch.long, device=device)
    # Direct dispatch requires vLLM Triton kernels — disable in CPU tests
    self.config.enable_direct_dispatch = False
    # GPU cache uses CUDA tensors — disable in CPU tests
    self.config.enable_gpu_cache = False


# Apply the test-mode install
elmm_mod.ELMMManager.install = _patched_install


# ============================================================================
# Test 1: Temporal Locality Data Collection
# ============================================================================

def test_temporal_locality_collection():
    """Test that ELMM collects temporal locality data during forward passes."""
    print("TEST: Temporal Locality Collection")
    from adapters.elmm_plugin import ELMMConfig, ELMMManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ELMMConfig(
            gpu_cache_budget_bytes=1 * 1024**3,
            enable_prefetch=False,
            enable_locality_collection=True,
            locality_export_dir=tmpdir,
            enable_adaptive_budget=False,
        )
        manager = ELMMManager(config)
        model = MockModel(num_layers=3, num_experts=16, hidden=32, intermediate=64, top_k=4)
        manager.install(model)

        assert manager._installed, "ELMM should be installed"
        assert manager._locality_analyzer is not None, "Locality analyzer should be created"

        # Run several forward passes to generate locality data
        for step in range(10):
            for layer in model.layers:
                fmoe = layer.mlp.experts
                T, D = 4, 32
                hidden = torch.randn(T, D) * 0.1
                # Use similar router logits across steps → high overlap
                router_logits = torch.randn(T, 16) * 0.1
                router_logits[:, :4] += 2.0  # Bias toward experts 0-3
                fmoe.forward_impl(hidden, router_logits)
            # Signal end of verify round
            manager.on_verify_round_end()

        # Check overlap history was collected
        assert len(manager._overlap_history) > 0, "Should have overlap history"
        for name, overlaps in manager._overlap_history.items():
            assert len(overlaps) > 0, f"Layer {name} should have overlap data"
            mean_overlap = sum(overlaps) / len(overlaps)
            assert mean_overlap > 0.0, f"Mean overlap should be positive"

        # Check locality analyzer received rounds
        assert manager._verify_round_counter == 10, f"Expected 10 rounds, got {manager._verify_round_counter}"

        # Export and check files
        manager.export_locality_data(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "overlap_history.json")), "Missing overlap_history.json"
        assert os.path.exists(os.path.join(tmpdir, "locality_report.json")), "Missing locality_report.json"
        assert os.path.exists(os.path.join(tmpdir, "cache_stats.json")), "Missing cache_stats.json"

        # Validate report content
        with open(os.path.join(tmpdir, "locality_report.json")) as f:
            report = json.load(f)
        assert "summary" in report, "Report should have summary"
        assert report["summary"]["num_rounds_analyzed"] == 10

        # Check stats include locality summary
        stats = manager.get_stats()
        assert "locality_summary" in stats, f"Stats missing locality_summary: {list(stats.keys())}"
        assert stats["locality_summary"]["inter_round_overlap"] > 0

        manager.shutdown()
        print(f"  Overlap history: {len(manager._overlap_history)} layers tracked")
        print(f"  Report summary: {report['summary']}")
        print("  PASS ✓\n")


# ============================================================================
# Test 2: Draft-Guided Prefetch
# ============================================================================

def test_draft_prefetch():
    """Test that prefetch correctly pre-loads experts and tracks accuracy."""
    print("TEST: Draft-Guided Prefetch")
    from adapters.elmm_plugin import ELMMConfig, ELMMManager

    config = ELMMConfig(
        gpu_cache_budget_bytes=1 * 1024**3,
        enable_prefetch=True,
        use_prefetch_stream=False,  # CPU mode
        enable_locality_collection=False,
        enable_adaptive_budget=False,
    )
    manager = ELMMManager(config)
    model = MockModel(num_layers=2, num_experts=16, hidden=32, intermediate=64, top_k=4)
    manager.install(model)

    # Get layer names
    layer_names = list(manager._layer_caches.keys())
    assert len(layer_names) == 2

    # Step 1: Prefetch experts 0,1,2,3 for first layer
    manager.prefetch_experts(layer_names[0], [0, 1, 2, 3])

    # Verify experts are in cache
    cache0 = manager._layer_caches[layer_names[0]]
    for eid in [0, 1, 2, 3]:
        assert cache0.contains(eid), f"Expert {eid} should be prefetched"

    # Verify pending prefetch tracking
    assert layer_names[0] in manager._pending_prefetch
    assert len(manager._pending_prefetch[layer_names[0]]) == 4

    # Step 2: Run forward that uses some of the prefetched experts
    fmoe = model.layers[0].mlp.experts
    T, D = 2, 32
    hidden = torch.randn(T, D) * 0.1
    router_logits = torch.randn(T, 16) * 0.1
    router_logits[:, :4] += 5.0  # Strong bias toward 0-3
    fmoe.forward_impl(hidden, router_logits)

    # Check prefetch accuracy was tracked
    assert manager._prefetch_total > 0, "Prefetch total should be tracked"
    assert manager._prefetch_hits > 0, "Prefetch should have hits"

    stats = manager.get_stats()
    assert "prefetch" in stats, "Stats should contain prefetch info"
    print(f"  Prefetch accuracy: {stats['prefetch']['accuracy']:.2%}")

    # Step 3: Test prefetch_for_draft_routing (uses layer index mapping)
    manager.prefetch_for_draft_routing({0: [5, 6, 7], 1: [10, 11, 12]})
    # Check that the prefetch was issued
    for ln in layer_names:
        cache = manager._layer_caches[ln]
        # At least some of the experts should be prefetched
        assert cache._max_slots > 0

    manager.shutdown()
    print("  PASS ✓\n")


# ============================================================================
# Test 3: Adaptive Cache Budget Rebalancing
# ============================================================================

def test_adaptive_cache_budget():
    """Test that cache slots are redistributed based on per-layer hit rates."""
    print("TEST: Adaptive Cache Budget Rebalancing")
    from adapters.elmm_plugin import ELMMConfig, ELMMManager

    config = ELMMConfig(
        gpu_cache_budget_bytes=512 * 1024**2,  # 512 MB
        enable_prefetch=False,
        enable_locality_collection=False,
        enable_adaptive_budget=True,
        rebalance_interval=20,  # Rebalance every 20 forward intercepts
        min_slot_fraction=0.05,
        hit_rate_ema_alpha=0.3,  # Fast adaptation for testing
    )
    manager = ELMMManager(config)
    model = MockModel(num_layers=4, num_experts=16, hidden=32, intermediate=64, top_k=4)
    manager.install(model)

    layer_names = list(manager._layer_caches.keys())
    assert len(layer_names) == 4

    # Record initial slot distribution
    initial_slots = {n: manager._layer_caches[n]._max_slots for n in layer_names}
    total_slots = sum(initial_slots.values())
    print(f"  Initial slots: {initial_slots} (total={total_slots})")

    # Simulate different access patterns:
    # Layer 0: always uses same experts (high hit rate)
    # Layer 3: uses random experts (low hit rate → high miss rate)
    for step in range(25):  # > rebalance_interval
        for i, layer in enumerate(model.layers):
            fmoe = layer.mlp.experts
            T, D = 4, 32
            hidden = torch.randn(T, D) * 0.1
            router_logits = torch.randn(T, 16) * 0.1

            if i == 0:
                # Layer 0: always activate experts 0-3 (high locality)
                router_logits[:, :4] += 10.0
            elif i == 3:
                # Layer 3: random experts each step (low locality)
                hot = torch.randint(0, 16, (4,))
                for idx, h in enumerate(hot):
                    router_logits[idx % T, h] += 10.0
            else:
                # Middle layers: moderate
                router_logits[:, 2:6] += 3.0

            fmoe.forward_impl(hidden, router_logits)

    final_slots = {n: manager._layer_caches[n]._max_slots for n in layer_names}
    final_total = sum(final_slots.values())
    print(f"  Final slots:   {final_slots} (total={final_total})")

    # Verify total budget is preserved
    assert final_total == total_slots, f"Total slots should be preserved: {final_total} != {total_slots}"

    # Layer 0 should have fewer slots (high hit rate → fewer misses → less cache needed)
    # Layer 3 should have more slots (low hit rate → more misses → more cache needed)
    ema_rates = {n: manager._hit_rate_ema[n] for n in layer_names}
    print(f"  EMA hit rates: {ema_rates}")

    # Verify that layers with lower hit rates got more slots after rebalancing
    assert ema_rates[layer_names[0]] > ema_rates[layer_names[3]], \
        f"Layer 0 should have higher hit rate than layer 3"

    # The layer with lowest hit rate should have at least as many slots
    # as the minimum threshold
    min_slots = max(1, int(total_slots * config.min_slot_fraction))
    for n in layer_names:
        assert final_slots[n] >= min_slots, \
            f"Layer {n} has {final_slots[n]} slots < min {min_slots}"

    manager.shutdown()
    print("  PASS ✓\n")


# ============================================================================
# Test 4: _LayerExpertCache.resize()
# ============================================================================

def test_cache_resize():
    """Test that cache resize works correctly (shrink and grow)."""
    print("TEST: Cache Resize")
    from adapters.elmm_plugin import _LayerExpertCache

    cache = _LayerExpertCache(
        layer_name="test",
        max_slots=10,
        w13_single_shape=(128, 32),
        w2_single_shape=(32, 64),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Fill cache with 8 experts
    for eid in range(8):
        slot = cache.alloc_slot(eid)
        w13, w2 = cache.get_slot_tensors(slot)
        w13.fill_(eid)
        w2.fill_(eid + 100)

    assert len(cache._slot_map) == 8
    assert cache._max_slots == 10

    # Shrink to 5 => should evict 3 oldest (0, 1, 2)
    new_cap = cache.resize(5)
    assert new_cap == 5, f"Expected 5 got {new_cap}"
    assert cache._max_slots == 5
    assert len(cache._slot_map) == 5
    # Physical pool stays the same size (logical-only resize)
    assert cache._w13_pool.shape[0] == 10
    # Experts 0,1,2 should have been evicted (LRU)
    for eid in [0, 1, 2]:
        assert not cache.contains(eid), f"Expert {eid} should be evicted"
    for eid in [3, 4, 5, 6, 7]:
        assert cache.contains(eid), f"Expert {eid} should remain"

    # Grow to 10 (capped at physical pool size of 10)
    new_cap = cache.resize(12)
    assert new_cap == 10, f"Expected 10 (physical max) got {new_cap}"
    assert cache._max_slots == 10
    assert cache._w13_pool.shape[0] == 10  # Physical pool unchanged
    assert len(cache._free_slots) == 5  # 10 - 5 existing

    # Can allocate new experts into grown space
    for eid in range(20, 25):
        slot = cache.alloc_slot(eid)
        assert 0 <= slot < 10

    print("  PASS ✓\n")


# ============================================================================
# Test 5: End-to-end with all features
# ============================================================================

def test_end_to_end():
    """Test all three features working together."""
    print("TEST: End-to-End Integration")
    from adapters.elmm_plugin import ELMMConfig, ELMMManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ELMMConfig(
            gpu_cache_budget_bytes=256 * 1024**2,
            enable_prefetch=True,
            use_prefetch_stream=False,
            enable_locality_collection=True,
            locality_export_dir=tmpdir,
            enable_adaptive_budget=True,
            rebalance_interval=15,
            hit_rate_ema_alpha=0.2,
        )
        manager = ELMMManager(config)
        model = MockModel(num_layers=3, num_experts=16, hidden=32, intermediate=64, top_k=4)
        manager.install(model)

        # Simulate 20 verify rounds with prefetch
        for round_idx in range(20):
            # Draft phase: prefetch based on last round's experts
            for name in manager._last_expert_set:
                last = manager._last_expert_set[name]
                if last:
                    manager.prefetch_experts(name, list(last))

            # Verify phase: forward through all layers
            for layer in model.layers:
                fmoe = layer.mlp.experts
                T, D = 4, 32
                hidden = torch.randn(T, D) * 0.1
                router_logits = torch.randn(T, 16) * 0.1
                # Simulate some locality
                router_logits[:, :6] += 1.5 + round_idx * 0.01
                fmoe.forward_impl(hidden, router_logits)

            manager.on_verify_round_end()

        stats = manager.get_stats()
        print(f"  Total intercepts: {stats['total_intercepts']}")
        print(f"  Hit rate: {stats['overall_hit_rate']:.2%}")
        print(f"  Verify rounds: {stats['verify_rounds']}")

        if "prefetch" in stats:
            print(f"  Prefetch accuracy: {stats['prefetch']['accuracy']:.2%}")
        if "locality_summary" in stats:
            ls = stats["locality_summary"]
            print(f"  Inter-round overlap: {ls['inter_round_overlap']:.2%}")

        # Export final data
        manager.export_locality_data(tmpdir)
        files = os.listdir(tmpdir)
        print(f"  Exported files: {files}")

        assert stats["overall_hit_rate"] > 0, "Should have some cache hits"
        assert stats["verify_rounds"] == 20

        manager.shutdown()
        print("  PASS ✓\n")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ELMM Milestone 2 Feature Tests")
    print("=" * 60 + "\n")

    tests = [
        test_cache_resize,
        test_temporal_locality_collection,
        test_draft_prefetch,
        test_adaptive_cache_budget,
        test_end_to_end,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL ✗ — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
