"""
Tests for Expert Weight Cache (GPU/CPU tiered, LRU, prefetch)
==============================================================
Tests cache operations on CPU (since GPU may not be available).
"""

import sys
import time
import torch

sys.path.insert(0, "/root/MoE-SD")


def _make_expert_weights(E=16, N=16, D=32):
    """Create test expert weights."""
    w1 = torch.randn(E, 2 * N, D) * 0.01  # [E, 2N, D]
    w2 = torch.randn(E, D, N) * 0.01       # [E, D, N]
    return w1, w2


def test_register_and_retrieve():
    """Register experts, then retrieve them."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    config = ExpertCacheConfig(
        gpu_budget_bytes=100 * 1024**2,  # 100 MB
        pin_cpu_memory=False,
    )
    cache = ExpertWeightCache(config, device="cpu")

    w1, w2 = _make_expert_weights(E=8, N=16, D=32)
    cache.register_experts(layer_id=0, w1=w1, w2=w2)

    # Retrieve expert 0
    result = cache.get_expert(0, 0)
    assert "w1" in result and "w2" in result
    assert result["w1"].shape == w1[0].shape
    assert result["w2"].shape == w2[0].shape

    stats = cache.get_statistics()
    assert stats["hits"] == 0  # First access is a miss
    assert stats["misses"] == 1
    print(f"  PASS: register and retrieve (stats={stats})")


def test_cache_hits():
    """Second access should be a hit."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    config = ExpertCacheConfig(gpu_budget_bytes=100 * 1024**2, pin_cpu_memory=False)
    cache = ExpertWeightCache(config, device="cpu")

    w1, w2 = _make_expert_weights(E=8)
    cache.register_experts(0, w1, w2)

    # First access = miss
    cache.get_expert(0, 3)
    # Second access = hit
    cache.get_expert(0, 3)

    stats = cache.get_statistics()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    print(f"  PASS: cache hits (hit_rate={stats['hit_rate']})")


def test_lru_eviction():
    """When cache is full, LRU should evict oldest entry."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    E, N, D = 8, 16, 32
    w1, w2 = _make_expert_weights(E=E, N=N, D=D)

    # Expert size: w1[0] = 2*16*32*4 = 4096 bytes, w2[0] = 32*16*4 = 2048 → ~6144 bytes
    expert_bytes = w1[0].nelement() * w1[0].element_size() + w2[0].nelement() * w2[0].element_size()

    # Budget for exactly 3 experts
    config = ExpertCacheConfig(
        gpu_budget_bytes=expert_bytes * 3,
        pin_cpu_memory=False,
    )
    cache = ExpertWeightCache(config, device="cpu")
    cache.register_experts(0, w1, w2)

    # Load 3 experts → cache should be full
    cache.get_expert(0, 0)
    cache.get_expert(0, 1)
    cache.get_expert(0, 2)

    assert len(cache._gpu_cache) == 3

    # Load 4th → should evict expert 0 (LRU)
    cache.get_expert(0, 3)

    assert len(cache._gpu_cache) == 3
    assert not cache.is_cached(0, 0)   # evicted
    assert cache.is_cached(0, 1)
    assert cache.is_cached(0, 2)
    assert cache.is_cached(0, 3)       # newly loaded

    stats = cache.get_statistics()
    assert stats["evictions"] == 1
    print(f"  PASS: LRU eviction (evictions={stats['evictions']})")


def test_lru_touch_prevents_eviction():
    """Accessing an entry should move it to MRU position."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    E, N, D = 8, 16, 32
    w1, w2 = _make_expert_weights(E=E, N=N, D=D)
    expert_bytes = w1[0].nelement() * w1[0].element_size() + w2[0].nelement() * w2[0].element_size()

    config = ExpertCacheConfig(
        gpu_budget_bytes=expert_bytes * 3,
        pin_cpu_memory=False,
    )
    cache = ExpertWeightCache(config, device="cpu")
    cache.register_experts(0, w1, w2)

    # Load 0, 1, 2
    cache.get_expert(0, 0)
    cache.get_expert(0, 1)
    cache.get_expert(0, 2)

    # Touch expert 0 → moves to MRU
    cache.get_expert(0, 0)

    # Load expert 3 → should evict expert 1 (now LRU)
    cache.get_expert(0, 3)

    assert cache.is_cached(0, 0)    # kept (was touched)
    assert not cache.is_cached(0, 1)  # evicted
    assert cache.is_cached(0, 2)
    assert cache.is_cached(0, 3)
    print(f"  PASS: LRU touch prevents eviction")


def test_batch_retrieve():
    """Batch retrieval should work and track stats correctly."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    config = ExpertCacheConfig(gpu_budget_bytes=100 * 1024**2, pin_cpu_memory=False)
    cache = ExpertWeightCache(config, device="cpu")

    w1, w2 = _make_expert_weights(E=16)
    cache.register_experts(0, w1, w2)

    # Pre-load some
    cache.get_expert(0, 5)

    # Batch retrieve: 5 is cached, 6 and 7 are misses
    results = cache.get_experts_batch(0, [5, 6, 7])

    assert 5 in results and 6 in results and 7 in results
    stats = cache.get_statistics()
    # Total: 1 (get_expert) + 3 (batch) = 4 lookups
    # Hits: 1 (expert 5 in batch), Misses: 3 (initial + 2 batch misses)
    assert stats["hits"] == 1
    assert stats["misses"] == 3
    print(f"  PASS: batch retrieve (hits={stats['hits']}, misses={stats['misses']})")


def test_prefetch_scheduler():
    """PrefetchScheduler should predict and track accuracy."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig, PrefetchScheduler

    config = ExpertCacheConfig(gpu_budget_bytes=100 * 1024**2, pin_cpu_memory=False)
    cache = ExpertWeightCache(config, device="cpu")

    w1, w2 = _make_expert_weights(E=16)
    cache.register_experts(0, w1, w2)

    scheduler = PrefetchScheduler(cache, num_layers=1)

    # Predict experts 0, 1, 2, 3 for layer 0
    scheduler.on_draft_routing(layer_id=0, expert_ids=[0, 1, 2, 3])

    # Prepare verify (sync)
    scheduler.prepare_verify(K=3)

    # Actually used: experts 0, 1, 5 → predicted 0, 1 correct; 2, 3 wasted; 5 missed
    scheduler.report_verify_result({0: [0, 1, 5]})

    stats = scheduler.get_statistics()
    assert stats["total_predicted"] == 4
    assert stats["total_correct"] == 2  # Experts 0 and 1 were correctly predicted
    assert stats["prefetch_accuracy"] == 0.5
    print(f"  PASS: prefetch scheduler (accuracy={stats['prefetch_accuracy']})")


def test_frequency_eviction():
    """Frequency-based eviction should evict least-used expert."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    E, N, D = 8, 16, 32
    w1, w2 = _make_expert_weights(E=E, N=N, D=D)
    expert_bytes = w1[0].nelement() * w1[0].element_size() + w2[0].nelement() * w2[0].element_size()

    config = ExpertCacheConfig(
        gpu_budget_bytes=expert_bytes * 3,
        pin_cpu_memory=False,
        eviction_policy="frequency",
    )
    cache = ExpertWeightCache(config, device="cpu")
    cache.register_experts(0, w1, w2)

    # Load 0, 1, 2
    cache.get_expert(0, 0)  # access count = 1
    cache.get_expert(0, 1)  # access count = 1
    cache.get_expert(0, 2)  # access count = 1

    # Access expert 0 and 2 more → higher frequency
    cache.get_expert(0, 0)  # count = 2
    cache.get_expert(0, 0)  # count = 3
    cache.get_expert(0, 2)  # count = 2

    # Load expert 3 → should evict expert 1 (least frequently accessed)
    cache.get_expert(0, 3)

    assert not cache.is_cached(0, 1)  # evicted (lowest frequency)
    assert cache.is_cached(0, 0)      # kept (highest frequency)
    assert cache.is_cached(0, 2)
    assert cache.is_cached(0, 3)
    print(f"  PASS: frequency-based eviction")


def test_cache_occupancy():
    """Cache occupancy should be tracked correctly."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    config = ExpertCacheConfig(gpu_budget_bytes=100 * 1024**2, pin_cpu_memory=False)
    cache = ExpertWeightCache(config, device="cpu")

    w1, w2 = _make_expert_weights(E=16)
    cache.register_experts(0, w1, w2)
    cache.register_experts(1, w1, w2)

    occ = cache.get_cache_occupancy()
    assert occ["registered_experts"] == 32  # 16 × 2 layers
    assert occ["cached_experts"] == 0

    cache.get_expert(0, 0)
    cache.get_expert(1, 5)

    occ = cache.get_cache_occupancy()
    assert occ["cached_experts"] == 2
    assert occ["utilization"] > 0
    print(f"  PASS: cache occupancy (registered={occ['registered_experts']}, cached={occ['cached_experts']})")


def test_clear_cache():
    """clear_gpu_cache should empty the GPU cache."""
    from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig

    config = ExpertCacheConfig(gpu_budget_bytes=100 * 1024**2, pin_cpu_memory=False)
    cache = ExpertWeightCache(config, device="cpu")

    w1, w2 = _make_expert_weights()
    cache.register_experts(0, w1, w2)

    cache.get_expert(0, 0)
    cache.get_expert(0, 1)
    assert len(cache._gpu_cache) == 2

    cache.clear_gpu_cache()
    assert len(cache._gpu_cache) == 0
    assert cache._gpu_cache_bytes == 0
    print(f"  PASS: clear_gpu_cache")


def run_all():
    print("=" * 60)
    print("Expert Weight Cache Tests")
    print("=" * 60)

    tests = [
        test_register_and_retrieve,
        test_cache_hits,
        test_lru_eviction,
        test_lru_touch_prevents_eviction,
        test_batch_retrieve,
        test_prefetch_scheduler,
        test_frequency_eviction,
        test_cache_occupancy,
        test_clear_cache,
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
