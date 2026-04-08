"""
Tests for BriskMoECache unified facade (ISSUE-004).
Covers: access_expert flow, eviction via ELP+SACR, verify callback,
         DIPP draft callback, full SD cycle simulation, ablation checks.
"""

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Import helpers — bypass adapters/__init__.py torch dependency
# ---------------------------------------------------------------------------

_ADAPTERS = os.path.join(os.path.dirname(__file__), "..", "adapters")


def _load(name, filename):
    path = os.path.join(_ADAPTERS, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
_tracker_mod = _load("adapters.accept_reject_tracker", "accept_reject_tracker.py")
_sacr_mod = _load("adapters.sacr", "sacr.py")
_elp_mod = _load("adapters.elp", "elp.py")
_dipp_mod = _load("adapters.dipp", "dipp.py")
_bmc_mod = _load("adapters.briskmoe_cache", "briskmoe_cache.py")

BriskMoECache = _bmc_mod.BriskMoECache
BriskMoECacheConfig = _bmc_mod.BriskMoECacheConfig
AcceptRejectTrackerConfig = _tracker_mod.AcceptRejectTrackerConfig
SACRConfig = _sacr_mod.SACRConfig
ELPConfig = _elp_mod.ELPConfig
DIPPConfig = _dipp_mod.DIPPConfig


# ========================================================================
# Helpers
# ========================================================================


def make_cache(slots=5, **overrides) -> BriskMoECache:
    """Create a small cache for testing (5 slots, quick promotion)."""
    cfg = BriskMoECacheConfig(
        total_slots_per_layer=slots,
        elp=ELPConfig(
            pin_ratio=0.6,            # 3 pin, 2 flex for slots=5
            promotion_threshold=3,
            demotion_window=20,
            rebalance_interval=5,
        ),
        sacr=SACRConfig(
            alpha=0.3, beta=0.2, gamma=0.5,
            recency_window=50,
        ),
        dipp=DIPPConfig(
            max_prefetch_experts=4,
            enable_progressive=True,
        ),
        tracker=AcceptRejectTrackerConfig(
            ema_alpha=0.2,
            min_observations=2,
        ),
        rebalance_interval=5,
        **overrides,
    )
    return BriskMoECache(cfg)


# ========================================================================
# T1: Basic hit / miss / insert
# ========================================================================


class TestBasicAccess:
    def test_first_access_is_miss(self):
        cache = make_cache()
        hit, victim = cache.access_expert(0, 10, step=1)
        assert hit is False
        assert victim is None  # cache not full

    def test_second_access_is_hit(self):
        cache = make_cache()
        cache.access_expert(0, 10, step=1)
        hit, victim = cache.access_expert(0, 10, step=2)
        assert hit is True
        assert victim is None

    def test_fill_cache_no_eviction(self):
        cache = make_cache(slots=3)
        for eid in [1, 2, 3]:
            hit, _ = cache.access_expert(0, eid, step=eid)
            assert hit is False
        assert cache.get_cache_state(0) == {1, 2, 3}

    def test_empty_layer_returns_empty(self):
        cache = make_cache()
        assert cache.get_cache_state(99) == set()


# ========================================================================
# T2: Eviction triggers after cache full
# ========================================================================


class TestEviction:
    def test_eviction_returns_victim(self):
        cache = make_cache(slots=3)
        # Fill cache
        for eid in [1, 2, 3]:
            cache.access_expert(0, eid, step=eid)
        # Insert new expert — must evict
        hit, victim = cache.access_expert(0, 99, step=10)
        assert hit is False
        assert victim is not None
        assert victim in {1, 2, 3}
        assert 99 in cache.get_cache_state(0)
        assert len(cache.get_cache_state(0)) == 3

    def test_eviction_increments_stats(self):
        cache = make_cache(slots=2)
        cache.access_expert(0, 1, step=1)
        cache.access_expert(0, 2, step=2)
        cache.access_expert(0, 3, step=3)
        assert cache.stats.evictions == 1

    def test_repeated_evictions(self):
        cache = make_cache(slots=2)
        for step, eid in enumerate([1, 2, 3, 4, 5], start=1):
            cache.access_expert(0, eid, step=step)
        assert cache.stats.evictions == 3
        assert len(cache.get_cache_state(0)) == 2


# ========================================================================
# T3: SACR prefer evicting low-AcceptRatio experts
# ========================================================================


class TestSACREviction:
    def test_low_ar_evicted_first(self):
        # Use high promotion threshold to keep all experts in Flex Zone,
        # isolating SACR's AcceptRatio-based eviction from ELP pinning.
        cfg = BriskMoECacheConfig(
            total_slots_per_layer=3,
            elp=ELPConfig(
                pin_ratio=0.6,
                promotion_threshold=999,  # disable promotion
                demotion_window=20,
            ),
            sacr=SACRConfig(alpha=0.3, beta=0.2, gamma=0.5, recency_window=50),
            tracker=AcceptRejectTrackerConfig(ema_alpha=0.2, min_observations=2),
        )
        cache = BriskMoECache(cfg)

        # Fill cache with experts 1, 2, 3
        for eid in [1, 2, 3]:
            cache.access_expert(0, eid, step=1)

        # Feed tracker: expert 1 = always rejected, expert 2/3 = always accepted
        for s in range(1, 5):
            cache.on_verify_complete(
                layer_id=0,
                token_expert_map={0: [1], 1: [2], 2: [3]},
                accepted_mask=[False, True, True],
                step=s,
            )
            # Keep accessing so SACR has metadata (same recency/frequency)
            for eid in [1, 2, 3]:
                cache.access_expert(0, eid, step=s + 1)

        # Insert new expert — expect expert 1 evicted (lowest AcceptRatio)
        _, victim = cache.access_expert(0, 99, step=10)
        assert victim == 1, f"Expected to evict expert 1 (low AR), got {victim}"


# ========================================================================
# T4: ELP pinning prevents eviction
# ========================================================================


class TestELPPinning:
    def test_frequently_accessed_gets_pinned(self):
        cache = make_cache(slots=3)
        # Access expert 1 many times → promoted to Pin
        for s in range(1, 6):
            cache.access_expert(0, 1, step=s)
        cache.elp.rebalance(0)  # promotion is deferred to rebalance
        assert cache.elp.is_pinned(0, 1) is True

    def test_pinned_not_in_flex_candidates(self):
        cache = make_cache(slots=3)
        # Promote expert 1
        for s in range(1, 6):
            cache.access_expert(0, 1, step=s)
        cache.elp.rebalance(0)  # promotion is deferred to rebalance
        # Add experts 2, 3
        cache.access_expert(0, 2, step=6)
        cache.access_expert(0, 3, step=7)

        # Expert 1 should NOT be a flex candidate
        flex = cache.elp.get_flex_candidates(0)
        assert 1 not in flex

    def test_eviction_avoids_pinned(self):
        cache = make_cache(slots=3)
        # Promote expert 1
        for s in range(1, 6):
            cache.access_expert(0, 1, step=s)
        cache.elp.rebalance(0)  # promotion is deferred to rebalance
        # Fill with 2, 3
        cache.access_expert(0, 2, step=6)
        cache.access_expert(0, 3, step=7)
        # Insert 4 → should evict 2 or 3, NOT 1
        _, victim = cache.access_expert(0, 4, step=8)
        assert victim in {2, 3}, f"Pinned expert 1 should be protected, got {victim}"


# ========================================================================
# T5: DIPP callback
# ========================================================================


class TestDIPPCallback:
    def test_draft_callback_returns_schedule(self):
        cache = make_cache(slots=3)
        # Pre-fill cache
        cache.access_expert(0, 1, step=1)
        cache.access_expert(1, 2, step=1)

        cache.begin_draft_round()
        router_preds = {0: [1, 5], 1: [2, 8]}  # 5 and 8 are misses
        schedule = cache.on_draft_token(0, router_preds, step=2)
        # Should include layer 0 expert 5 and layer 1 expert 8
        scheduled_keys = {(l, e) for l, e, v in schedule}
        assert (0, 5) in scheduled_keys
        assert (1, 8) in scheduled_keys
        assert cache.stats.draft_callbacks == 1

    def test_progressive_no_duplicates(self):
        cache = make_cache(slots=3)
        cache.begin_draft_round()

        # Draft token 0
        s0 = cache.on_draft_token(0, {0: [5]}, step=1)
        # Draft token 1 — same expert
        s1 = cache.on_draft_token(1, {0: [5]}, step=2)

        # Expert 5 should only appear once across both rounds
        all_entries = s0 + s1
        layer0_expert5 = [(l, e) for l, e, v in all_entries if l == 0 and e == 5]
        assert len(layer0_expert5) == 1

    def test_begin_round_resets(self):
        cache = make_cache(slots=3)
        cache.begin_draft_round()
        cache.on_draft_token(0, {0: [5]}, step=1)
        # New round
        cache.begin_draft_round()
        s = cache.on_draft_token(0, {0: [5]}, step=2)
        # Expert 5 should appear again after reset
        keys = {(l, e) for l, e, v in s}
        assert (0, 5) in keys


# ========================================================================
# T6: Full SD cycle simulation
# ========================================================================


class TestFullSDCycle:
    def test_full_cycle(self):
        """Simulate: fill cache → draft → prefetch → verify → update."""
        cache = make_cache(slots=5, num_layers=2)

        # Step 1: Warm-up AR phase (fill cache for layers 0 and 1)
        for eid in [0, 1, 2, 3, 4]:
            cache.access_expert(0, eid, step=1)
            cache.access_expert(1, eid, step=1)

        # Step 2: Draft phase — K=2 tokens
        cache.begin_draft_round()
        # Token 0: layer 0 needs [0, 5], layer 1 needs [1, 6]
        s0 = cache.on_draft_token(0, {0: [0, 5], 1: [1, 6]}, step=2)
        # Token 1: layer 0 needs [2, 7], layer 1 needs [3, 8]
        s1 = cache.on_draft_token(1, {0: [2, 7], 1: [3, 8]}, step=3)

        # Check that prefetch includes miss experts 5, 6, 7, 8
        all_scheduled = {(l, e) for l, e, v in s0 + s1}
        for miss in [(0, 5), (1, 6), (0, 7), (1, 8)]:
            assert miss in all_scheduled, f"Miss {miss} not in prefetch schedule"

        # Step 3: Verify phase — access the needed experts
        # Simulate loading: access experts (some will trigger eviction)
        cache.access_expert(0, 0, step=4)  # hit
        cache.access_expert(0, 5, step=4)  # miss → evict something
        cache.access_expert(1, 1, step=4)  # hit
        cache.access_expert(1, 6, step=4)  # miss → evict something

        assert cache.stats.cache_hits >= 2
        assert cache.stats.cache_misses >= 2

        # Step 4: Verify callback — K=2 tokens, first accepted, second rejected
        cache.on_verify_complete(
            layer_id=0,
            token_expert_map={0: [0, 5], 1: [2, 7]},
            accepted_mask=[True, False],
            step=5,
        )
        cache.on_verify_complete(
            layer_id=1,
            token_expert_map={0: [1, 6], 1: [3, 8]},
            accepted_mask=[True, False],
            step=5,
        )

        # Accept ratio should be updated
        ar_0_0 = cache.tracker.get_accept_ratio(0, 0)
        ar_0_7 = cache.tracker.get_accept_ratio(0, 7)
        # Expert 0 was in accepted token → higher AR
        # Expert 7 was in rejected token → lower AR
        assert ar_0_0 > ar_0_7

    def test_multiple_cycles_stable(self):
        """Run 10 SD cycles — no crashes, stats consistent."""
        cache = make_cache(slots=4, num_layers=1)

        for cycle in range(10):
            step_base = cycle * 5

            # Access some experts
            for eid in range(6):
                cache.access_expert(0, eid % 8, step=step_base + 1)

            # Draft
            cache.begin_draft_round()
            cache.on_draft_token(
                0, {0: [cycle % 8, (cycle + 3) % 8]}, step=step_base + 2
            )

            # Verify
            cache.on_verify_complete(
                layer_id=0,
                token_expert_map={0: [cycle % 8], 1: [(cycle + 3) % 8]},
                accepted_mask=[True, cycle % 2 == 0],
                step=step_base + 3,
            )

        assert cache.stats.total_accesses == 60
        assert cache.stats.verify_callbacks == 10
        assert cache.stats.draft_callbacks == 10


# ========================================================================
# T7: Statistics
# ========================================================================


class TestStats:
    def test_stats_summary(self):
        cache = make_cache(slots=2)
        cache.access_expert(0, 1, step=1)
        cache.access_expert(0, 2, step=2)
        cache.access_expert(0, 1, step=3)  # hit
        cache.access_expert(0, 3, step=4)  # eviction

        summary = cache.get_stats_summary()
        assert summary["cache"]["hits"] == 1
        assert summary["cache"]["misses"] == 3
        assert summary["cache"]["evictions"] == 1
        assert 0 < summary["cache"]["hit_rate"] < 1


# ========================================================================
# T8: Multi-layer independence
# ========================================================================


class TestMultiLayer:
    def test_layers_independent(self):
        cache = make_cache(slots=2)
        cache.access_expert(0, 1, step=1)
        cache.access_expert(1, 1, step=1)
        cache.access_expert(0, 2, step=2)
        cache.access_expert(1, 3, step=2)

        assert cache.get_cache_state(0) == {1, 2}
        assert cache.get_cache_state(1) == {1, 3}

    def test_eviction_per_layer(self):
        cache = make_cache(slots=2)
        # Fill layer 0
        cache.access_expert(0, 1, step=1)
        cache.access_expert(0, 2, step=2)
        # Fill layer 1
        cache.access_expert(1, 10, step=3)
        cache.access_expert(1, 20, step=4)
        # Eviction in layer 0 doesn't affect layer 1
        _, v0 = cache.access_expert(0, 99, step=5)
        assert v0 in {1, 2}
        assert cache.get_cache_state(1) == {10, 20}


# ========================================================================
# T9: Compute full prefetch schedule
# ========================================================================


class TestFullSchedule:
    def test_full_schedule(self):
        cache = make_cache(slots=3)
        cache.access_expert(0, 1, step=1)
        cache.access_expert(1, 2, step=1)

        predictions = {
            0: {0: [1, 5], 1: [5, 6]},   # 1 cached, 5 and 6 are misses
            1: {0: [2, 8], 1: [8]},       # 2 cached, 8 is miss
        }
        schedule = cache.compute_full_prefetch_schedule(predictions)

        keys = {(l, e) for l, e, v in schedule}
        assert (0, 1) not in keys  # cached, should not be in schedule
        assert (1, 2) not in keys  # cached
        assert (0, 5) in keys
        assert (0, 6) in keys
        assert (1, 8) in keys

    def test_schedule_respects_budget(self):
        cache = make_cache(slots=2)
        # Many miss experts
        predictions = {
            i: {0: [100 + j for j in range(10)]}
            for i in range(5)
        }
        schedule = cache.compute_full_prefetch_schedule(predictions)
        # Budget is 4
        assert len(schedule) <= 4


# ========================================================================
# T10: Edge cases
# ========================================================================


class TestEdgeCases:
    def test_access_same_expert_many_times(self):
        cache = make_cache(slots=3)
        for s in range(1, 20):
            hit, _ = cache.access_expert(0, 42, step=s)
            if s == 1:
                assert hit is False
            else:
                assert hit is True
        assert cache.stats.cache_hits == 18
        assert cache.stats.cache_misses == 1

    def test_empty_verify(self):
        cache = make_cache()
        # Empty token map — should not crash
        cache.on_verify_complete(0, {}, [], step=1)
        assert cache.stats.verify_callbacks == 1

    def test_empty_draft(self):
        cache = make_cache()
        cache.begin_draft_round()
        schedule = cache.on_draft_token(0, {}, step=1)
        assert schedule == []
