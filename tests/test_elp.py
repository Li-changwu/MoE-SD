"""Tests for ELP — Expert Lifecycle Partitioning (ISSUE-002)."""

import importlib.util
import os
import sys


def _load_module(name, filename):
    path = os.path.join(os.path.dirname(__file__), "..", "adapters", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_elp_mod = _load_module("adapters.elp", "elp.py")

ELPConfig = _elp_mod.ELPConfig
ExpertLifecyclePartition = _elp_mod.ExpertLifecyclePartition


class TestELP:

    def _make_elp(self, total_slots=17, **kwargs) -> ExpertLifecyclePartition:
        config = ELPConfig(**kwargs)
        return ExpertLifecyclePartition(config=config, total_slots=total_slots)

    # ------------------------------------------------------------------
    # Basic partition behavior
    # ------------------------------------------------------------------

    def test_new_expert_enters_flex(self):
        elp = self._make_elp()
        elp.access(0, 1, step=1)
        assert elp.classify(0, 1) == "flex"

    def test_uncached_expert(self):
        elp = self._make_elp()
        assert elp.classify(0, 999) == "uncached"

    def test_promotion_after_threshold(self):
        elp = self._make_elp(promotion_threshold=3)
        elp.access(0, 1, step=1)
        elp.access(0, 1, step=2)
        assert elp.classify(0, 1) == "flex"
        elp.access(0, 1, step=3)  # 3rd access → promotion
        elp.rebalance(0)
        assert elp.classify(0, 1) == "pin"

    def test_pin_zone_capacity_limit(self):
        """Pin Zone respects capacity — weakest pinned expert is replaced."""
        # total_slots=10, pin_ratio=0.5 → pin_capacity=5
        elp = self._make_elp(total_slots=10, pin_ratio=0.5, promotion_threshold=3)

        # Fill Pin Zone with experts 1-5 (each accessed 3 times)
        for eid in range(1, 6):
            for step in range(3):
                elp.access(0, eid, step=step + eid * 10)

        elp.rebalance(0)
        for eid in range(1, 6):
            assert elp.classify(0, eid) == "pin"

        # Expert 6 gets accessed 10 times → stronger than any pinned expert (3 each)
        for step in range(10):
            elp.access(0, 6, step=100 + step)

        elp.rebalance(0)
        # Expert 6 should be promoted, weakest (expert with lowest count) demoted
        assert elp.classify(0, 6) == "pin"
        pin_set = elp.get_pin_set(0)
        assert len(pin_set) <= 5  # Still within capacity

    # ------------------------------------------------------------------
    # Flex candidates for eviction
    # ------------------------------------------------------------------

    def test_flex_candidates_exclude_pinned(self):
        elp = self._make_elp(promotion_threshold=2)
        # Expert 1: promoted (2 accesses)
        elp.access(0, 1, step=1)
        elp.access(0, 1, step=2)
        # Expert 2: still flex (1 access)
        elp.access(0, 2, step=3)
        elp.rebalance(0)

        assert elp.classify(0, 1) == "pin"
        assert elp.classify(0, 2) == "flex"

        candidates = elp.get_flex_candidates(0)
        assert 1 not in candidates
        assert 2 in candidates

    # ------------------------------------------------------------------
    # Cascade isolation: burst access scenario
    # ------------------------------------------------------------------

    def test_burst_access_does_not_evict_pinned(self):
        """
        Simulate SD burst: W_sd=27 experts needed, cache S=17.
        Pin Zone should protect persistent experts from burst overflow.
        """
        elp = self._make_elp(total_slots=17, pin_ratio=0.7, promotion_threshold=3)
        # pin_capacity = 11, flex_capacity = 6

        # Establish 10 persistent experts (each accessed 5 times)
        for eid in range(10):
            for step in range(5):
                elp.access(0, eid, step=step)

        elp.rebalance(0)
        # All 10 should be pinned
        for eid in range(10):
            assert elp.is_pinned(0, eid), f"Expert {eid} should be pinned"

        # Now simulate burst: 27 experts needed, 17 new transient ones
        for eid in range(100, 117):
            elp.access(0, eid, step=10)

        # Pinned experts should still be pinned
        for eid in range(10):
            assert elp.is_pinned(0, eid), (
                f"Expert {eid} should still be pinned after burst"
            )

        # Flex candidates should only include transient experts
        flex = elp.get_flex_candidates(0)
        for eid in range(10):
            assert eid not in flex, f"Pinned expert {eid} should not be in flex candidates"

    # ------------------------------------------------------------------
    # Demotion
    # ------------------------------------------------------------------

    def test_demotion_of_stale_pinned_expert(self):
        elp = self._make_elp(promotion_threshold=2, demotion_window=5)
        # Promote expert 1
        elp.access(0, 1, step=1)
        elp.access(0, 1, step=2)
        elp.rebalance(0)
        assert elp.classify(0, 1) == "pin"

        # Advance time far beyond demotion window
        elp._current_step = 100
        elp.rebalance(0)

        assert elp.classify(0, 1) == "flex", "Stale pinned expert should be demoted"

    def test_active_pinned_expert_not_demoted(self):
        elp = self._make_elp(promotion_threshold=2, demotion_window=5)
        elp.access(0, 1, step=1)
        elp.access(0, 1, step=2)
        elp.rebalance(0)
        assert elp.classify(0, 1) == "pin"

        # Access again at step 8
        elp.access(0, 1, step=8)
        elp._current_step = 8
        elp.rebalance(0)

        assert elp.classify(0, 1) == "pin", "Active pinned expert should not be demoted"

    # ------------------------------------------------------------------
    # Remove and stats
    # ------------------------------------------------------------------

    def test_remove_expert(self):
        elp = self._make_elp(promotion_threshold=2)
        elp.access(0, 1, step=1)
        elp.access(0, 1, step=2)
        elp.rebalance(0)
        assert elp.classify(0, 1) == "pin"

        elp.remove_expert(0, 1)
        assert elp.classify(0, 1) == "uncached"
        assert 1 not in elp.get_pin_set(0)
        assert 1 not in elp.get_flex_candidates(0)

    def test_partition_stats(self):
        elp = self._make_elp(total_slots=10, pin_ratio=0.5, promotion_threshold=2)
        elp.access(0, 1, step=1)
        elp.access(0, 1, step=2)  # promoted
        elp.access(0, 2, step=3)  # flex
        elp.rebalance(0)

        stats = elp.get_partition_stats(0)
        assert stats.pin_count == 1
        assert stats.flex_count == 1
        assert stats.pin_capacity == 5
        assert stats.flex_capacity == 5
        assert stats.promotions == 1

    # ------------------------------------------------------------------
    # Cascade length simulation
    # ------------------------------------------------------------------

    def test_cascade_length_with_elp(self):
        """
        Simulate cascade eviction measurement:
        Without ELP: burst overflow propagates through entire cache.
        With ELP: overflow is confined to Flex Zone.
        """
        total_slots = 17
        elp = self._make_elp(
            total_slots=total_slots, pin_ratio=0.7,
            promotion_threshold=3,
        )
        # pin_capacity=11, flex_capacity=6

        # Build 10 persistent experts
        for eid in range(10):
            for step in range(5):
                elp.access(0, eid, step=step)

        elp.rebalance(0)
        # Verify pin state
        pin_count = len(elp.get_pin_set(0))
        flex_count = len(elp.get_flex_candidates(0))
        assert pin_count == 10
        assert flex_count == 0

        # Burst: W_sd=27 means 27 experts needed, but only flex_capacity=6 slots
        # available for new transient experts. With ELP, cascade is limited
        # to flex zone size (6), not entire cache (17).
        burst_experts = list(range(100, 127))
        for eid in burst_experts:
            elp.access(0, eid, step=20)

        # Pin Zone unchanged
        assert len(elp.get_pin_set(0)) == 10
        # Flex Zone absorbed the burst (up to flex_capacity worth of tracking)
        flex_ids = elp.get_flex_candidates(0)
        # All burst experts should be in flex
        for eid in burst_experts:
            assert eid in flex_ids
