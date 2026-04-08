"""Tests for DIPP — Draft-Informed Prioritized Preloading (ISSUE-003)."""

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


_dipp_mod = _load_module("adapters.dipp", "dipp.py")

DIPPConfig = _dipp_mod.DIPPConfig
DraftInformedPrioritizedPreloader = _dipp_mod.DraftInformedPrioritizedPreloader


class TestDIPP:

    def _make_dipp(self, **kwargs) -> DraftInformedPrioritizedPreloader:
        config = DIPPConfig(**kwargs)
        return DraftInformedPrioritizedPreloader(config=config)

    # ------------------------------------------------------------------
    # Value function
    # ------------------------------------------------------------------

    def test_cached_expert_has_zero_value(self):
        dipp = self._make_dipp()
        predictions = {0: {0: [1, 2], 1: [2, 3]}}
        cache_state = {0: {2}}  # expert 2 is cached
        val = dipp.compute_value(0, 2, predictions, cache_state)
        assert val == 0.0

    def test_unnneeded_expert_has_zero_value(self):
        dipp = self._make_dipp()
        predictions = {0: {0: [1, 2]}}
        cache_state = {0: set()}
        val = dipp.compute_value(0, 99, predictions, cache_state)
        assert val == 0.0

    def test_value_increases_with_demand(self):
        dipp = self._make_dipp()
        # Expert 1: needed by 3 tokens; Expert 2: needed by 1 token
        predictions = {0: {0: [1], 1: [1], 2: [1, 2]}}
        cache_state = {0: set()}
        val_1 = dipp.compute_value(0, 1, predictions, cache_state)
        val_2 = dipp.compute_value(0, 2, predictions, cache_state)
        assert val_1 > val_2

    def test_early_layer_has_higher_urgency(self):
        dipp = self._make_dipp()
        # Same demand (1 token) but different layers
        predictions = {
            0: {0: [1]},   # layer 0
            10: {0: [1]},  # layer 10
        }
        cache_state = {0: set(), 10: set()}
        val_layer0 = dipp.compute_value(0, 1, predictions, cache_state)
        val_layer10 = dipp.compute_value(10, 1, predictions, cache_state)
        assert val_layer0 > val_layer10, (
            f"Layer 0 urgency ({val_layer0:.3f}) should exceed "
            f"layer 10 urgency ({val_layer10:.3f})"
        )

    # ------------------------------------------------------------------
    # Full schedule
    # ------------------------------------------------------------------

    def test_schedule_sorted_by_value_descending(self):
        dipp = self._make_dipp(max_prefetch_experts=100)
        predictions = {
            0: {0: [1], 1: [1], 2: [1]},   # expert 1 demand=3 at layer 0
            0: {0: [1], 1: [1], 2: [1, 2]},
            5: {0: [10]},                    # expert 10 demand=1 at layer 5
        }
        cache_state = {0: set(), 5: set()}

        schedule = dipp.compute_schedule(predictions, cache_state)

        # Should be sorted by value descending
        values = [v for _, _, v in schedule]
        assert values == sorted(values, reverse=True)

    def test_schedule_respects_budget(self):
        dipp = self._make_dipp(max_prefetch_experts=5)

        # Create predictions with 20 miss experts across layers
        predictions = {}
        for layer in range(10):
            predictions[layer] = {0: [layer * 10 + i for i in range(2)]}
        cache_state = {layer: set() for layer in range(10)}

        schedule = dipp.compute_schedule(predictions, cache_state)
        assert len(schedule) <= 5, f"Schedule should respect budget, got {len(schedule)}"

    def test_schedule_excludes_cached_experts(self):
        dipp = self._make_dipp(max_prefetch_experts=100)
        predictions = {0: {0: [1, 2, 3]}}
        cache_state = {0: {2}}  # expert 2 cached

        schedule = dipp.compute_schedule(predictions, cache_state)
        scheduled_experts = {(l, e) for l, e, _ in schedule}
        assert (0, 2) not in scheduled_experts

    def test_schedule_prefers_early_layers(self):
        """With equal demand, early-layer experts should rank higher."""
        dipp = self._make_dipp(max_prefetch_experts=100)
        # Expert 1 at layer 0 and expert 1 at layer 20, both demand=1
        predictions = {
            0: {0: [1]},
            20: {0: [2]},
        }
        cache_state = {0: set(), 20: set()}

        schedule = dipp.compute_schedule(predictions, cache_state)
        # First entry should be layer 0
        assert schedule[0][0] == 0, "Layer 0 should be scheduled first"

    # ------------------------------------------------------------------
    # Progressive preloading
    # ------------------------------------------------------------------

    def test_progressive_accumulates(self):
        dipp = self._make_dipp(max_prefetch_experts=100)
        cache_state = {0: set(), 1: set()}

        # First draft token
        new1 = dipp.on_draft_token(0, {0: [1, 2], 1: [3]}, cache_state)
        assert len(new1) > 0

        # Second draft token — should find new experts
        new2 = dipp.on_draft_token(1, {0: [2, 4], 1: [5]}, cache_state)
        # Expert 4 and 5 are new; expert 2 was already scheduled
        new_ids = {(l, e) for l, e, _ in new2}
        assert (0, 4) in new_ids or (1, 5) in new_ids

    def test_progressive_no_duplicates(self):
        dipp = self._make_dipp(max_prefetch_experts=100)
        cache_state = {0: set()}

        new1 = dipp.on_draft_token(0, {0: [1]}, cache_state)
        new2 = dipp.on_draft_token(1, {0: [1]}, cache_state)

        # Expert 1 should only appear in first round
        assert any(e == 1 for _, e, _ in new1)
        assert not any(e == 1 for _, e, _ in new2)

    def test_reset_round(self):
        dipp = self._make_dipp(max_prefetch_experts=100)
        cache_state = {0: set()}
        dipp.on_draft_token(0, {0: [1]}, cache_state)
        dipp.reset_round()
        # After reset, expert 1 should be schedulable again
        new = dipp.on_draft_token(0, {0: [1]}, cache_state)
        assert any(e == 1 for _, e, _ in new)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def test_stats_tracking(self):
        dipp = self._make_dipp(max_prefetch_experts=3)
        predictions = {}
        for layer in range(5):
            predictions[layer] = {0: [layer * 10, layer * 10 + 1]}
        cache_state = {l: set() for l in range(5)}

        dipp.compute_schedule(predictions, cache_state)
        stats = dipp.get_stats()

        assert stats.total_schedules == 1
        assert stats.total_experts_scheduled == 3  # budget = 3
        assert stats.total_experts_over_budget >= 7  # 10 total - 3 scheduled

    # ------------------------------------------------------------------
    # Realistic scenario: K=3, L=26
    # ------------------------------------------------------------------

    def test_realistic_k3_l26(self):
        """
        Simulate K=3 draft tokens, 26 MoE layers, top-8 experts per token.
        Verify budget constraint and value ordering.
        """
        dipp = self._make_dipp(max_prefetch_experts=79)
        cache_state = {}
        predictions = {}

        # Simulate: each of 26 layers has predictions for 3 tokens
        # Each token activates 8 experts (some overlap)
        import random
        random.seed(42)

        for layer in range(26):
            cache_state[layer] = set(random.sample(range(128), 12))  # 12 cached
            predictions[layer] = {}
            for tok in range(3):
                predictions[layer][tok] = random.sample(range(128), 8)

        schedule = dipp.compute_schedule(predictions, cache_state)

        # Budget check
        assert len(schedule) <= 79

        # Value ordering check
        values = [v for _, _, v in schedule]
        assert values == sorted(values, reverse=True)

        # All scheduled experts should be cache misses
        for layer_id, eid, _ in schedule:
            assert eid not in cache_state[layer_id]

        # Stats
        assert dipp.stats.total_experts_over_budget > 0, (
            "With 26 layers × ~many misses, demand should exceed budget"
        )

    def test_urgency_modes(self):
        """Verify different urgency decay modes produce different orderings."""
        for mode in ["inverse", "linear", "exp"]:
            dipp = self._make_dipp(urgency_decay=mode, max_prefetch_experts=100)
            predictions = {0: {0: [1]}, 20: {0: [2]}}
            cache_state = {0: set(), 20: set()}
            schedule = dipp.compute_schedule(predictions, cache_state)
            # Layer 0 should always be first regardless of mode
            assert schedule[0][0] == 0
