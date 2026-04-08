"""Tests for PredCache (Predictive Expert Cache Management)."""
import pytest
from adapters.pred_cache import PredCacheConfig, PredictiveExpertCacheManager


@pytest.fixture
def pc():
    config = PredCacheConfig(num_experts=16, top_k=4, lru_fallback_weight=0.1)
    return PredictiveExpertCacheManager(config=config)


class TestPredCacheBasic:
    def test_init(self, pc):
        assert pc._step == 0
        assert pc.config.num_experts == 16

    def test_record_access(self, pc):
        pc.record_access_batch(0, [1, 2, 3])
        assert pc._last_access[0][1] == 0
        assert pc._last_access[0][2] == 0
        assert pc._last_access[0][3] == 0
        # Untouched expert should be 0 (default)
        assert pc._last_access[0][0] == 0

    def test_update_predictions_flat(self, pc):
        pc.update_predictions_from_flat(0, [1, 2, 3, 1, 2])
        assert pc._predicted_demand[0][1] == 2
        assert pc._predicted_demand[0][2] == 2
        assert pc._predicted_demand[0][3] == 1
        assert pc._predicted_demand[0][0] == 0

    def test_advance_step(self, pc):
        pc.advance_step()
        assert pc._step == 1
        pc.advance_step()
        assert pc._step == 2


class TestPredEvict:
    def test_select_victim_by_demand(self, pc):
        """Victim should be the expert with lowest predicted demand."""
        # Expert 1 has high demand, expert 5 has no demand
        pc.update_predictions_from_flat(0, [1, 1, 1, 2])
        # Both accessed at same time
        pc.record_access_batch(0, [1, 2, 5])
        victim = pc.select_victim(0, [1, 2, 5])
        # Expert 5 has 0 demand → lowest PredScore → evicted
        assert victim == 5

    def test_select_victim_lru_fallback(self, pc):
        """When all demand is 0, LRU fallback picks least recently used."""
        # No predictions, so demand is all 0
        pc.record_access_batch(0, [1])
        pc.advance_step()
        pc.record_access_batch(0, [2])
        pc.advance_step()
        pc.record_access_batch(0, [3])
        # Expert 1 accessed at step 0 (oldest), 3 at step 2 (newest)
        victim = pc.select_victim(0, [1, 2, 3])
        assert victim == 1  # LRU = expert 1

    def test_select_victim_prefers_no_demand(self, pc):
        """Expert with demand is protected over expert without."""
        pc.update_predictions_from_flat(0, [2, 2, 2])
        pc.record_access_batch(0, [1, 2])
        victim = pc.select_victim(0, [1, 2])
        # Expert 1 has 0 demand, expert 2 has demand 3
        assert victim == 1

    def test_select_victim_single_candidate(self, pc):
        victim = pc.select_victim(0, [7])
        assert victim == 7


class TestPrefetchSchedule:
    def test_basic_schedule(self, pc):
        # Set up: layer 0 has demand for expert 5, which is not in cache
        pc.update_predictions_from_flat(0, [5, 5, 5])
        cache_states = {0: {0, 1, 2, 3}}  # experts 0-3 cached
        schedule = pc.compute_prefetch_schedule(cache_states, num_layers=1)
        # Expert 5 should be in the schedule
        layer_ids = [s[0] for s in schedule]
        expert_ids = [s[1] for s in schedule]
        assert 0 in layer_ids
        assert 5 in expert_ids

    def test_already_cached_not_prefetched(self, pc):
        pc.update_predictions_from_flat(0, [1, 1, 1])
        cache_states = {0: {1, 2, 3}}  # expert 1 already cached
        schedule = pc.compute_prefetch_schedule(cache_states, num_layers=1)
        expert_ids = [s[1] for s in schedule]
        assert 1 not in expert_ids  # already cached

    def test_schedule_limited(self, pc):
        """Schedule should be limited to max_prefetch_experts."""
        config = PredCacheConfig(num_experts=16, top_k=4, max_prefetch_experts=3)
        pc2 = PredictiveExpertCacheManager(config=config)
        # Give demand to many experts
        pc2.update_predictions_from_flat(0, list(range(16)))
        cache_states = {0: set()}
        schedule = pc2.compute_prefetch_schedule(cache_states, num_layers=1)
        assert len(schedule) <= 3

    def test_get_demand_boost(self, pc):
        """get_demand_boost returns lazily-decayed demand."""
        pc.update_predictions_from_flat(0, [5, 5, 5])  # demand=3 for expert 5
        assert pc.get_demand_boost(0, 5) > 0.0
        # Unknown layer/expert returns 0
        assert pc.get_demand_boost(99, 5) == 0.0
        assert pc.get_demand_boost(0, 999) == 0.0
