"""Tests for SACR — Speculation-Aware Cache Replacement (ISSUE-001)."""

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


_art_mod = _load_module("adapters.accept_reject_tracker", "accept_reject_tracker.py")
_sacr_mod = _load_module("adapters.sacr", "sacr.py")

AcceptRejectTracker = _art_mod.AcceptRejectTracker
AcceptRejectTrackerConfig = _art_mod.AcceptRejectTrackerConfig
SACREvictionPolicy = _sacr_mod.SACREvictionPolicy
SACRConfig = _sacr_mod.SACRConfig


class TestSACR:

    def _make_tracker_and_sacr(self, **sacr_kwargs):
        tracker = AcceptRejectTracker(AcceptRejectTrackerConfig(ema_alpha=0.3))
        config = SACRConfig(**sacr_kwargs)
        sacr = SACREvictionPolicy(config=config, tracker=tracker)
        return tracker, sacr

    # ------------------------------------------------------------------
    # Basic scoring
    # ------------------------------------------------------------------

    def test_score_with_no_access(self):
        _, sacr = self._make_tracker_and_sacr()
        # Unaccessed expert → score 0
        assert sacr.score(0, 99) == 0.0

    def test_recency_dominates_when_no_tracker(self):
        sacr = SACREvictionPolicy(config=SACRConfig(), tracker=None)
        sacr.record_access(0, 1, step=1)
        sacr.record_access(0, 2, step=10)
        sacr._current_step = 10
        # Expert 2 was accessed more recently → higher score
        assert sacr.score(0, 2) > sacr.score(0, 1)

    def test_accept_ratio_influences_score(self):
        tracker, sacr = self._make_tracker_and_sacr()

        # Simulate: expert 1 always accepted, expert 2 always rejected
        for step in range(5):
            tracker.record_verify_result(
                layer_id=0,
                token_expert_map={0: [1], 1: [2]},
                accepted_mask=[True, False],
                step_id=step,
            )
            sacr.record_access(0, 1, step=step)
            sacr.record_access(0, 2, step=step)

        # Both have same recency and frequency → AcceptRatio differentiates
        score_1 = sacr.score(0, 1)
        score_2 = sacr.score(0, 2)
        assert score_1 > score_2, (
            f"Accepted expert score ({score_1:.3f}) should exceed "
            f"rejected expert score ({score_2:.3f})"
        )

    # ------------------------------------------------------------------
    # Victim selection
    # ------------------------------------------------------------------

    def test_select_victim_prefers_rejected_expert(self):
        tracker, sacr = self._make_tracker_and_sacr()

        # Expert 10: high AcceptRatio; Expert 20: low AcceptRatio
        for step in range(5):
            tracker.record_verify_result(
                layer_id=0,
                token_expert_map={0: [10], 1: [20]},
                accepted_mask=[True, False],
                step_id=step,
            )
            sacr.record_access(0, 10, step=step)
            sacr.record_access(0, 20, step=step)

        victim = sacr.select_victim(0, [10, 20])
        assert victim == 20, "Should evict the rejected-token expert"

    def test_select_victim_prefers_stale_when_no_ar(self):
        """Without AcceptRatio data, SACR falls back to LRU+LFU."""
        sacr = SACREvictionPolicy(config=SACRConfig(), tracker=None)
        sacr.record_access(0, 1, step=1)
        sacr.record_access(0, 2, step=10)
        sacr._current_step = 10

        victim = sacr.select_victim(0, [1, 2])
        assert victim == 1, "Should evict the staler expert when no AR data"

    def test_select_victim_empty_candidates_raises(self):
        _, sacr = self._make_tracker_and_sacr()
        try:
            sacr.select_victim(0, [])
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Degradation: all accepted → behaves like weighted LRU+LFU
    # ------------------------------------------------------------------

    def test_degradation_all_accepted(self):
        tracker, sacr = self._make_tracker_and_sacr()

        # All tokens accepted → AcceptRatio ≈ 1.0 for all experts
        for step in range(5):
            tracker.record_verify_result(
                layer_id=0,
                token_expert_map={0: [1], 1: [2]},
                accepted_mask=[True, True],
                step_id=step,
            )
            sacr.record_access(0, 1, step=step)
            sacr.record_access(0, 2, step=step)

        # Both have similar AcceptRatio → recency/frequency should decide
        # Access expert 1 once more at step 10 to differentiate
        sacr.record_access(0, 1, step=10)
        sacr._current_step = 10

        victim = sacr.select_victim(0, [1, 2])
        assert victim == 2, "With equal AR, should evict the staler expert"

    # ------------------------------------------------------------------
    # Multi-expert realistic scenario
    # ------------------------------------------------------------------

    def test_realistic_mixed_scenario(self):
        """
        5 experts in cache. Expert 1,2 have high AR, 3 has medium, 4,5 have low.
        SACR should evict from low AR experts first.
        """
        tracker, sacr = self._make_tracker_and_sacr(
            alpha=0.2, beta=0.1, gamma=0.7
        )

        for step in range(10):
            # Experts 1,2: always accepted
            # Expert 3: 50/50
            # Experts 4,5: always rejected
            tracker.record_verify_result(
                layer_id=0,
                token_expert_map={
                    0: [1, 3],
                    1: [2],
                    2: [3, 4],
                    3: [5],
                },
                accepted_mask=[True, True, False, False],
                step_id=step,
            )
            for e in [1, 2, 3, 4, 5]:
                sacr.record_access(0, e, step=step)

        # Check ordering of scores
        scores = {e: sacr.score(0, e) for e in [1, 2, 3, 4, 5]}

        # High AR experts should have higher scores
        assert scores[1] > scores[4], f"Expert 1 ({scores[1]:.3f}) > Expert 4 ({scores[4]:.3f})"
        assert scores[2] > scores[5], f"Expert 2 ({scores[2]:.3f}) > Expert 5 ({scores[5]:.3f})"

        # First victim should be from {4, 5}
        victim = sacr.select_victim(0, [1, 2, 3, 4, 5])
        assert victim in [4, 5], f"First victim should be low-AR expert, got {victim}"

    # ------------------------------------------------------------------
    # Metadata management
    # ------------------------------------------------------------------

    def test_remove_expert(self):
        _, sacr = self._make_tracker_and_sacr()
        sacr.record_access(0, 1, step=1)
        assert sacr.get_meta(0, 1) is not None
        sacr.remove_expert(0, 1)
        assert sacr.get_meta(0, 1) is None
        assert sacr.score(0, 1) == 0.0

    def test_advance_step(self):
        _, sacr = self._make_tracker_and_sacr()
        assert sacr.current_step == 0
        sacr.advance_step()
        assert sacr.current_step == 1
