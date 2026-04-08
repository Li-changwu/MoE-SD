"""Tests for AcceptRejectTracker (ISSUE-000)."""

import importlib.util
import os
import sys
import threading

# Direct import to bypass adapters/__init__.py (which pulls torch)
_mod_name = "adapters.accept_reject_tracker"
_spec = importlib.util.spec_from_file_location(
    _mod_name,
    os.path.join(os.path.dirname(__file__), "..", "adapters", "accept_reject_tracker.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_mod_name] = _mod
_spec.loader.exec_module(_mod)

AcceptRejectTracker = _mod.AcceptRejectTracker
AcceptRejectTrackerConfig = _mod.AcceptRejectTrackerConfig


class TestAcceptRejectTracker:
    """Unit tests for AcceptRejectTracker."""

    def _make_tracker(self, **kwargs) -> AcceptRejectTracker:
        config = AcceptRejectTrackerConfig(**kwargs)
        return AcceptRejectTracker(config)

    # ------------------------------------------------------------------
    # Basic accept/reject recording
    # ------------------------------------------------------------------

    def test_single_step_all_accepted(self):
        tracker = self._make_tracker()
        # 4 tokens, all accepted, each activating 2 experts in layer 0
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1, 2], 1: [2, 3], 2: [3, 4], 3: [4, 5]},
            accepted_mask=[True, True, True, True],
            step_id=1,
        )
        # Expert 1: accessed by 1 accepted token → ratio should be high
        assert tracker.get_accept_ratio(0, 1) > 0.5
        # Expert 5: accessed by 1 accepted token → ratio should be high
        assert tracker.get_accept_ratio(0, 5) > 0.5
        assert tracker.get_total_count(0, 1) == 1
        assert tracker.get_accept_count(0, 1) == 1

    def test_single_step_mixed(self):
        tracker = self._make_tracker()
        # 4 tokens: 2 accepted, 2 rejected
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1, 2], 1: [2, 3], 2: [4, 5], 3: [5, 6]},
            accepted_mask=[True, True, False, False],
            step_id=1,
        )
        # Expert 1: only accessed by accepted token 0 → high ratio
        assert tracker.get_accept_ratio(0, 1) > 0.5
        # Expert 6: only accessed by rejected token 3 → low ratio
        assert tracker.get_accept_ratio(0, 6) < 0.5
        # Expert 2: accessed by accepted token 0 and accepted token 1 → high
        assert tracker.get_accept_ratio(0, 2) > 0.5
        # Expert 5: accessed by rejected token 2 and rejected token 3 → low
        assert tracker.get_accept_ratio(0, 5) < 0.5

    def test_multi_step_ema_convergence(self):
        tracker = self._make_tracker(ema_alpha=0.3)
        # Step 1: expert 10 accessed by accepted token
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [10]},
            accepted_mask=[True],
            step_id=1,
        )
        r1 = tracker.get_accept_ratio(0, 10)

        # Step 2: expert 10 accessed by rejected token
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [10]},
            accepted_mask=[False],
            step_id=2,
        )
        r2 = tracker.get_accept_ratio(0, 10)

        # EMA should decrease after rejected access
        assert r2 < r1

        # Step 3: expert 10 again accepted
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [10]},
            accepted_mask=[True],
            step_id=3,
        )
        r3 = tracker.get_accept_ratio(0, 10)
        assert r3 > r2

    def test_unseen_expert_returns_neutral(self):
        tracker = self._make_tracker()
        assert tracker.get_accept_ratio(0, 999) == 0.5
        assert tracker.get_total_count(0, 999) == 0

    def test_is_reliable(self):
        tracker = self._make_tracker(min_observations=3)
        # 2 observations: not reliable
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1], 1: [1]},
            accepted_mask=[True, False],
            step_id=1,
        )
        assert not tracker.is_reliable(0, 1)

        # 3rd observation: now reliable
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1]},
            accepted_mask=[True],
            step_id=2,
        )
        assert tracker.is_reliable(0, 1)

    def test_multiple_layers(self):
        tracker = self._make_tracker()
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1]},
            accepted_mask=[True],
            step_id=1,
        )
        tracker.record_verify_result(
            layer_id=5,
            token_expert_map={0: [1]},
            accepted_mask=[False],
            step_id=1,
        )
        # Same expert_id=1, different layers → different stats
        assert tracker.get_accept_ratio(0, 1) > 0.5
        assert tracker.get_accept_ratio(5, 1) < 0.5

    def test_get_expert_stats(self):
        tracker = self._make_tracker()
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1, 2], 1: [3]},
            accepted_mask=[True, False],
            step_id=1,
        )
        stats = tracker.get_expert_stats(0)
        assert 1 in stats
        assert 2 in stats
        assert 3 in stats
        assert stats[1].accepted_count == 1
        assert stats[3].rejected_count == 1

    def test_reset(self):
        tracker = self._make_tracker()
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={0: [1]},
            accepted_mask=[True],
            step_id=1,
        )
        tracker.reset()
        assert tracker.get_total_count(0, 1) == 0
        assert tracker.global_step == 0

    def test_advance_step(self):
        tracker = self._make_tracker()
        assert tracker.global_step == 0
        s = tracker.advance_step()
        assert s == 1
        assert tracker.global_step == 1

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_concurrent_writes(self):
        tracker = self._make_tracker()
        errors = []

        def writer(layer_id: int, n: int):
            try:
                for step in range(n):
                    tracker.record_verify_result(
                        layer_id=layer_id,
                        token_expert_map={0: [1, 2], 1: [3]},
                        accepted_mask=[True, False],
                        step_id=step,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(i, 100))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Each layer had 100 rounds, 3 experts per round
        for layer_id in range(4):
            assert tracker.get_total_count(layer_id, 1) == 100
            assert tracker.get_total_count(layer_id, 3) == 100

    # ------------------------------------------------------------------
    # SD-realistic scenario: 3 verify steps
    # ------------------------------------------------------------------

    def test_three_step_sd_scenario(self):
        """
        Simulates 3 SD verify steps with K=3 (4 tokens each).
        Verifies AcceptRatio distinguishes accepted vs rejected experts.
        """
        tracker = self._make_tracker(ema_alpha=0.2)

        # Step 1: tokens 0,1 accepted; 2,3 rejected
        # Expert 100: only accepted tokens
        # Expert 200: only rejected tokens
        # Expert 50: both
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={
                0: [100, 50],   # accepted
                1: [100, 50],   # accepted
                2: [200, 50],   # rejected
                3: [200],       # rejected
            },
            accepted_mask=[True, True, False, False],
            step_id=1,
        )

        # Step 2: similar pattern
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={
                0: [100],
                1: [100, 50],
                2: [200],
                3: [200, 50],
            },
            accepted_mask=[True, True, False, False],
            step_id=2,
        )

        # Step 3: all accepted (good draft model)
        tracker.record_verify_result(
            layer_id=0,
            token_expert_map={
                0: [100, 50],
                1: [100],
                2: [50],
                3: [200],
            },
            accepted_mask=[True, True, True, True],
            step_id=3,
        )

        # Expert 100: mostly accepted → high ratio
        ratio_100 = tracker.get_accept_ratio(0, 100)
        # Expert 200: mostly rejected → lower ratio
        ratio_200 = tracker.get_accept_ratio(0, 200)
        # Expert 50: mixed → middle
        ratio_50 = tracker.get_accept_ratio(0, 50)

        assert ratio_100 > ratio_200, (
            f"Expert 100 (mostly accepted) should have higher ratio than "
            f"Expert 200 (mostly rejected): {ratio_100:.3f} vs {ratio_200:.3f}"
        )
        # Expert 100 should generally be highest
        assert ratio_100 > 0.6
        # Expert 200 should be lower (it was rejected in steps 1,2 but accepted in step 3)
        assert ratio_200 < ratio_100

        # All experts should be reliable after 3+ observations
        assert tracker.is_reliable(0, 100)
        assert tracker.is_reliable(0, 200)
        assert tracker.is_reliable(0, 50)
