"""
Tests for Speculation Divergence Detector (SDD)
"""

import torch
import pytest


class TestSDDBasic:

    def test_init_verify_round(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        sdd = SpeculationDivergenceDetector(config=SDDConfig(), num_layers=48)
        sdd.init_verify_round(num_draft_tokens=3)
        mask = sdd.get_frozen_mask()
        assert len(mask) == 3
        assert all(not v for v in mask.values())

    def test_no_freeze_before_min_layer(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        config = SDDConfig(min_check_layer=10, consecutive_threshold=2)
        sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
        sdd.init_verify_round(num_draft_tokens=2)

        # Check layers before min_check_layer — should never freeze
        for layer_id in range(10):
            logits = torch.randn(2, 128)
            frozen = sdd.check_layer(layer_id, logits, [0, 1])
            assert not frozen.any(), f"Should not freeze before layer {config.min_check_layer}"

    def test_freeze_after_consecutive_divergence(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        config = SDDConfig(
            min_check_layer=0,
            consecutive_threshold=2,
            method="entropy",
            entropy_threshold=0.5,  # Low threshold → easier to trigger
        )
        sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
        sdd.init_verify_round(num_draft_tokens=1)

        # Create highly concentrated logits (low entropy → high divergence score)
        concentrated = torch.zeros(1, 128)
        concentrated[0, 0] = 100.0  # Very concentrated distribution

        # Should freeze after consecutive_threshold layers of divergence
        frozen_happened = False
        for layer_id in range(20):
            frozen = sdd.check_layer(layer_id, concentrated, [0])
            if frozen.any():
                frozen_happened = True
                break

        assert frozen_happened, "Should have frozen with concentrated logits"

    def test_no_freeze_with_uniform_logits(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        config = SDDConfig(
            min_check_layer=0,
            consecutive_threshold=3,
            method="entropy",
            entropy_threshold=1.0,
        )
        sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
        sdd.init_verify_round(num_draft_tokens=1)

        # Uniform logits → high entropy → low divergence score
        uniform = torch.ones(1, 128)

        for layer_id in range(48):
            frozen = sdd.check_layer(layer_id, uniform, [0])
            assert not frozen.any(), f"High entropy should not trigger freeze at layer {layer_id}"

    def test_statistics_tracking(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        config = SDDConfig(min_check_layer=0, consecutive_threshold=2, method="entropy")
        sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
        sdd.init_verify_round(num_draft_tokens=2)

        for layer_id in range(10):
            logits = torch.randn(2, 128)
            sdd.check_layer(layer_id, logits, [0, 1])

        stats = sdd.get_statistics()
        assert stats["total_checks"] > 0
        assert "precision" in stats
        assert "recall" in stats

    def test_report_acceptance(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        config = SDDConfig(min_check_layer=0, consecutive_threshold=1, method="entropy")
        sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
        sdd.init_verify_round(num_draft_tokens=2)

        # Force freeze token 0
        concentrated = torch.zeros(2, 128)
        concentrated[:, 0] = 100.0
        for layer_id in range(5):
            sdd.check_layer(layer_id, concentrated, [0, 1])

        # Report: token 0 was indeed rejected → true positive
        sdd.report_acceptance(0, accepted=False)
        # Report: token 1 was indeed rejected → could be TP or FN
        sdd.report_acceptance(1, accepted=True)

        stats = sdd.get_statistics()
        total = stats["true_positive"] + stats["false_positive"] + stats["true_negative"] + stats["false_negative"]
        assert total == 2

    def test_maf_reduction_estimate(self):
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

        config = SDDConfig(min_check_layer=0, consecutive_threshold=1, method="entropy")
        sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
        sdd.init_verify_round(num_draft_tokens=3)

        # Freeze all tokens early
        concentrated = torch.zeros(3, 128)
        concentrated[:, 0] = 100.0
        for layer_id in range(10):
            sdd.check_layer(layer_id, concentrated, [0, 1, 2])

        estimate = sdd.estimate_maf_reduction(original_maf=2.93, K=3)
        assert estimate["original_maf"] == 2.93
        assert estimate["reduced_maf"] <= estimate["original_maf"]
        assert estimate["reduction_pct"] >= 0
