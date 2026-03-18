"""
Tests for SpecFusedMoE operator
"""

import torch
import pytest


def _make_expert_weights(num_experts, hidden_size, intermediate_size, device="cpu"):
    """Create random expert weights for testing."""
    weights = {}
    for eid in range(num_experts):
        weights[eid] = {
            "gate_proj": torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32),
            "up_proj": torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32),
            "down_proj": torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float32),
        }
    return weights


def _vanilla_moe_forward(hidden_states, router_logits, expert_weights, top_k=8, norm_topk_prob=True):
    """Reference vanilla MoE implementation for correctness checking."""
    routing_weights = torch.softmax(router_logits.float(), dim=-1)
    top_weights, top_indices = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(hidden_states)
    for t in range(hidden_states.shape[0]):
        for s in range(top_k):
            eid = top_indices[t, s].item()
            w = top_weights[t, s]
            params = expert_weights[eid]
            gate_out = torch.nn.functional.silu(hidden_states[t] @ params["gate_proj"].T)
            up_out = hidden_states[t] @ params["up_proj"].T
            expert_out = (gate_out * up_out) @ params["down_proj"].T
            output[t] += w * expert_out
    return output


class TestSpecFusedMoE:
    """Test suite for SpecFusedMoE operator."""

    def setup_method(self):
        self.num_experts = 16  # Small for testing
        self.top_k = 4
        self.hidden_size = 32
        self.intermediate_size = 64
        self.weights = _make_expert_weights(
            self.num_experts, self.hidden_size, self.intermediate_size
        )

    def test_correctness_single_token(self):
        """SpecFusedMoE should produce same output as vanilla MoE for single token."""
        from adapters.spec_fused_moe import SpecFusedMoE

        model = SpecFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
        )

        hidden = torch.randn(1, self.hidden_size)
        logits = torch.randn(1, self.num_experts)

        spec_output = model(hidden, logits, self.weights)
        vanilla_output = _vanilla_moe_forward(hidden, logits, self.weights, self.top_k)

        assert torch.allclose(spec_output, vanilla_output, atol=1e-5), \
            f"Max diff: {(spec_output - vanilla_output).abs().max().item()}"

    def test_correctness_verify_batch(self):
        """SpecFusedMoE should produce same output as vanilla MoE for K+1 token batch."""
        from adapters.spec_fused_moe import SpecFusedMoE

        model = SpecFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
        )

        K = 3
        batch_size = K + 1
        hidden = torch.randn(batch_size, self.hidden_size)
        logits = torch.randn(batch_size, self.num_experts)

        spec_output = model(hidden, logits, self.weights)
        vanilla_output = _vanilla_moe_forward(hidden, logits, self.weights, self.top_k)

        assert torch.allclose(spec_output, vanilla_output, atol=1e-5), \
            f"Max diff: {(spec_output - vanilla_output).abs().max().item()}"

    def test_dedup_statistics(self):
        """Verify dedup reduces load count when tokens share experts."""
        from adapters.spec_fused_moe import SpecFusedMoE

        model = SpecFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
        )

        # Use logits that route all tokens to the same experts
        K = 3
        batch_size = K + 1
        hidden = torch.randn(batch_size, self.hidden_size)
        # Same logits for all tokens → same experts → max dedup
        single_logits = torch.randn(1, self.num_experts)
        logits = single_logits.expand(batch_size, -1)

        model(hidden, logits, self.weights)
        stats = model.get_statistics()

        assert stats["total_calls"] == 1
        # All tokens use same experts → dedup_loads = top_k
        assert stats["total_dedup_loads"] == self.top_k
        assert stats["total_naive_loads"] == batch_size * self.top_k
        assert stats["dedup_ratio"] > 0

    def test_frozen_mask(self):
        """Frozen tokens should be passed through unchanged."""
        from adapters.spec_fused_moe import SpecFusedMoE

        model = SpecFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
        )

        batch_size = 4
        hidden = torch.randn(batch_size, self.hidden_size)
        logits = torch.randn(batch_size, self.num_experts)
        frozen = torch.tensor([False, True, False, True])

        output = model(hidden, logits, self.weights, frozen_mask=frozen)

        # Frozen tokens should be unchanged
        assert torch.equal(output[1], hidden[1])
        assert torch.equal(output[3], hidden[3])
        # Active tokens should be different from input
        assert not torch.equal(output[0], hidden[0])
        assert not torch.equal(output[2], hidden[2])

    def test_acceptance_priority_ordering(self):
        """With acceptance_probs, output should be same regardless of ordering."""
        from adapters.spec_fused_moe import SpecFusedMoE

        model = SpecFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
        )

        batch_size = 4
        hidden = torch.randn(batch_size, self.hidden_size)
        logits = torch.randn(batch_size, self.num_experts)
        probs = torch.tensor([0.3, 0.9, 0.1, 0.7])

        output_with_probs = model(hidden, logits, self.weights, acceptance_probs=probs)
        model.reset_statistics()
        output_without = model(hidden, logits, self.weights)

        # Outputs should be identical (priority only affects processing order)
        assert torch.allclose(output_with_probs, output_without, atol=1e-5)

    def test_reset_statistics(self):
        """Statistics should be cleared on reset."""
        from adapters.spec_fused_moe import SpecFusedMoE

        model = SpecFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
        )

        hidden = torch.randn(2, self.hidden_size)
        logits = torch.randn(2, self.num_experts)
        model(hidden, logits, self.weights)
        assert model.get_statistics()["total_calls"] > 0

        model.reset_statistics()
        stats = model.get_statistics()
        assert stats["total_calls"] == 0
        assert stats["total_naive_loads"] == 0
        assert stats["total_dedup_loads"] == 0


class TestDedupAnalyzer:
    """Test DedupAnalyzer for trace-based analysis."""

    def test_analyze_verify_batch(self):
        from adapters.spec_fused_moe import DedupAnalyzer

        analyzer = DedupAnalyzer(num_experts=128, top_k=8)

        # K=2: 3 tokens, each with 8 experts
        # All same experts → dedup saves a lot
        indices = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ])
        result = analyzer.analyze_verify_batch(indices)
        assert result["K"] == 2
        assert result["dedup_loads"] == 8
        assert result["naive_loads"] == 24
        assert result["savings_pct"] > 60

    def test_analyze_no_overlap(self):
        from adapters.spec_fused_moe import DedupAnalyzer

        analyzer = DedupAnalyzer(num_experts=128, top_k=8)

        indices = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
        ])
        result = analyzer.analyze_verify_batch(indices)
        assert result["dedup_loads"] == 24  # No overlap
        assert result["savings_pct"] == 0.0

    def test_analyze_from_trace(self):
        from adapters.spec_fused_moe import DedupAnalyzer

        analyzer = DedupAnalyzer(num_experts=128, top_k=8)

        events = []
        for token_idx in range(10):
            for layer_id in range(3):
                base = (token_idx * 2) % 128
                experts = [(base + j) % 128 for j in range(8)]
                events.append({
                    "request_id": "req_0",
                    "token_idx": token_idx,
                    "layer_id": layer_id,
                    "experts": experts,
                    "phase": "decode",
                })

        result = analyzer.analyze_from_trace(events, K=2)
        assert "mean_maf" in result
        assert "mean_savings_pct" in result
        assert result["num_windows"] > 0
