"""
Tests for MAF computation and Expert Trace Collection
"""

import json
import tempfile
from pathlib import Path

import pytest


def test_compute_theoretical_maf():
    """Test theoretical MAF formula: MAF_random(K) = N*(1-(1-k/N)^(K+1))/k"""
    from collectors.expert_trace_hook import compute_theoretical_maf

    # K=0: MAF should be 1.0 (single token, k experts)
    assert abs(compute_theoretical_maf(K=0, k=8, N=128) - 1.0) < 0.01

    # K=1: MAF(1) = 128*(1-(1-8/128)^2)/8 = 128*(1-0.9375^2)/8
    # = 128*(1-0.8789)/8 = 128*0.1211/8 = 1.9375
    maf1 = compute_theoretical_maf(K=1, k=8, N=128)
    assert 1.9 < maf1 < 2.0, f"MAF(1) = {maf1}"

    # K=2: MAF should be ~2.82
    maf2 = compute_theoretical_maf(K=2, k=8, N=128)
    assert 2.7 < maf2 < 2.9, f"MAF(2) = {maf2}"

    # MAF should increase with K
    prev = 1.0
    for K in range(1, 6):
        maf = compute_theoretical_maf(K=K)
        assert maf > prev, f"MAF({K}) = {maf} should be > MAF({K-1}) = {prev}"
        prev = maf

    # MAF <= K+1 (upper bound: all experts unique)
    for K in range(1, 6):
        maf = compute_theoretical_maf(K=K, k=8, N=128)
        assert maf <= K + 1, f"MAF({K}) = {maf} should be <= {K+1}"


def test_compute_maf_from_trace():
    """Test MAF computation from trace data."""
    from collectors.expert_trace_hook import compute_maf_from_trace

    # Create synthetic trace: 10 tokens, 48 layers, top-8 experts
    events = []
    for token_idx in range(10):
        for layer_id in range(48):
            # Use deterministic experts: token i gets experts [i*8..(i+1)*8-1] mod 128
            base = (token_idx * 3) % 128
            experts = [(base + j) % 128 for j in range(8)]
            events.append({
                "request_id": "req_0",
                "token_idx": token_idx,
                "layer_id": layer_id,
                "experts": experts,
                "router_probs": [0.125] * 8,
                "phase": "decode",
            })

    # K=0: MAF should be 1.0 (trivial: one token window)
    result = compute_maf_from_trace(events, K=0, num_experts_per_tok=8)
    assert abs(result.mean_maf - 1.0) < 0.01, f"MAF(K=0) = {result.mean_maf}"

    # K=1: Two consecutive tokens — check union size
    result = compute_maf_from_trace(events, K=1, num_experts_per_tok=8)
    assert result.mean_maf >= 1.0
    assert result.num_windows > 0

    # K=2: Should have expected statistics
    result = compute_maf_from_trace(events, K=2, num_experts_per_tok=8)
    assert result.num_windows > 0
    assert result.mean_maf >= 1.0
    assert result.std_maf >= 0


def test_compute_maf_prefix_overlap():
    """Test MAF when consecutive tokens share many experts (low MAF)."""
    from collectors.expert_trace_hook import compute_maf_from_trace

    # All tokens use the same 8 experts → MAF should be 1.0
    events = []
    same_experts = list(range(8))
    for token_idx in range(10):
        for layer_id in range(3):  # Fewer layers for speed
            events.append({
                "request_id": "req_0",
                "token_idx": token_idx,
                "layer_id": layer_id,
                "experts": same_experts,
                "phase": "decode",
            })

    result = compute_maf_from_trace(events, K=2, num_experts_per_tok=8)
    assert abs(result.mean_maf - 1.0) < 0.01, f"MAF should be 1.0 when all experts identical, got {result.mean_maf}"


def test_compute_maf_no_overlap():
    """Test MAF when consecutive tokens have zero overlap (max MAF)."""
    from collectors.expert_trace_hook import compute_maf_from_trace

    # Each token uses completely different experts (requires N >= (K+1)*k)
    events = []
    for token_idx in range(5):
        for layer_id in range(3):
            base = token_idx * 8
            experts = list(range(base, base + 8))
            events.append({
                "request_id": "req_0",
                "token_idx": token_idx,
                "layer_id": layer_id,
                "experts": experts,
                "phase": "decode",
            })

    # K=2: 3 tokens × 8 = 24 unique experts → MAF = 24/8 = 3.0
    result = compute_maf_from_trace(events, K=2, num_experts_per_tok=8)
    assert abs(result.mean_maf - 3.0) < 0.01, f"MAF should be 3.0 with no overlap, got {result.mean_maf}"


def test_compute_mmaf_per_token():
    """Test marginal MAF computation."""
    from collectors.expert_trace_hook import compute_mmaf_per_token

    # First token: all 8 experts are new → mmaf = 1.0
    # Second token same experts → mmaf = 0.0
    events = [
        {"request_id": "r0", "token_idx": 0, "layer_id": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7], "phase": "decode"},
        {"request_id": "r0", "token_idx": 1, "layer_id": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7], "phase": "decode"},
    ]

    results = compute_mmaf_per_token(events, num_experts_per_tok=8)
    assert len(results) == 2
    assert abs(results[0]["mmaf"] - 1.0) < 0.01  # First token: all new
    assert abs(results[1]["mmaf"] - 0.0) < 0.01  # Second token: all overlap


def test_maf_result_per_layer():
    """Test per-layer MAF breakdown."""
    from collectors.expert_trace_hook import compute_maf_from_trace

    events = []
    for token_idx in range(5):
        for layer_id in range(3):
            # Different overlap patterns per layer
            if layer_id == 0:
                experts = list(range(8))  # Same for all tokens
            else:
                experts = [(token_idx * 3 + j) % 128 for j in range(8)]
            events.append({
                "request_id": "req_0",
                "token_idx": token_idx,
                "layer_id": layer_id,
                "experts": experts,
                "phase": "decode",
            })

    result = compute_maf_from_trace(events, K=2, num_experts_per_tok=8)
    assert 0 in result.per_layer_maf
    # Layer 0 should have union size 8 (all same)
    assert result.per_layer_maf[0] == 8.0


def test_trace_output_format():
    """Test trace event serialization."""
    from collectors.expert_trace_hook import TraceEvent

    event = TraceEvent(
        request_id="req_0",
        token_idx=5,
        layer_id=10,
        experts=[1, 2, 3, 4, 5, 6, 7, 8],
        router_probs=[0.2, 0.15, 0.12, 0.1, 0.1, 0.1, 0.1, 0.13],
        phase="decode",
    )

    d = event.to_dict()
    assert d["request_id"] == "req_0"
    assert d["token_idx"] == 5
    assert d["layer_id"] == 10
    assert len(d["experts"]) == 8
    assert len(d["router_probs"]) == 8
    assert d["phase"] == "decode"

    # Should be JSON serializable
    json_str = json.dumps(d, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert parsed == d
