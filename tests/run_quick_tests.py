#!/usr/bin/env python3
"""Quick test runner for SpecMoE core modules."""
import sys
sys.path.insert(0, '/root/MoE-SD')

def test_maf():
    from collectors.expert_trace_hook import compute_theoretical_maf, compute_maf_from_trace, compute_mmaf_per_token

    print('=== Theoretical MAF ===')
    for K in range(6):
        maf = compute_theoretical_maf(K)
        print(f'  MAF(K={K}) = {maf:.4f}')

    assert abs(compute_theoretical_maf(0) - 1.0) < 0.01
    maf2 = compute_theoretical_maf(2)
    assert 2.7 < maf2 < 2.9, f"MAF(2) = {maf2}"

    # Same experts → MAF=1
    same_events = []
    for t in range(10):
        for l in range(3):
            same_events.append({'request_id': 'r0', 'token_idx': t, 'layer_id': l, 'experts': list(range(8)), 'phase': 'decode'})
    r = compute_maf_from_trace(same_events, K=2)
    assert abs(r.mean_maf - 1.0) < 0.01, f"expected 1.0, got {r.mean_maf}"
    print(f'  Same experts MAF(K=2) = {r.mean_maf:.4f} ✓')

    # No overlap → MAF=K+1
    non_events = []
    for t in range(5):
        for l in range(3):
            non_events.append({'request_id': 'r0', 'token_idx': t, 'layer_id': l, 'experts': list(range(t*8, t*8+8)), 'phase': 'decode'})
    r = compute_maf_from_trace(non_events, K=2)
    assert abs(r.mean_maf - 3.0) < 0.01, f"expected 3.0, got {r.mean_maf}"
    print(f'  No overlap MAF(K=2) = {r.mean_maf:.4f} ✓')

    # mMAF
    test_events = [
        {'request_id': 'r0', 'token_idx': 0, 'layer_id': 0, 'experts': list(range(8)), 'phase': 'decode'},
        {'request_id': 'r0', 'token_idx': 1, 'layer_id': 0, 'experts': list(range(8)), 'phase': 'decode'},
    ]
    mmaf = compute_mmaf_per_token(test_events)
    assert abs(mmaf[0]['mmaf'] - 1.0) < 0.01
    assert abs(mmaf[1]['mmaf'] - 0.0) < 0.01
    print(f'  mMAF test ✓')

    print('MAF TESTS PASSED ✓\n')


def test_spec_fused_moe():
    import torch
    from adapters.spec_fused_moe import SpecFusedMoE, DedupAnalyzer

    ne, tk, hs, inter = 16, 4, 32, 64
    weights = {}
    for eid in range(ne):
        weights[eid] = {
            'gate_proj': torch.randn(inter, hs),
            'up_proj': torch.randn(inter, hs),
            'down_proj': torch.randn(hs, inter),
        }

    model = SpecFusedMoE(num_experts=ne, top_k=tk, hidden_size=hs, moe_intermediate_size=inter)

    # Correctness: single token
    h = torch.randn(1, hs)
    l = torch.randn(1, ne)
    out = model(h, l, weights)
    assert out.shape == (1, hs), f"Wrong shape: {out.shape}"
    print('  Single token forward ✓')

    # Batch correctness
    model.reset_statistics()
    h4 = torch.randn(4, hs)
    l4 = torch.randn(4, ne)
    out4 = model(h4, l4, weights)
    assert out4.shape == (4, hs)
    print(f'  Batch forward ✓ (stats: {model.get_statistics()})')

    # Dedup with identical logits
    model.reset_statistics()
    same_l = torch.randn(1, ne).expand(4, -1)
    model(torch.randn(4, hs), same_l, weights)
    s = model.get_statistics()
    assert s['total_dedup_loads'] == tk, f"Dedup should yield {tk} loads, got {s['total_dedup_loads']}"
    assert s['total_naive_loads'] == 4 * tk
    print(f'  Dedup with identical logits ✓ (naive={s["total_naive_loads"]}, dedup={s["total_dedup_loads"]})')

    # Frozen mask
    model.reset_statistics()
    h4 = torch.randn(4, hs)
    l4 = torch.randn(4, ne)
    frozen = torch.tensor([False, True, False, True])
    out = model(h4, l4, weights, frozen_mask=frozen)
    assert torch.equal(out[1], h4[1]), "Frozen token should be unchanged"
    assert torch.equal(out[3], h4[3]), "Frozen token should be unchanged"
    print('  Frozen mask ✓')

    # DedupAnalyzer
    analyzer = DedupAnalyzer(num_experts=128, top_k=8)
    indices = torch.tensor([[0,1,2,3,4,5,6,7]]*3)
    r = analyzer.analyze_verify_batch(indices)
    assert r['dedup_loads'] == 8
    print(f'  DedupAnalyzer ✓ (savings={r["savings_pct"]}%)')

    print('SPEC_FUSED_MOE TESTS PASSED ✓\n')


def test_sdd():
    import torch
    from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

    # Basic init
    config = SDDConfig(min_check_layer=0, consecutive_threshold=2, method='entropy')
    sdd = SpeculationDivergenceDetector(config=config, num_layers=48)
    sdd.init_verify_round(num_draft_tokens=2)
    mask = sdd.get_frozen_mask()
    assert len(mask) == 2
    print('  Init ✓')

    # No freeze with uniform logits
    config2 = SDDConfig(min_check_layer=0, consecutive_threshold=3, method='entropy', entropy_threshold=1.0)
    sdd2 = SpeculationDivergenceDetector(config=config2, num_layers=48)
    sdd2.init_verify_round(num_draft_tokens=1)
    for layer in range(48):
        frozen = sdd2.check_layer(layer, torch.ones(1, 128), [0])
        assert not frozen.any(), f"Uniform should not freeze at layer {layer}"
    print('  Uniform no-freeze ✓')

    # Concentrated logits should eventually freeze
    config3 = SDDConfig(min_check_layer=0, consecutive_threshold=2, method='entropy', entropy_threshold=0.5)
    sdd3 = SpeculationDivergenceDetector(config=config3, num_layers=48)
    sdd3.init_verify_round(num_draft_tokens=1)
    concentrated = torch.zeros(1, 128)
    concentrated[0, 0] = 100.0
    frozen_happened = False
    for layer in range(20):
        f = sdd3.check_layer(layer, concentrated, [0])
        if f.any():
            frozen_happened = True
            print(f'  Concentrated froze at layer {layer} ✓')
            break
    assert frozen_happened

    # Statistics
    stats = sdd3.get_statistics()
    assert stats['total_freezes'] > 0
    print(f'  Statistics: {stats}')

    # MAF reduction estimate
    est = sdd3.estimate_maf_reduction(original_maf=2.93, K=3)
    assert est['reduced_maf'] <= 2.93
    print(f'  MAF reduction: {est["original_maf"]} → {est["reduced_maf"]} ({est["reduction_pct"]}%)')

    print('SDD TESTS PASSED ✓\n')


def test_locality_analyzer():
    from collectors.expert_locality_analyzer import ExpertTemporalLocalityAnalyzer
    import numpy as np

    analyzer = ExpertTemporalLocalityAnalyzer(num_experts=128, top_k=8, num_layers=3)

    # Simulate 10 verify rounds with overlapping experts
    for round_id in range(10):
        layer_experts = {}
        for layer_id in range(3):
            token_lists = []
            for t in range(4):  # K=3 → 4 tokens
                base = (round_id * 2 + t) % 128
                token_lists.append([(base + j) % 128 for j in range(8)])
            layer_experts[layer_id] = token_lists
        analyzer.record_verify_round(round_id=round_id, expert_indices=layer_experts)

    report = analyzer.generate_report()
    print(f'  Overlap: {report["summary"]["inter_round_overlap"]:.4f}')
    print(f'  Reuse distance: {report["summary"]["mean_reuse_distance"]:.2f}')
    print(f'  Recommendations: {report["recommendations"]}')

    stats = analyzer.compute_statistics()
    assert stats.num_rounds == 10
    assert stats.mean_interround_overlap > 0
    print('LOCALITY ANALYZER TESTS PASSED ✓\n')


if __name__ == '__main__':
    print('='*60)
    print('SpecMoE Core Module Tests')
    print('='*60)
    print()

    test_maf()
    test_spec_fused_moe()
    test_sdd()
    test_locality_analyzer()

    print('='*60)
    print('ALL TESTS PASSED ✓')
    print('='*60)
