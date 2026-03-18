#!/usr/bin/env python3
"""
Experiment 4: Expert Temporal Locality 测量
=============================================
验证 Cross-Phase Expert Cache 的可行性 — SD 连续 round 之间 expert 复用程度如何?

验证假设:
  H9:  Inter-round expert overlap > 40%    →  cache 有价值
  H10: Expert reuse distance ≤ 3 rounds    →  cache window 不需要太大
  H11: Draft-target routing correlation > 30%  →  可以用 draft routing 做 prefetch

输入: results/validation/expert_trace.jsonl
输出: results/validation/exp4_locality_report.json

对应 Issue: #37
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


def load_trace(trace_path: str) -> list[dict]:
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def analyze_locality(events: list[dict], K: int, output_dir: Path):
    """Full temporal locality analysis from trace data."""
    import csv
    import numpy as np
    from collectors.expert_locality_analyzer import (
        ExpertTemporalLocalityAnalyzer,
        analyze_from_trace_file,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"hypothesis_tests": {}, "measurements": {}}

    # ── 1. Run locality analyzer ──
    logger.info("=== Temporal Locality Analysis ===")
    analyzer = ExpertTemporalLocalityAnalyzer(num_experts=128, top_k=8, num_layers=48)

    # Group events by request_id
    by_request = defaultdict(list)
    for e in events:
        if e.get("phase", "decode") == "decode":
            by_request[e["request_id"]].append(e)

    round_id = 0
    for req_id, req_events in sorted(by_request.items()):
        req_events.sort(key=lambda x: x["token_idx"])

        # Group events by token_idx within request
        by_token = defaultdict(list)
        for e in req_events:
            by_token[e["token_idx"]].append(e)

        token_ids = sorted(by_token.keys())

        # Simulate SD rounds: each round = K+1 consecutive tokens
        for start in range(0, len(token_ids) - K, K + 1):
            window_tokens = token_ids[start: start + K + 1]
            if len(window_tokens) < K + 1:
                continue

            # Build layer -> [token_experts...] mapping
            layer_experts = defaultdict(list)
            for tid in window_tokens:
                for evt in by_token[tid]:
                    layer_experts[evt["layer_id"]].append(evt["experts"][:8])

            analyzer.record_verify_round(
                round_id=round_id,
                expert_indices=dict(layer_experts),
            )
            round_id += 1

    logger.info(f"Processed {round_id} simulated verify rounds")

    # ── 2. Compute statistics ──
    full_report = analyzer.generate_report()
    stats = analyzer.compute_statistics()

    report["measurements"]["summary"] = full_report["summary"]
    report["measurements"]["recommendations"] = full_report["recommendations"]
    if "reuse_distance_distribution" in full_report:
        report["measurements"]["reuse_distance"] = full_report["reuse_distance_distribution"]
    if "overlap_distribution" in full_report:
        report["measurements"]["overlap_distribution"] = full_report["overlap_distribution"]

    # ── 3. Hypothesis tests ──
    # H9: Inter-round overlap > 40%?
    overlap = stats.mean_interround_overlap
    h9_pass = overlap > 0.40
    report["hypothesis_tests"]["H9_cache_valuable"] = {
        "hypothesis": "Inter-round expert overlap > 40%",
        "mean_overlap": round(overlap, 4),
        "std_overlap": round(stats.std_interround_overlap, 4),
        "pass": h9_pass,
        "interpretation": (
            f"Inter-round overlap = {overlap:.1%}, expert cache 可显著减少重复加载"
            if h9_pass
            else f"Inter-round overlap = {overlap:.1%}, cache 收益有限，但仍可能有价值"
        ),
    }

    # H10: Reuse distance ≤ 3?
    reuse_dist = stats.mean_reuse_distance
    h10_pass = reuse_dist <= 3.0
    report["hypothesis_tests"]["H10_small_cache_window"] = {
        "hypothesis": "Expert 平均复用距离 ≤ 3 rounds",
        "mean_reuse_distance": round(reuse_dist, 2),
        "pass": h10_pass,
        "interpretation": (
            f"平均复用距离 = {reuse_dist:.1f} rounds, 小 cache window (3 rounds) 足够"
            if h10_pass
            else f"平均复用距离 = {reuse_dist:.1f} rounds, 需要较大 cache 或 LRU 策略"
        ),
    }

    # H11: Draft-target correlation > 30%?
    dt_corr = stats.mean_draft_target_correlation
    h11_pass = dt_corr > 0.30
    report["hypothesis_tests"]["H11_draft_prefetch"] = {
        "hypothesis": "Draft-target routing correlation > 30%",
        "correlation": round(dt_corr, 4),
        "pass": h11_pass if dt_corr > 0 else False,
        "interpretation": (
            f"Draft-target correlation = {dt_corr:.1%}, 可用 draft routing 做 prefetch"
            if h11_pass
            else (
                "无 draft routing 数据 (需在 SD 模式下采集)"
                if dt_corr == 0
                else f"Draft-target correlation = {dt_corr:.1%}, prefetch 准确率不够"
            )
        ),
    }

    # ── 4. Cache sizing estimate ──
    logger.info("=== Cache Sizing Estimate ===")
    # With Qwen3-30B: 128 experts/layer, ~9MB per expert
    # GPU memory budget for cache: ~2GB (in 48GB A6000)
    cache_budget_mb = 2048
    expert_size_mb = 3 * 768 * 2048 * 2 / (1024 * 1024)  # ~9MB

    experts_in_cache = int(cache_budget_mb / expert_size_mb)
    experts_per_round = 8 * (K + 1)  # top-k × (K+1) tokens, per layer
    # With overlap, unique per round:
    unique_per_round = experts_per_round * (1 - overlap) if overlap > 0 else experts_per_round

    cache_estimate = {
        "cache_budget_mb": cache_budget_mb,
        "expert_size_mb": round(expert_size_mb, 2),
        "max_cached_experts": experts_in_cache,
        "unique_experts_per_round_per_layer": round(unique_per_round, 1),
        "cache_can_hold_rounds": round(experts_in_cache / max(1, unique_per_round), 1),
        "estimated_hit_rate": round(overlap, 4),
    }
    report["measurements"]["cache_estimate"] = cache_estimate
    logger.info(f"  Cache can hold {experts_in_cache} experts "
                f"(~{experts_in_cache / max(1, unique_per_round):.0f} rounds worth)")

    # ── 5. Expert frequency analysis ──
    logger.info("=== Expert Frequency Distribution ===")
    expert_freq = defaultdict(int)  # (layer, expert_id) -> count
    for e in events:
        if e.get("phase", "decode") == "decode":
            for eid in e["experts"][:8]:
                expert_freq[(e["layer_id"], eid)] += 1

    # Top-20 most used experts
    top_experts = sorted(expert_freq.items(), key=lambda x: -x[1])[:20]
    total_activations = sum(expert_freq.values())
    top_20_share = sum(v for _, v in top_experts) / max(1, total_activations)

    report["measurements"]["expert_concentration"] = {
        "total_unique_experts_activated": len(expert_freq),
        "total_activations": total_activations,
        "top_20_experts_activation_share": round(top_20_share, 4),
        "top_10_experts": [
            {"layer": k[0], "expert_id": k[1], "count": v, "share": round(v / total_activations, 4)}
            for (k, v) in top_experts[:10]
        ],
    }

    # Gini coefficient for expert usage distribution
    if expert_freq:
        import numpy as np
        counts = np.array(sorted(expert_freq.values()))
        n = len(counts)
        gini = (2 * np.sum(np.arange(1, n + 1) * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))
        report["measurements"]["expert_gini_coefficient"] = round(float(gini), 4)
        logger.info(f"  Expert usage Gini = {gini:.4f} (0=uniform, 1=concentrated)")

    # ── 6. Export CSV ──
    analyzer.export_csv(str(output_dir))

    # ── Save report ──
    report_path = output_dir / "exp4_locality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Exp4: Expert Temporal Locality 验证结果")
    print("=" * 60)
    for name, test in report["hypothesis_tests"].items():
        status = "✅ PASS" if test["pass"] else "❌ FAIL"
        print(f"  {status} | {name}: {test['interpretation']}")
    print(f"\n  Cache estimate: {experts_in_cache} experts in {cache_budget_mb}MB, "
          f"hit rate ~{overlap:.1%}")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Exp4: Expert Temporal Locality")
    parser.add_argument("--trace", default="results/validation/expert_trace.jsonl")
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--K", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    events = load_trace(args.trace)
    analyze_locality(events, K=args.K, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
