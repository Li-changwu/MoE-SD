#!/usr/bin/env python3
"""
Experiment 1: MAF 实证测量与理论对比
======================================
核心假设验证 — 验证 MoE 路由在 SD 场景下是否存在 expert 重叠。

验证假设:
  H1: 真实 MAF(K) < MAF_random(K)  →  连续 token 共享 expert，不是随机路由
  H2: MAF 存在层间差异             →  不同层的路由相关性不同
  H3: mMAF 递减                    →  第 j 个 draft token 的边际贡献递减

关键判据:
  ✅ 若 real MAF(K=3) < 2.82 (理论随机值): 路由有相关性，dedup 有意义
  ❌ 若 real MAF(K=3) ≈ 2.82 或更高: 路由接近随机，dedup benefit 小

输入: results/validation/expert_trace.jsonl (由 exp0 生成)
输出: results/validation/exp1_maf_report.json + CSV files + 画图

对应 Issue: #31, #32
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
    """Load expert trace from JSONL file."""
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    logger.info(f"Loaded {len(events)} trace events from {trace_path}")
    return events


def analyze_maf(events: list[dict], output_dir: Path):
    """Core MAF analysis."""
    import csv
    import numpy as np
    from collectors.expert_trace_hook import (
        compute_maf_from_trace,
        compute_mmaf_per_token,
        compute_theoretical_maf,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"hypothesis_tests": {}, "measurements": {}, "conclusions": []}

    # ── 1. MAF(K) for K=1..5, compare to theoretical ──
    logger.info("=== MAF(K) Measurement ===")
    maf_rows = []
    for K in range(1, 6):
        result = compute_maf_from_trace(events, K=K)
        theoretical = compute_theoretical_maf(K, k=8, N=128)
        ratio = result.mean_maf / theoretical if theoretical > 0 else float("inf")

        row = {
            "K": K,
            "real_MAF": round(result.mean_maf, 4),
            "std_MAF": round(result.std_maf, 4),
            "p25_MAF": round(result.p25_maf, 4),
            "p75_MAF": round(result.p75_maf, 4),
            "theoretical_MAF": round(theoretical, 4),
            "ratio_real_vs_random": round(ratio, 4),
            "num_windows": result.num_windows,
        }
        maf_rows.append(row)
        logger.info(f"  K={K}: real={result.mean_maf:.4f} ± {result.std_maf:.4f}, "
                     f"theoretical={theoretical:.4f}, ratio={ratio:.4f}")

    # Save MAF table
    with open(output_dir / "maf_by_k.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=maf_rows[0].keys())
        w.writeheader()
        w.writerows(maf_rows)

    report["measurements"]["maf_by_k"] = maf_rows

    # H1 test: real MAF(K=3) < theoretical?
    maf_k3 = next(r for r in maf_rows if r["K"] == 3)
    h1_pass = maf_k3["ratio_real_vs_random"] < 0.95  # at least 5% lower
    report["hypothesis_tests"]["H1_routing_correlated"] = {
        "hypothesis": "Real MAF(K=3) < MAF_random(K=3)",
        "real_maf": maf_k3["real_MAF"],
        "theoretical_random_maf": maf_k3["theoretical_MAF"],
        "ratio": maf_k3["ratio_real_vs_random"],
        "pass": h1_pass,
        "interpretation": (
            "路由存在相关性，consecutive tokens 共享 experts，dedup 有额外收益"
            if h1_pass
            else "路由接近随机，dedup 收益主要来自 top-k 的 combinatorial overlap"
        ),
    }

    # ── 2. Per-layer MAF analysis ──
    logger.info("=== Per-layer MAF Analysis ===")
    result_k3 = compute_maf_from_trace(events, K=3)
    per_layer_rows = []
    for layer_id in sorted(result_k3.per_layer_maf.keys()):
        mean_union = result_k3.per_layer_maf[layer_id]
        maf_layer = mean_union / 8.0  # top_k = 8
        per_layer_rows.append({
            "layer": layer_id,
            "mean_union_size": round(mean_union, 2),
            "maf": round(maf_layer, 4),
        })

    with open(output_dir / "maf_per_layer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "mean_union_size", "maf"])
        w.writeheader()
        w.writerows(per_layer_rows)

    # H2 test: is there layer-wise variance?
    if per_layer_rows:
        maf_values = [r["maf"] for r in per_layer_rows]
        layer_std = float(np.std(maf_values))
        layer_min = min(maf_values)
        layer_max = max(maf_values)
        h2_pass = layer_std > 0.05  # meaningful variation
        report["hypothesis_tests"]["H2_layer_variance"] = {
            "hypothesis": "MAF varies significantly across layers",
            "layer_maf_std": round(layer_std, 4),
            "layer_maf_min": round(layer_min, 4),
            "layer_maf_max": round(layer_max, 4),
            "pass": h2_pass,
            "interpretation": (
                "不同层路由相关性不同，可针对 high-MAF 层优先优化"
                if h2_pass
                else "各层 MAF 相似，可统一处理"
            ),
        }
        report["measurements"]["per_layer_maf"] = per_layer_rows

    # ── 3. Marginal MAF (mMAF) analysis ──
    logger.info("=== Marginal MAF (mMAF) Analysis ===")
    mmaf_results = compute_mmaf_per_token(events, num_experts_per_tok=8)

    # Group by token position within sequence
    mmaf_by_position = defaultdict(list)
    for r in mmaf_results:
        mmaf_by_position[r["token_idx"]].append(r["mmaf"])

    # Compute average mMAF by position (first 20 tokens)
    mmaf_trend = []
    for pos in sorted(mmaf_by_position.keys())[:20]:
        values = mmaf_by_position[pos]
        mmaf_trend.append({
            "token_position": pos,
            "mean_mmaf": round(float(np.mean(values)), 4),
            "num_samples": len(values),
        })

    if mmaf_trend:
        with open(output_dir / "mmaf_by_position.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=mmaf_trend[0].keys())
            w.writeheader()
            w.writerows(mmaf_trend)

        # H3: mMAF decreasing?
        if len(mmaf_trend) >= 5:
            first_5 = np.mean([r["mean_mmaf"] for r in mmaf_trend[:5]])
            last_5 = np.mean([r["mean_mmaf"] for r in mmaf_trend[-5:]])
            h3_pass = last_5 < first_5 * 0.9  # at least 10% decrease
            report["hypothesis_tests"]["H3_mmaf_diminishing"] = {
                "hypothesis": "mMAF decreases with token position (diminishing returns)",
                "first_5_mmaf": round(float(first_5), 4),
                "last_5_mmaf": round(float(last_5), 4),
                "pass": h3_pass,
                "interpretation": (
                    "边际 expert 贡献递减，支持 adaptive K 策略"
                    if h3_pass
                    else "mMAF 未明显递减，K 选择对 MAF 影响均匀"
                ),
            }

    # ── 4. Breakeven analysis ──
    logger.info("=== Breakeven Analysis ===")
    # S = (ᾱ(K+1)) / (1 + γ + β(MAF(K) - 1))
    # For speedup > 1: ᾱ > (1 + γ + β(MAF-1)) / (K+1)
    # Assume γ ≈ 0.1 (draft overhead), β ≈ 1.0 (PCIe-bound)
    gamma = 0.1
    beta = 1.0
    breakeven_data = []
    for row in maf_rows:
        K = row["K"]
        maf = row["real_MAF"]
        alpha_min = (1 + gamma + beta * (maf - 1)) / (K + 1)
        breakeven_data.append({
            "K": K,
            "real_MAF": maf,
            "min_acceptance_rate": round(alpha_min, 4),
            "feasible": alpha_min < 0.85,  # typical SD acceptance rate
        })
        logger.info(f"  K={K}: need ᾱ > {alpha_min:.4f} for speedup > 1")

    report["measurements"]["breakeven"] = breakeven_data

    # ── 5. Summary & conclusions ──
    all_pass = all(t.get("pass", False) for t in report["hypothesis_tests"].values())
    if all_pass:
        report["conclusions"].append("✅ 所有假设验证通过: MoE 路由存在显著相关性, SpecMoE 优化方向可行")
    else:
        failed = [k for k, v in report["hypothesis_tests"].items() if not v.get("pass", False)]
        report["conclusions"].append(f"⚠️ 以下假设未通过: {failed}, 需重新评估技术路线")

    # Add paper data points
    report["paper_ready_data"] = {
        "table1_maf_by_k": maf_rows,
        "figure2_per_layer_maf": per_layer_rows[:10] if per_layer_rows else [],  # sample
        "key_numbers": {
            "MAF_K3_real": maf_k3["real_MAF"],
            "MAF_K3_theoretical": maf_k3["theoretical_MAF"],
            "MAF_ratio": maf_k3["ratio_real_vs_random"],
            "min_acceptance_K3": next(
                (b["min_acceptance_rate"] for b in breakeven_data if b["K"] == 3), None
            ),
        },
    }

    # Save report
    report_path = output_dir / "exp1_maf_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Exp1: MAF 假设验证结果")
    print("=" * 60)
    for name, test in report["hypothesis_tests"].items():
        status = "✅ PASS" if test["pass"] else "❌ FAIL"
        print(f"  {status} | {name}: {test['interpretation']}")
    print()
    for c in report["conclusions"]:
        print(f"  {c}")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Exp1: MAF Measurement & Analysis")
    parser.add_argument("--trace", default="results/validation/expert_trace.jsonl",
                        help="Path to expert trace JSONL from exp0")
    parser.add_argument("--output-dir", default="results/validation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    events = load_trace(args.trace)
    analyze_maf(events, Path(args.output_dir))


if __name__ == "__main__":
    main()
