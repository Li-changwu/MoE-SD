#!/usr/bin/env python3
"""
Experiment 2: Expert 去重收益估算 + PCIe 带宽分析
==================================================
验证 SpecFusedMoE 的核心价值 — dedup 到底能省多少 expert weight 加载?

验证假设:
  H4: Dedup 可省 >15% expert loads  →  SpecFusedMoE 有实际意义
  H5: PCIe 是 offload 场景瓶颈        →  减少 expert 传输直接降延迟

实验内容:
  1. 从 trace 数据模拟 verify batch，计算去重后的 unique expert 数量
  2. 对比 naive (K+1)×k 加载 vs dedup |∪E_i| 加载
  3. 估算 PCIe 带宽节省 (每 expert ≈ 6MB for Qwen3-30B)
  4. 微基准: 实际测量 CPU→GPU expert 传输延迟

输入: results/validation/expert_trace.jsonl
输出: results/validation/exp2_dedup_report.json

对应 Issue: #33, #34
"""

import argparse
import json
import logging
import sys
import time
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


def analyze_dedup_from_trace(events: list[dict], output_dir: Path):
    """Analyze dedup savings from real trace data."""
    import csv
    import numpy as np
    from adapters.spec_fused_moe import DedupAnalyzer

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = DedupAnalyzer(num_experts=128, top_k=8)
    report = {"hypothesis_tests": {}, "measurements": {}}

    # ── 1. Dedup savings for K=1..5 ──
    logger.info("=== Dedup Savings by K ===")
    dedup_rows = []
    for K in range(1, 6):
        result = analyzer.analyze_from_trace(events, K=K)
        if "error" in result:
            logger.warning(f"  K={K}: {result['error']}")
            continue

        row = {
            "K": K,
            "mean_savings_pct": result["mean_savings_pct"],
            "mean_maf": result["mean_maf"],
            "std_maf": result["std_maf"],
            "p25_maf": result["p25_maf"],
            "p75_maf": result["p75_maf"],
            "max_savings_pct": result["max_savings_pct"],
            "min_savings_pct": result["min_savings_pct"],
            "num_windows": result["num_windows"],
        }
        dedup_rows.append(row)
        logger.info(f"  K={K}: mean savings = {result['mean_savings_pct']:.1f}%, "
                     f"MAF = {result['mean_maf']:.4f} ± {result['std_maf']:.4f}")

    with open(output_dir / "dedup_savings_by_k.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dedup_rows[0].keys())
        w.writeheader()
        w.writerows(dedup_rows)

    report["measurements"]["dedup_by_k"] = dedup_rows

    # H4: Dedup saves >15% at K=3?
    dedup_k3 = next((r for r in dedup_rows if r["K"] == 3), None)
    if dedup_k3:
        h4_pass = dedup_k3["mean_savings_pct"] > 15.0
        report["hypothesis_tests"]["H4_dedup_meaningful"] = {
            "hypothesis": "Dedup 节省 >15% expert loads at K=3",
            "savings_pct": dedup_k3["mean_savings_pct"],
            "pass": h4_pass,
            "interpretation": (
                f"去重有效，平均节省 {dedup_k3['mean_savings_pct']:.1f}% 的 expert 加载"
                if h4_pass
                else f"去重收益有限 ({dedup_k3['mean_savings_pct']:.1f}%)，可能需要更大 K 或其他优化"
            ),
        }

    # ── 2. Per-layer dedup savings ──
    logger.info("=== Per-layer Dedup Analysis ===")
    K = 3
    grouped = defaultdict(list)
    for e in events:
        if e.get("phase", "decode") == "decode":
            key = (e["request_id"], e["layer_id"])
            grouped[key].append(e)

    per_layer_savings = defaultdict(list)
    for (req_id, layer_id), layer_events in grouped.items():
        layer_events.sort(key=lambda x: x["token_idx"])
        for start in range(len(layer_events) - K):
            window = layer_events[start: start + K + 1]
            all_experts = set()
            for evt in window:
                all_experts.update(evt["experts"][:8])
            naive = (K + 1) * 8
            dedup = len(all_experts)
            savings = (1 - dedup / naive) * 100
            per_layer_savings[layer_id].append(savings)

    per_layer_rows = []
    for layer_id in sorted(per_layer_savings.keys()):
        values = per_layer_savings[layer_id]
        per_layer_rows.append({
            "layer": layer_id,
            "mean_savings_pct": round(float(np.mean(values)), 2),
            "std_savings_pct": round(float(np.std(values)), 2),
            "num_windows": len(values),
        })

    if per_layer_rows:
        with open(output_dir / "dedup_per_layer.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_layer_rows[0].keys())
            w.writeheader()
            w.writerows(per_layer_rows)
        report["measurements"]["dedup_per_layer_sample"] = per_layer_rows[:5]

    # ── 3. PCIe bandwidth estimation ──
    logger.info("=== PCIe Bandwidth Impact ===")
    # Qwen3-30B expert size: gate(768×2048) + up(768×2048) + down(2048×768) = 3×768×2048 = 4,718,592 params
    # In bf16: 4,718,592 × 2 = 9,437,184 bytes ≈ 9.0 MB per expert
    expert_size_bytes = 3 * 768 * 2048 * 2  # bf16
    expert_size_mb = expert_size_bytes / (1024 * 1024)

    pcie_bandwidth_gbps = 32.0  # PCIe 4.0 x16 theoretical

    bandwidth_rows = []
    for row in dedup_rows:
        K = row["K"]
        naive_experts_per_layer = (K + 1) * 8
        dedup_experts_per_layer = row["mean_maf"] * 8

        naive_bytes = naive_experts_per_layer * expert_size_bytes * 48  # 48 layers
        dedup_bytes = dedup_experts_per_layer * expert_size_bytes * 48

        naive_time_ms = naive_bytes / (pcie_bandwidth_gbps * 1e9 / 8) * 1000
        dedup_time_ms = dedup_bytes / (pcie_bandwidth_gbps * 1e9 / 8) * 1000
        saved_ms = naive_time_ms - dedup_time_ms

        bandwidth_rows.append({
            "K": K,
            "naive_experts_total": round(naive_experts_per_layer * 48),
            "dedup_experts_total": round(dedup_experts_per_layer * 48, 1),
            "naive_transfer_mb": round(naive_bytes / 1e6, 1),
            "dedup_transfer_mb": round(dedup_bytes / 1e6, 1),
            "naive_time_ms": round(naive_time_ms, 2),
            "dedup_time_ms": round(dedup_time_ms, 2),
            "saved_time_ms": round(saved_ms, 2),
        })
        logger.info(f"  K={K}: naive={naive_time_ms:.1f}ms, dedup={dedup_time_ms:.1f}ms, "
                     f"saved={saved_ms:.1f}ms per verify step")

    report["measurements"]["bandwidth_estimation"] = bandwidth_rows
    report["measurements"]["expert_size_mb"] = round(expert_size_mb, 2)

    # ── 4. PCIe micro-benchmark (optional, GPU only) ──
    pcie_measured = run_pcie_microbench()
    if pcie_measured:
        report["measurements"]["pcie_microbench"] = pcie_measured
        # H5: PCIe is bottleneck?
        transfer_frac = pcie_measured.get("transfer_fraction", 0)
        h5_pass = transfer_frac > 0.5
        report["hypothesis_tests"]["H5_pcie_bottleneck"] = {
            "hypothesis": "PCIe 传输占 MoE 推理 >50% 时间",
            "transfer_fraction": transfer_frac,
            "pass": h5_pass,
            "interpretation": (
                f"PCIe 传输占 {transfer_frac*100:.0f}%，是 offload 场景主要瓶颈"
                if h5_pass
                else f"PCIe 传输占 {transfer_frac*100:.0f}%，计算也是瓶颈"
            ),
        }

    # ── Summary ──
    report_path = output_dir / "exp2_dedup_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Exp2: Dedup 收益估算结果")
    print("=" * 60)
    for name, test in report["hypothesis_tests"].items():
        status = "✅ PASS" if test["pass"] else "❌ FAIL"
        print(f"  {status} | {name}: {test['interpretation']}")
    if bandwidth_rows:
        bw_k3 = next((r for r in bandwidth_rows if r["K"] == 3), bandwidth_rows[0])
        print(f"\n  Key: K=3 verify step saves ~{bw_k3['saved_time_ms']:.1f}ms "
              f"({bw_k3['dedup_transfer_mb']:.0f}MB vs {bw_k3['naive_transfer_mb']:.0f}MB)")
    print("=" * 60)

    return report


def run_pcie_microbench() -> dict | None:
    """Micro-benchmark: CPU→GPU transfer for expert-sized tensors."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("No CUDA available, skipping PCIe microbench")
            return None

        logger.info("=== PCIe Micro-benchmark ===")
        # Expert size: 3 × 768 × 2048 params in bf16
        expert_params = 3 * 768 * 2048
        expert_tensor = torch.randn(expert_params, dtype=torch.bfloat16, device="cpu")

        # Warmup
        for _ in range(5):
            _ = expert_tensor.to("cuda", non_blocking=False)
            torch.cuda.synchronize()

        # Measure transfer time
        transfer_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_tensor = expert_tensor.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            transfer_times.append((t1 - t0) * 1000)  # ms
            del gpu_tensor

        # Measure compute time (matmul as proxy for expert FFN)
        x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(768, 2048, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            _ = x @ w.T
            torch.cuda.synchronize()

        compute_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = x @ w.T
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            compute_times.append((t1 - t0) * 1000)

        import numpy as np
        mean_transfer = float(np.mean(transfer_times))
        mean_compute = float(np.mean(compute_times))
        transfer_fraction = mean_transfer / (mean_transfer + mean_compute)

        result = {
            "expert_size_bytes": expert_params * 2,
            "mean_transfer_ms": round(mean_transfer, 4),
            "std_transfer_ms": round(float(np.std(transfer_times)), 4),
            "mean_compute_ms": round(mean_compute, 4),
            "transfer_fraction": round(transfer_fraction, 4),
            "effective_bandwidth_gbps": round(
                expert_params * 2 / mean_transfer / 1e6, 2
            ),
        }
        logger.info(f"  Transfer: {mean_transfer:.3f}ms, Compute: {mean_compute:.3f}ms, "
                     f"Transfer fraction: {transfer_fraction:.1%}")
        return result

    except Exception as e:
        logger.warning(f"PCIe microbench failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Exp2: Dedup Savings Estimation")
    parser.add_argument("--trace", default="results/validation/expert_trace.jsonl")
    parser.add_argument("--output-dir", default="results/validation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    events = load_trace(args.trace)
    analyze_dedup_from_trace(events, Path(args.output_dir))


if __name__ == "__main__":
    main()
