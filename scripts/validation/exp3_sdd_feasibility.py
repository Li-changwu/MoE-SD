#!/usr/bin/env python3
"""
Experiment 3: SDD (Speculation Divergence Detector) 可行性验证
================================================================
验证 Layer-wise Early Termination 的核心前提 — router logits 能否预测 token rejection?

验证假设:
  H6: Router logits 分歧可区分 accepted/rejected tokens  →  SDD 有预测价值
  H7: SDD 能在 L/2 层之前冻结             →  计算节省超过 50%
  H8: SDD precision > 80%                  →  误杀 accepted token 的代价可接受

实验方法:
  1. 在 vLLM SD 模式下收集 verify phase 的:
     - 每层 router logits (from exp0 trace)
     - 每个 draft token 的 acceptance/rejection label
  2. 离线回放: 用 SDD 在每层做判定，对比真实 accept/reject
  3. 搜索最佳阈值 (grid search over SDD config)

注意: 此实验需要同时有 trace 数据和 acceptance labels。
     exp0 只采集了 offline transformers trace (无 SD acceptance)。
     因此这里提供两种模式:
       (a) Offline 模拟: 用 trace 数据 + 模拟 acceptance (基于 top-k overlap 作为 proxy)
       (b) Online 采集: 在 vLLM SD serving 中同时记录 acceptance

输入: results/validation/expert_trace.jsonl
输出: results/validation/exp3_sdd_report.json

对应 Issue: #35, #36
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


def simulate_acceptance_labels(events: list[dict], K: int = 3) -> list[dict]:
    """
    模拟 acceptance labels 作为 SDD 可行性的 proxy。

    策略: 相邻 token 的 top-k expert overlap 高 → 视为 accepted (路由一致)
          overlap 低 → 视为 rejected (路由不一致)

    这是一个 proxy，真正的 acceptance 需要在 SD verify 中采集。
    但 routing overlap 和 acceptance 有正相关性:
    - Draft model 若能预测到 target 的路由，说明 hidden state 相似
    - Hidden state 相似 → token 更可能被 accept

    阈值: overlap < 0.3 → rejected
    """
    # Group events by (request_id, layer_id)
    grouped = defaultdict(list)
    for e in events:
        if e.get("phase", "decode") == "decode":
            key = (e["request_id"],)
            grouped[key].append(e)

    labelled_windows = []

    for (req_id,), req_events in grouped.items():
        # Group by token_idx to get per-token routing across layers
        by_token = defaultdict(dict)
        for e in req_events:
            by_token[e["token_idx"]][e["layer_id"]] = set(e["experts"][:8])

        token_ids = sorted(by_token.keys())
        if len(token_ids) < K + 1:
            continue

        # Slide window of K+1 tokens
        for start in range(0, len(token_ids) - K, K + 1):
            window_tokens = token_ids[start: start + K + 1]
            if len(window_tokens) < K + 1:
                continue

            # Target token is the first (t_0), draft tokens are t_1..t_K
            target_routing = by_token[window_tokens[0]]

            draft_labels = []
            for j in range(1, len(window_tokens)):
                draft_routing = by_token[window_tokens[j]]

                # Compute overlap between draft token and previous token
                overlaps = []
                prev_routing = by_token[window_tokens[j - 1]]
                for layer_id in draft_routing:
                    if layer_id in prev_routing:
                        curr = draft_routing[layer_id]
                        prev = prev_routing[layer_id]
                        if curr and prev:
                            overlap = len(curr & prev) / max(1, len(curr | prev))
                            overlaps.append(overlap)

                mean_overlap = sum(overlaps) / max(1, len(overlaps))
                # Simulate: low overlap → likely rejected
                accepted = mean_overlap > 0.3
                draft_labels.append({
                    "token_idx": window_tokens[j],
                    "position_in_draft": j,
                    "overlap_with_prev": round(mean_overlap, 4),
                    "simulated_accepted": accepted,
                })

            labelled_windows.append({
                "request_id": req_id,
                "window_start": window_tokens[0],
                "K": K,
                "drafts": draft_labels,
                "per_layer_routing": {
                    tid: {lid: list(experts) for lid, experts in by_token[tid].items()}
                    for tid in window_tokens
                },
            })

    logger.info(f"Generated {len(labelled_windows)} labelled windows "
                f"({sum(len(w['drafts']) for w in labelled_windows)} draft tokens)")
    return labelled_windows


def evaluate_sdd_on_windows(windows: list[dict], output_dir: Path):
    """Evaluate SDD with different configs on labelled windows."""
    import csv
    import numpy as np
    import torch
    from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"hypothesis_tests": {}, "measurements": {}, "grid_search": []}

    # ── Grid search over SDD configs ──
    configs = [
        {"method": "entropy", "min_check_layer": 8, "consecutive_threshold": 2, "entropy_threshold": 0.5},
        {"method": "entropy", "min_check_layer": 8, "consecutive_threshold": 3, "entropy_threshold": 0.5},
        {"method": "entropy", "min_check_layer": 4, "consecutive_threshold": 2, "entropy_threshold": 0.3},
        {"method": "entropy", "min_check_layer": 12, "consecutive_threshold": 2, "entropy_threshold": 0.5},
        {"method": "entropy", "min_check_layer": 8, "consecutive_threshold": 2, "entropy_threshold": 0.8},
        {"method": "combined", "min_check_layer": 8, "consecutive_threshold": 2},
        {"method": "combined", "min_check_layer": 8, "consecutive_threshold": 3},
        {"method": "overlap", "min_check_layer": 8, "consecutive_threshold": 2, "topk_overlap_threshold": 0.25},
        {"method": "overlap", "min_check_layer": 8, "consecutive_threshold": 3, "topk_overlap_threshold": 0.3},
    ]

    logger.info(f"=== SDD Grid Search ({len(configs)} configs) ===")
    best_f1 = -1
    best_config = None

    for cfg_dict in configs:
        sdd_config = SDDConfig(**cfg_dict)
        tp, fp, tn, fn = 0, 0, 0, 0
        freeze_layers = []

        for window in windows:
            K = window["K"]
            routing_data = window["per_layer_routing"]

            for draft in window["drafts"]:
                t_idx = draft["token_idx"]
                accepted = draft["simulated_accepted"]

                # Reconstruct router logits from expert indices (one-hot proxy)
                sdd = SpeculationDivergenceDetector(config=sdd_config, num_layers=48)
                sdd.init_verify_round(num_draft_tokens=1)

                frozen = False
                freeze_layer = 48

                token_routing = routing_data.get(str(t_idx), routing_data.get(t_idx, {}))
                for layer_id in range(48):
                    layer_key = str(layer_id) if str(layer_id) in token_routing else layer_id
                    if layer_key in token_routing:
                        # Construct proxy logits: high values for selected experts
                        logits = torch.zeros(1, 128)
                        for eid in token_routing[layer_key]:
                            if eid < 128:
                                logits[0, eid] = 5.0 + torch.randn(1).item() * 0.5
                        # Add small noise to non-selected
                        logits += torch.randn_like(logits) * 0.1

                        mask = sdd.check_layer(layer_id, logits, [0])
                        if mask.any():
                            frozen = True
                            freeze_layer = layer_id
                            break

                # Evaluate
                if frozen and not accepted:
                    tp += 1
                elif frozen and accepted:
                    fp += 1
                elif not frozen and not accepted:
                    fn += 1
                else:
                    tn += 1

                if frozen:
                    freeze_layers.append(freeze_layer)

        total = tp + fp + tn + fn
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        avg_freeze = np.mean(freeze_layers) if freeze_layers else 48.0

        result = {
            "config": cfg_dict,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "freeze_rate": round(len(freeze_layers) / max(1, total), 4),
            "avg_freeze_layer": round(float(avg_freeze), 1),
            "total_evaluated": total,
        }
        report["grid_search"].append(result)

        logger.info(f"  {cfg_dict['method']:8s} min_layer={cfg_dict['min_check_layer']:2d} "
                     f"consec={cfg_dict['consecutive_threshold']}: "
                     f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
                     f"freeze@L{avg_freeze:.0f}")

        if f1 > best_f1:
            best_f1 = f1
            best_config = result

    report["measurements"]["best_config"] = best_config

    # ── H6: Can SDD distinguish accepted/rejected? ──
    if best_config:
        h6_pass = best_config["f1"] > 0.3  # Even moderate F1 is useful
        report["hypothesis_tests"]["H6_sdd_predictive"] = {
            "hypothesis": "Router logits divergence can predict rejection (F1 > 0.3)",
            "best_f1": best_config["f1"],
            "best_precision": best_config["precision"],
            "best_recall": best_config["recall"],
            "best_config": best_config["config"],
            "pass": h6_pass,
            "interpretation": (
                f"SDD 有预测能力 (F1={best_config['f1']:.3f}), 可用于 early termination"
                if h6_pass
                else f"SDD 预测能力不足 (F1={best_config['f1']:.3f}), 需要更好的特征或方法"
            ),
        }

        # H7: Freeze before L/2 (layer 24)?
        h7_pass = best_config["avg_freeze_layer"] < 24
        report["hypothesis_tests"]["H7_early_freeze"] = {
            "hypothesis": "SDD 在 L/2 (layer 24) 之前冻结",
            "avg_freeze_layer": best_config["avg_freeze_layer"],
            "target_layer": 24,
            "pass": h7_pass,
            "interpretation": (
                f"平均在 layer {best_config['avg_freeze_layer']:.0f} 冻结, "
                f"节省 {(48 - best_config['avg_freeze_layer']) / 48 * 100:.0f}% 后续层计算"
                if h7_pass
                else f"冻结太晚 (layer {best_config['avg_freeze_layer']:.0f}), 计算节省不足"
            ),
        }

        # H8: Precision > 80%?
        h8_pass = best_config["precision"] > 0.80
        report["hypothesis_tests"]["H8_high_precision"] = {
            "hypothesis": "SDD precision > 80% (误杀 accepted token 比例 < 20%)",
            "precision": best_config["precision"],
            "false_positive_rate": round(
                best_config["fp"] / max(1, best_config["fp"] + best_config["tn"]), 4
            ),
            "pass": h8_pass,
            "interpretation": (
                f"SDD 精度高 ({best_config['precision']:.1%}), 误杀率可接受"
                if h8_pass
                else f"SDD 精度不足 ({best_config['precision']:.1%}), 需要更保守的阈值"
            ),
        }

    # ── MAF reduction estimate ──
    if best_config and best_config["freeze_rate"] > 0:
        from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig
        dummy = SpeculationDivergenceDetector(num_layers=48)
        dummy._total_freezes = int(best_config["freeze_rate"] * best_config["total_evaluated"])
        dummy._freeze_layer_sum = int(best_config["avg_freeze_layer"] * dummy._total_freezes)
        dummy._token_states = {i: None for i in range(best_config["total_evaluated"])}

        # Use MAF=2.82 (theoretical K=3) as baseline, will be replaced with real MAF from exp1
        maf_estimate = dummy.estimate_maf_reduction(original_maf=2.82, K=3)
        report["measurements"]["maf_reduction_estimate"] = maf_estimate
        logger.info(f"  Estimated MAF reduction: {maf_estimate['original_maf']} → "
                     f"{maf_estimate['reduced_maf']} ({maf_estimate['reduction_pct']}%)")

    # ── Save report ──
    report_path = output_dir / "exp3_sdd_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Exp3: SDD 可行性验证结果")
    print("=" * 60)
    for name, test in report["hypothesis_tests"].items():
        status = "✅ PASS" if test["pass"] else "❌ FAIL"
        print(f"  {status} | {name}: {test['interpretation']}")
    print("=" * 60)

    # ── Note about limitations ──
    print("\n⚠️  注意: 此实验使用 simulated acceptance labels (基于 routing overlap)")
    print("   真实 acceptance labels 需要在 vLLM SD 模式下采集 (exp3b)")
    print("   当前结果仅作为可行性的初步判断")

    return report


def main():
    parser = argparse.ArgumentParser(description="Exp3: SDD Feasibility")
    parser.add_argument("--trace", default="results/validation/expert_trace.jsonl")
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--K", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    events = load_trace(args.trace)
    windows = simulate_acceptance_labels(events, K=args.K)
    evaluate_sdd_on_windows(windows, Path(args.output_dir))


if __name__ == "__main__":
    main()
