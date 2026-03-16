import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

FIELDS = [
    "experiment_id",
    "date",
    "owner",
    "issue_id",
    "branch",
    "commit_hash",
    "tag",
    "status",
    "optimization_target",
    "optimization_module",
    "change_summary",
    "hypothesis",
    "model",
    "spec_method",
    "num_speculative_tokens",
    "policy_name",
    "workload_profile",
    "hardware_profile",
    "gpu_type",
    "gpu_count",
    "memory_budget",
    "cuda_version",
    "vllm_version",
    "speculators_version",
    "seed",
    "baseline_experiment_id",
    "baseline_type",
    "comparison_scope",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "itl_mean_ms",
    "throughput_tok_per_s",
    "goodput",
    "acceptance_rate_mean",
    "accepted_tokens",
    "proposed_tokens",
    "rejected_tokens",
    "expert_cache_hit_rate",
    "wasted_prefetched_bytes",
    "gpu_mem_peak_mb",
    "kv_cache_peak_mb",
    "speculative_budget_mb",
    "expert_budget_mb",
    "oom_count",
    "fallback_count",
    "run_success_rate",
    "result_label",
    "primary_gain",
    "primary_cost",
    "final_conclusion",
    "delta_ttft_p95_pct",
    "delta_tpot_p95_pct",
    "delta_throughput_pct",
    "delta_goodput_pct",
    "delta_gpu_mem_peak_pct",
    "score_main",
    "is_merge_candidate",
    "result_path",
    "last_updated",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_num(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _compute_score(delta: Dict[str, Any], w_ttft: float, w_tpot: float, w_tput: float, w_goodput: float, w_mem: float) -> float:
    ttft = _to_num(delta.get("delta_ttft_p95_pct"))
    tpot = _to_num(delta.get("delta_tpot_p95_pct"))
    tput = _to_num(delta.get("delta_throughput_pct"))
    goodput = _to_num(delta.get("delta_goodput_pct"))
    mem = _to_num(delta.get("delta_gpu_mem_peak_pct"))
    return (
        w_ttft * (-ttft)
        + w_tpot * (-tpot)
        + w_tput * tput
        + w_goodput * goodput
        - w_mem * mem
    )


def _merge_candidate(label: str, summary: Dict[str, Any], compare: Dict[str, Any]) -> bool:
    if label == "win":
        return True
    if label != "partial_win":
        return False
    stability = summary.get("stability", {})
    oom = _to_num(stability.get("oom_count"))
    fallback = _to_num(stability.get("fallback_count"))
    ttft_cost = _to_num(compare.get("delta_ttft_p95_pct"))
    mem_cost = _to_num(compare.get("delta_gpu_mem_peak_pct"))
    return oom == 0 and fallback == 0 and ttft_cost <= 2.0 and mem_cost <= 8.0


def _write_header_if_needed(registry_csv: Path) -> None:
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    if not registry_csv.exists() or registry_csv.stat().st_size == 0:
        with registry_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()


def cmd_init(args: argparse.Namespace) -> None:
    registry_csv = Path(args.registry_csv)
    _write_header_if_needed(registry_csv)
    print(f"initialized registry: {registry_csv}")


def _template_meta(exp_id: str, owner: str, issue_id: str) -> Dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "date": datetime.now().date().isoformat(),
        "owner": owner,
        "issue_id": issue_id,
        "branch": "",
        "commit_hash": "",
        "tag": "",
        "status": "planned",
        "optimization_target": "",
        "optimization_module": "",
        "change_summary": "",
        "hypothesis": "",
        "model": "",
        "spec_method": "",
        "num_speculative_tokens": 0,
        "policy_name": "",
        "workload_profile": "",
        "hardware_profile": "",
        "gpu_type": "",
        "gpu_count": 1,
        "memory_budget": "",
        "cuda_version": "",
        "vllm_version": "",
        "speculators_version": "",
        "seed": 42,
    }


def _template_summary(exp_id: str) -> Dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "metrics": {
            "ttft_p50_ms": 0,
            "ttft_p95_ms": 0,
            "tpot_p50_ms": 0,
            "tpot_p95_ms": 0,
            "itl_mean_ms": 0,
            "throughput_tok_per_s": 0,
            "goodput": 0,
            "acceptance_rate_mean": 0,
            "accepted_tokens": 0,
            "proposed_tokens": 0,
            "rejected_tokens": 0,
            "expert_cache_hit_rate": 0,
            "wasted_prefetched_bytes": 0,
            "gpu_mem_peak_mb": 0,
            "kv_cache_peak_mb": 0,
            "speculative_budget_mb": 0,
            "expert_budget_mb": 0,
        },
        "stability": {
            "oom_count": 0,
            "fallback_count": 0,
            "run_success_rate": 1.0,
        },
        "result_label": "neutral",
        "primary_gain": "",
        "primary_cost": "",
        "final_conclusion": "",
    }


def _template_compare() -> Dict[str, Any]:
    return {
        "baseline_experiment_id": "",
        "baseline_type": "",
        "comparison_scope": "same workload + same hardware + same seed family",
        "delta_ttft_p95_pct": 0,
        "delta_tpot_p95_pct": 0,
        "delta_throughput_pct": 0,
        "delta_goodput_pct": 0,
        "delta_gpu_mem_peak_pct": 0,
        "result_label": "neutral",
    }


def cmd_scaffold(args: argparse.Namespace) -> None:
    exp_id = args.experiment_id
    exp_dir = Path(args.results_root) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)

    files = {
        "meta.json": _template_meta(exp_id, args.owner, args.issue_id),
        "bench_raw.json": {},
        "summary.json": _template_summary(exp_id),
        "trace_summary.json": {"experiment_id": exp_id, "acceptance": {}, "prefetch": {}, "memory": {}, "moe": {}},
        "compare_to_baseline.json": _template_compare(),
    }

    for name, data in files.items():
        target = exp_dir / name
        if not target.exists():
            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True, indent=2)

    notes = exp_dir / "notes.md"
    if not notes.exists():
        notes.write_text(
            "# Experiment Notes\n\n"
            "1. This change:\n"
            "2. Result:\n"
            "3. Next step:\n",
            encoding="utf-8",
        )

    print(f"scaffolded experiment directory: {exp_dir}")


def cmd_append(args: argparse.Namespace) -> None:
    exp_dir = Path(args.exp_dir)
    meta = _read_json(exp_dir / "meta.json")
    summary = _read_json(exp_dir / "summary.json")
    compare = _read_json(exp_dir / "compare_to_baseline.json")

    metrics = summary.get("metrics", {})
    stability = summary.get("stability", {})

    label = compare.get("result_label") or summary.get("result_label") or "neutral"
    score_main = _compute_score(
        compare,
        args.weight_ttft,
        args.weight_tpot,
        args.weight_throughput,
        args.weight_goodput,
        args.weight_mem,
    )
    is_candidate = _merge_candidate(label, summary, compare)

    row = {
        "experiment_id": meta.get("experiment_id", exp_dir.name),
        "date": meta.get("date", datetime.now().date().isoformat()),
        "owner": meta.get("owner", ""),
        "issue_id": meta.get("issue_id", ""),
        "branch": meta.get("branch", ""),
        "commit_hash": meta.get("commit_hash", ""),
        "tag": meta.get("tag", ""),
        "status": meta.get("status", "done"),
        "optimization_target": meta.get("optimization_target", ""),
        "optimization_module": meta.get("optimization_module", ""),
        "change_summary": meta.get("change_summary", ""),
        "hypothesis": meta.get("hypothesis", ""),
        "model": meta.get("model", ""),
        "spec_method": meta.get("spec_method", ""),
        "num_speculative_tokens": meta.get("num_speculative_tokens", ""),
        "policy_name": meta.get("policy_name", ""),
        "workload_profile": meta.get("workload_profile", ""),
        "hardware_profile": meta.get("hardware_profile", ""),
        "gpu_type": meta.get("gpu_type", ""),
        "gpu_count": meta.get("gpu_count", ""),
        "memory_budget": meta.get("memory_budget", ""),
        "cuda_version": meta.get("cuda_version", ""),
        "vllm_version": meta.get("vllm_version", ""),
        "speculators_version": meta.get("speculators_version", ""),
        "seed": meta.get("seed", ""),
        "baseline_experiment_id": compare.get("baseline_experiment_id", ""),
        "baseline_type": compare.get("baseline_type", ""),
        "comparison_scope": compare.get("comparison_scope", ""),
        "ttft_p50_ms": metrics.get("ttft_p50_ms", ""),
        "ttft_p95_ms": metrics.get("ttft_p95_ms", ""),
        "tpot_p50_ms": metrics.get("tpot_p50_ms", ""),
        "tpot_p95_ms": metrics.get("tpot_p95_ms", ""),
        "itl_mean_ms": metrics.get("itl_mean_ms", ""),
        "throughput_tok_per_s": metrics.get("throughput_tok_per_s", ""),
        "goodput": metrics.get("goodput", ""),
        "acceptance_rate_mean": metrics.get("acceptance_rate_mean", ""),
        "accepted_tokens": metrics.get("accepted_tokens", ""),
        "proposed_tokens": metrics.get("proposed_tokens", ""),
        "rejected_tokens": metrics.get("rejected_tokens", ""),
        "expert_cache_hit_rate": metrics.get("expert_cache_hit_rate", ""),
        "wasted_prefetched_bytes": metrics.get("wasted_prefetched_bytes", ""),
        "gpu_mem_peak_mb": metrics.get("gpu_mem_peak_mb", ""),
        "kv_cache_peak_mb": metrics.get("kv_cache_peak_mb", ""),
        "speculative_budget_mb": metrics.get("speculative_budget_mb", ""),
        "expert_budget_mb": metrics.get("expert_budget_mb", ""),
        "oom_count": stability.get("oom_count", ""),
        "fallback_count": stability.get("fallback_count", ""),
        "run_success_rate": stability.get("run_success_rate", ""),
        "result_label": label,
        "primary_gain": summary.get("primary_gain", ""),
        "primary_cost": summary.get("primary_cost", ""),
        "final_conclusion": summary.get("final_conclusion", ""),
        "delta_ttft_p95_pct": compare.get("delta_ttft_p95_pct", ""),
        "delta_tpot_p95_pct": compare.get("delta_tpot_p95_pct", ""),
        "delta_throughput_pct": compare.get("delta_throughput_pct", ""),
        "delta_goodput_pct": compare.get("delta_goodput_pct", ""),
        "delta_gpu_mem_peak_pct": compare.get("delta_gpu_mem_peak_pct", ""),
        "score_main": round(score_main, 4),
        "is_merge_candidate": str(is_candidate).lower(),
        "result_path": str(exp_dir),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }

    registry_csv = Path(args.registry_csv)
    _write_header_if_needed(registry_csv)

    existing_rows = []
    with registry_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("experiment_id") != row["experiment_id"]:
                existing_rows.append(r)

    existing_rows.append(row)

    with registry_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"upserted experiment row: {row['experiment_id']}")
    print(f"registry path: {registry_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment registry utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize experiment registry CSV")
    p_init.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    p_init.set_defaults(func=cmd_init)

    p_scaffold = sub.add_parser("scaffold", help="Create standard experiment directory template")
    p_scaffold.add_argument("--experiment-id", required=True)
    p_scaffold.add_argument("--results-root", default="results/experiments")
    p_scaffold.add_argument("--owner", default="")
    p_scaffold.add_argument("--issue-id", default="")
    p_scaffold.set_defaults(func=cmd_scaffold)

    p_append = sub.add_parser("append", help="Append or update one experiment in registry")
    p_append.add_argument("--exp-dir", required=True)
    p_append.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    p_append.add_argument("--weight-ttft", type=float, default=0.25)
    p_append.add_argument("--weight-tpot", type=float, default=0.35)
    p_append.add_argument("--weight-throughput", type=float, default=0.2)
    p_append.add_argument("--weight-goodput", type=float, default=0.15)
    p_append.add_argument("--weight-mem", type=float, default=0.05)
    p_append.set_defaults(func=cmd_append)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
