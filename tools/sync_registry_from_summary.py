import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tools.experiment_registry import FIELDS


def to_float(value: str) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_registry_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()


def normalize_method(method: str) -> str:
    m = (method or "").strip().lower()
    if not m:
        return "no_sd"
    if m in {"none", "disabled", "baseline"}:
        return "no_sd"
    return m


def make_auto_experiment_id(file_path: str) -> str:
    payload = (file_path or "unknown").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(payload).hexdigest()[:10].upper()
    return f"AUTO-{digest}"


def derive_workload(row: Dict[str, str]) -> str:
    existing = (row.get("workload_profile") or "").strip()
    if existing:
        return existing
    prompt_len = row.get("prompt_len") or "?"
    output_len = row.get("output_len") or "?"
    request_rate = row.get("request_rate") or "?"
    return f"len{prompt_len}x{output_len}-qps{request_rate}"


def has_metric_signal(row: Dict[str, str]) -> bool:
    return any(to_float(row.get(k, "")) > 0 for k in ["p95_ttft_ms", "p95_tpot_ms", "throughput", "goodput"])


def row_template() -> Dict[str, str]:
    return {k: "" for k in FIELDS}


def update_row_from_summary(base: Dict[str, str], summary: Dict[str, str], owner: str, issue_id: str) -> Dict[str, str]:
    now = datetime.now().isoformat(timespec="seconds")
    method = normalize_method(summary.get("method", ""))
    is_no_sd = method in {"no_sd", "vanilla", "base"}

    base["experiment_id"] = base.get("experiment_id") or make_auto_experiment_id(summary.get("file", ""))
    base["date"] = base.get("date") or datetime.now().date().isoformat()
    base["owner"] = base.get("owner") or owner
    base["issue_id"] = base.get("issue_id") or issue_id
    base["status"] = "done"
    base["optimization_target"] = "latency_throughput"
    base["optimization_module"] = "baseline"
    base["model"] = summary.get("model", "")
    base["spec_method"] = "" if is_no_sd else method
    base["num_speculative_tokens"] = base.get("num_speculative_tokens") or "0"
    base["policy_name"] = base.get("policy_name") or "baseline"
    base["workload_profile"] = derive_workload(summary)
    base["seed"] = summary.get("seed", "")
    base["commit_hash"] = summary.get("git_commit", "")

    base["ttft_p50_ms"] = summary.get("p50_ttft_ms", "")
    base["ttft_p95_ms"] = summary.get("p95_ttft_ms", "")
    base["tpot_p50_ms"] = summary.get("p50_tpot_ms", "")
    base["tpot_p95_ms"] = summary.get("p95_tpot_ms", "")
    base["itl_mean_ms"] = summary.get("mean_itl_ms", "")
    base["throughput_tok_per_s"] = summary.get("throughput", "")
    base["goodput"] = summary.get("goodput", "")

    base["run_success_rate"] = base.get("run_success_rate") or "1.0"
    base["result_label"] = base.get("result_label") or "neutral"
    base["score_main"] = base.get("score_main") or "0.0"
    base["is_merge_candidate"] = base.get("is_merge_candidate") or "false"
    base["result_path"] = summary.get("file", "")
    base["last_updated"] = now
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync registry CSV from parsed summary CSV")
    parser.add_argument("--summary-csv", default="results/parsed/summary.csv")
    parser.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    parser.add_argument("--owner", default="auto")
    parser.add_argument("--issue-id", default="MOESD-AUTO")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    registry_csv = Path(args.registry_csv)

    summaries = read_csv_rows(summary_csv)
    if not summaries:
        print(f"no summary rows found: {summary_csv}")
        return

    ensure_registry_header(registry_csv)
    existing_rows = read_csv_rows(registry_csv)

    by_exp: Dict[str, Dict[str, str]] = {}
    for row in existing_rows:
        exp_id = row.get("experiment_id", "")
        if exp_id:
            by_exp[exp_id] = row

    inserted = 0
    updated = 0
    skipped = 0

    for summary in summaries:
        if not has_metric_signal(summary):
            skipped += 1
            continue

        exp_id = make_auto_experiment_id(summary.get("file", ""))
        base = by_exp.get(exp_id, row_template())
        if not base.get("experiment_id"):
            base["experiment_id"] = exp_id
            inserted += 1
        else:
            updated += 1

        by_exp[exp_id] = update_row_from_summary(base, summary, args.owner, args.issue_id)

    merged_rows = sorted(by_exp.values(), key=lambda r: (r.get("date", ""), r.get("experiment_id", "")))

    with registry_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"synced registry: {registry_csv}")
    print(f"inserted={inserted}, updated={updated}, skipped={skipped}, total={len(merged_rows)}")


if __name__ == "__main__":
    main()
