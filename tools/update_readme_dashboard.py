import argparse
import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

START_MARK = "<!-- AUTO_DASHBOARD_START -->"
END_MARK = "<!-- AUTO_DASHBOARD_END -->"


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(registry_csv: Path) -> List[Dict[str, str]]:
    if not registry_csv.exists():
        return []
    with registry_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def valid_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        status = (r.get("status") or "").lower()
        label = (r.get("result_label") or "").lower()
        has_signal = any((to_float(r.get(k)) or 0) > 0 for k in ["ttft_p95_ms", "tpot_p95_ms", "throughput_tok_per_s", "goodput"])
        if status in {"done", "running"} and label != "invalid" and has_signal:
            out.append(r)
    return out


def best_row(rows: List[Dict[str, str]], metric: str, minimize: bool) -> Optional[Dict[str, str]]:
    candidates = [r for r in rows if to_float(r.get(metric)) is not None]
    if not candidates:
        return None
    if minimize:
        return min(candidates, key=lambda r: to_float(r.get(metric)) or 1e18)
    return max(candidates, key=lambda r: to_float(r.get(metric)) or -1e18)


def recommended_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    candidates = [r for r in rows if to_float(r.get("score_main")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: to_float(r.get("score_main")) or -1e18)


def top_issues(rows: List[Dict[str, str]]) -> List[str]:
    issues: List[str] = []

    high_ttft = [r for r in rows if (to_float(r.get("ttft_p95_ms")) or 0) > 2000]
    if high_ttft:
        issues.append("Long prompt workload 下 TTFT p95 偏高")

    prefetch_unstable = [
        r
        for r in rows
        if (to_float(r.get("acceptance_rate_mean")) or 1) < 0.5
        and (to_float(r.get("wasted_prefetched_bytes")) or 0) > 0
    ]
    if prefetch_unstable:
        issues.append("低 acceptance 下 prefetch 收益不稳定")

    unstable = [
        r
        for r in rows
        if (to_float(r.get("fallback_count")) or 0) > 0 or (to_float(r.get("oom_count")) or 0) > 0
    ]
    if unstable:
        issues.append("存在 fallback/OOM 稳定性风险")

    if not issues:
        issues.append("暂无明显阻塞，建议扩展 workload 覆盖后再判断")
    return issues[:3]


def fmt_delta(v: Any) -> str:
    num = to_float(v)
    if num is None:
        return "n/a"
    return f"{num:+.2f}%"


def display_method(row: Dict[str, str]) -> str:
    method = (row.get("spec_method") or "").strip()
    return method if method else "no_sd"


def display_model(row: Dict[str, str]) -> str:
    model = (row.get("model") or "").strip()
    if not model:
        return "unknown"
    return model.split("/")[-1]


def card_line(title: str, row: Optional[Dict[str, str]], metric: str, delta_metric: str) -> str:
    if not row:
        return f"- **{title}**: n/a"
    return (
        f"- **{title}**: {row.get('experiment_id', '')} | "
        f"model={display_model(row)} | "
        f"{metric}={row.get(metric, 'n/a')} | "
        f"delta={fmt_delta(row.get(delta_metric))} | "
        f"method={display_method(row)} | "
        f"policy={row.get('policy_name', '')}"
    )


def build_dashboard_block(rows: List[Dict[str, str]], dashboard_rel_path: str) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    valid = valid_rows(rows)

    best_ttft = best_row(valid, "ttft_p95_ms", True)
    best_tpot = best_row(valid, "tpot_p95_ms", True)
    best_tput = best_row(valid, "throughput_tok_per_s", False)
    best_goodput = best_row(valid, "goodput", False)

    dist = Counter((r.get("result_label") or "unknown") for r in valid)
    modules = Counter((r.get("optimization_module") or "unknown") for r in valid)
    rec = recommended_row(valid)
    issues = top_issues(valid)

    latest = sorted(valid, key=lambda r: (r.get("date") or "", r.get("experiment_id") or ""), reverse=True)[:8]

    lines: List[str] = []
    lines.append("## Optimization Dashboard Snapshot")
    lines.append("")
    lines.append(f"Updated: {now}  ")
    lines.append(f"Dashboard HTML: [{dashboard_rel_path}]({dashboard_rel_path})")
    lines.append("")
    lines.append("### A. 当前最优结果卡片")
    lines.append(card_line("Best TTFT", best_ttft, "ttft_p95_ms", "delta_ttft_p95_pct"))
    lines.append(card_line("Best TPOT", best_tpot, "tpot_p95_ms", "delta_tpot_p95_pct"))
    lines.append(card_line("Best Throughput", best_tput, "throughput_tok_per_s", "delta_throughput_pct"))
    lines.append(card_line("Best Goodput", best_goodput, "goodput", "delta_goodput_pct"))
    lines.append("")
    lines.append("### C. 实验结论分布")
    if dist:
        for k, v in sorted(dist.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- no data")
    lines.append("")
    lines.append("### D. 模块贡献分布")
    if modules:
        for k, v in modules.most_common(8):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- no data")
    lines.append("")
    lines.append("### E. 当前推荐配置")
    if rec:
        lines.append(f"- model: {display_model(rec)}")
        lines.append(f"- config: {display_method(rec)} + {rec.get('policy_name', '')}")
        lines.append(f"- workload_scope: {rec.get('workload_profile', '')}")
        lines.append(f"- risk: {rec.get('primary_cost', '') or 'n/a'}")
        lines.append(f"- score_main: {rec.get('score_main', 'n/a')}")
        lines.append(f"- is_merge_candidate: {rec.get('is_merge_candidate', 'false')}")
    else:
        lines.append("- no recommendation yet")
    lines.append("")
    lines.append("### F. 当前主要问题")
    for i, item in enumerate(issues, start=1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append("### 最近实验台账")
    lines.append("")
    lines.append("| experiment_id | date | model | method | module | workload | ttft_p95_ms | tpot_p95_ms | throughput | goodput | result_label | score_main | merge_candidate |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |")
    if latest:
        for r in latest:
            lines.append(
                "| {exp} | {date} | {model} | {method} | {mod} | {wk} | {ttft} | {tpot} | {tput} | {gp} | {label} | {score} | {merge} |".format(
                    exp=r.get("experiment_id", ""),
                    date=r.get("date", ""),
                    model=display_model(r),
                    method=display_method(r),
                    mod=r.get("optimization_module", ""),
                    wk=r.get("workload_profile", ""),
                    ttft=r.get("ttft_p95_ms", ""),
                    tpot=r.get("tpot_p95_ms", ""),
                    tput=r.get("throughput_tok_per_s", ""),
                    gp=r.get("goodput", ""),
                    label=r.get("result_label", ""),
                    score=r.get("score_main", ""),
                    merge=r.get("is_merge_candidate", ""),
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    return "\n".join(lines)


def update_readme(readme_path: Path, block_text: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    wrapped = f"{START_MARK}\n{block_text}\n{END_MARK}"

    if START_MARK in content and END_MARK in content:
        start_idx = content.index(START_MARK)
        end_idx = content.index(END_MARK) + len(END_MARK)
        new_content = content[:start_idx] + wrapped + content[end_idx:]
    else:
        lines = content.splitlines()
        insert_idx = 2 if len(lines) >= 2 else len(lines)
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, wrapped)
        lines.insert(insert_idx + 2, "")
        new_content = "\n".join(lines) + "\n"

    readme_path.write_text(new_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update README with dashboard snapshot")
    parser.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--dashboard-rel-path", default="docs/dashboard/optimization_dashboard.html")
    args = parser.parse_args()

    rows = load_rows(Path(args.registry_csv))
    block = build_dashboard_block(rows, args.dashboard_rel_path)
    update_readme(Path(args.readme), block)
    print(f"updated README dashboard block in {args.readme}")


if __name__ == "__main__":
    main()
