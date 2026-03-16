import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Set

START_MARK = "<!-- AUTO_DASHBOARD_START -->"
END_MARK = "<!-- AUTO_DASHBOARD_END -->"


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


def has_signal(row: Dict[str, str]) -> bool:
    keys = ["ttft_p95_ms", "tpot_p95_ms", "throughput_tok_per_s", "goodput"]
    return any(to_float(row.get(k, "")) > 0 for k in keys)


def valid_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        status = (row.get("status") or "").lower()
        label = (row.get("result_label") or "").lower()
        if status in {"done", "running"} and label != "invalid" and has_signal(row):
            out.append(row)
    return out


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def extract_snapshot_block(readme_text: str) -> str:
    if START_MARK not in readme_text or END_MARK not in readme_text:
        return ""
    start = readme_text.index(START_MARK)
    end = readme_text.index(END_MARK) + len(END_MARK)
    return readme_text[start:end]


def check_registry(rows: List[Dict[str, str]], min_valid_rows: int, required_methods: Set[str]) -> List[str]:
    errors: List[str] = []
    valid = valid_rows(rows)

    if len(valid) < min_valid_rows:
        errors.append(f"valid registry rows too few: got={len(valid)} expected>={min_valid_rows}")

    methods = {(r.get("spec_method") or "").strip().lower() or "no_sd" for r in valid}
    missing = sorted(m for m in required_methods if m not in methods)
    if missing:
        errors.append(f"required methods missing in registry: {','.join(missing)}")

    for row in valid:
        exp_id = row.get("experiment_id", "")
        model = (row.get("model") or "").strip()
        method = (row.get("spec_method") or "").strip()
        module = (row.get("optimization_module") or "").strip()
        if not model:
            errors.append(f"row {exp_id} missing model")
        if not method:
            errors.append(f"row {exp_id} missing spec_method")
        if not module:
            errors.append(f"row {exp_id} missing optimization_module")

    return errors


def check_readme_snapshot(snapshot: str) -> List[str]:
    errors: List[str] = []
    if not snapshot:
        errors.append("README missing AUTO_DASHBOARD block")
        return errors

    required_patterns = [
        r"model=",
        r"\| experiment_id \| date \| model \| method \| module \| workload \|",
        r"method=no_sd|\| no_sd \|",
        r"method=eagle3|\| eagle3 \|",
    ]
    for pat in required_patterns:
        if not re.search(pat, snapshot):
            errors.append(f"README snapshot missing pattern: {pat}")

    return errors


def check_dashboard_html(html_text: str) -> List[str]:
    errors: List[str] = []
    if not html_text:
        errors.append("dashboard HTML missing or empty")
        return errors

    required_tokens = [
        "class='table-wrap'",
        "class='ledger-table'",
        "<th>model</th>",
        "<th>spec_method</th>",
        "baseline/no_sd",
        "baseline/eagle3",
    ]
    for token in required_tokens:
        if token not in html_text:
            errors.append(f"dashboard HTML missing token: {token}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dashboard-refresh output consistency")
    parser.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--dashboard-html", default="docs/dashboard/optimization_dashboard.html")
    parser.add_argument("--min-valid-rows", type=int, default=2)
    parser.add_argument("--required-methods", default="no_sd,eagle3")
    args = parser.parse_args()

    required_methods = {x.strip().lower() for x in args.required_methods.split(",") if x.strip()}

    rows = read_csv_rows(Path(args.registry_csv))
    readme_text = read_text(Path(args.readme))
    html_text = read_text(Path(args.dashboard_html))

    errors: List[str] = []
    errors.extend(check_registry(rows, args.min_valid_rows, required_methods))
    errors.extend(check_readme_snapshot(extract_snapshot_block(readme_text)))
    errors.extend(check_dashboard_html(html_text))

    if errors:
        print("dashboard-refresh validation FAILED")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(2)

    print("dashboard-refresh validation PASSED")
    print(f"- valid_rows >= {args.min_valid_rows}")
    print(f"- required methods present: {','.join(sorted(required_methods))}")
    print("- README snapshot + dashboard HTML shape OK")


if __name__ == "__main__":
    main()
