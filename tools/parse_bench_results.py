import argparse
import csv
import json
from pathlib import Path


def collect_json_files(input_dir: Path):
    return list(input_dir.rglob("*.json"))


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_file(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row = {
        "file": str(fp),
        "model": data.get("model"),
        "num_prompts": data.get("num_prompts"),
        "request_rate": data.get("request_rate"),
        "mean_ttft_ms": safe_get(data, "ttft", "mean"),
        "p50_ttft_ms": safe_get(data, "ttft", "p50"),
        "p95_ttft_ms": safe_get(data, "ttft", "p95"),
        "mean_tpot_ms": safe_get(data, "tpot", "mean"),
        "p50_tpot_ms": safe_get(data, "tpot", "p50"),
        "p95_tpot_ms": safe_get(data, "tpot", "p95"),
        "mean_itl_ms": safe_get(data, "itl", "mean"),
        "throughput": data.get("throughput"),
        "goodput": data.get("goodput"),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_json_files(input_dir)
    rows = [parse_file(fp) for fp in files]

    out_csv = output_dir / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
