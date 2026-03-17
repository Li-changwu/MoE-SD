#!/usr/bin/env python3
"""Issue #25: Collect vLLM bench results into unified CSV."""
import argparse, csv, json, sys
from pathlib import Path

PERCENTILE_KEYS = ["10", "25", "50", "75", "90", "95", "99"]

def extract_row(json_path):
    with open(json_path) as f:
        data = json.load(f)
    meta = data.get("_mvp_meta") or data.get("_meta") or {}
    row = {
        "file": json_path.name,
        "label": meta.get("label", json_path.stem),
        "pressure": meta.get("pressure", "unknown"),
        "K": meta.get("K", -1),
        "workload": meta.get("workload", "decode_heavy"),
        "gpu_memory_utilization": meta.get("gpu_memory_utilization", -1),
        "input_len": meta.get("input_len", -1),
        "output_len": meta.get("output_len", -1),
        "avg_latency": data.get("avg_latency"),
        "avg_per_token_latency": data.get("avg_per_token_latency"),
        "avg_per_output_token_latency": data.get("avg_per_output_token_latency"),
    }
    tpot = data.get("avg_per_output_token_latency")
    row["tpot_ms"] = tpot * 1000 if tpot else None
    avg_lat = data.get("avg_latency")
    out_len = meta.get("output_len", 128)
    row["throughput_tok_s"] = out_len / avg_lat if avg_lat and avg_lat > 0 else None
    pcts = data.get("percentiles", {})
    for p in PERCENTILE_KEYS:
        row[f"p{p}_latency_s"] = pcts.get(p)
    row["speedup_vs_k0"] = None
    return row

def compute_speedups(rows):
    baselines = {}
    for r in rows:
        if r["K"] == 0 and r["avg_latency"]:
            baselines[(r["pressure"], r["workload"])] = r["avg_latency"]
    for r in rows:
        bl = baselines.get((r["pressure"], r["workload"]))
        if bl and r["avg_latency"]:
            r["speedup_vs_k0"] = bl / r["avg_latency"]

def collect(results_dir):
    rows = []
    for jf in sorted(Path(results_dir).glob("*.json")):
        if jf.name == "experiment_config.json": continue
        row = extract_row(jf)
        if row: rows.append(row)
    compute_speedups(rows)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()
    rd = Path(args.results_dir)
    rows = collect(rd)
    if not rows:
        print("No results found", file=sys.stderr); sys.exit(1)
    print(f"\n{'Label':<20} {'Prs':<6} {'K':<4} {'AvgLat(s)':<12} {'TPOT(ms)':<12} {'Tok/s':<10} {'Speedup':<8}")
    print("=" * 80)
    for r in sorted(rows, key=lambda x: (x["pressure"], x["K"])):
        avg = f"{r['avg_latency']:.3f}" if r['avg_latency'] else "N/A"
        tpot = f"{r['tpot_ms']:.2f}" if r['tpot_ms'] else "N/A"
        toks = f"{r['throughput_tok_s']:.1f}" if r['throughput_tok_s'] else "N/A"
        sp = f"{r['speedup_vs_k0']:.2f}x" if r['speedup_vs_k0'] else "—"
        print(f"{r['label']:<20} {r['pressure']:<6} {r['K']:<4} {avg:<12} {tpot:<12} {toks:<10} {sp:<8}")
    out_csv = rd / "collected.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV: {out_csv} ({len(rows)} results)")

if __name__ == "__main__":
    main()
