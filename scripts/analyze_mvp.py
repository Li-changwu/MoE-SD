#!/usr/bin/env python3
"""
MVP Pilot Experiment Analyzer
Parses results from run_mvp_pilot.sh and generates:
1. Summary table (CSV + terminal)
2. Main figure: Latency vs K grouped by pressure level
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_results(results_dir: Path) -> list[dict]:
    """Load all experiment result JSONs from results_dir."""
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name == "experiment_config.json":
            continue
        with open(jf, "r") as f:
            data = json.load(f)

        meta = data.get("_mvp_meta", {})
        label = meta.get("label", jf.stem)
        pressure = meta.get("pressure", "unknown")
        k = meta.get("K", -1)
        gpu_mem = meta.get("gpu_memory_utilization", -1)

        # Extract latency metrics from vllm bench latency output
        avg_latency = data.get("avg_latency")           # seconds
        percentiles = data.get("percentiles", {})        # dict of percentile -> seconds
        avg_per_token_latency = data.get("avg_per_token_latency")
        avg_per_output_token_latency = data.get("avg_per_output_token_latency")

        # Compute derived metrics
        output_len = meta.get("output_len", 512)
        input_len = meta.get("input_len", 128)

        # Throughput: tokens / second
        if avg_latency and avg_latency > 0:
            throughput_tok_s = output_len / avg_latency
        else:
            throughput_tok_s = None

        rows.append({
            "label": label,
            "pressure": pressure,
            "K": k,
            "gpu_memory_utilization": gpu_mem,
            "avg_latency_s": avg_latency,
            "p50_latency_s": percentiles.get("50") or percentiles.get("p50"),
            "p90_latency_s": percentiles.get("90") or percentiles.get("p90"),
            "p95_latency_s": percentiles.get("95") or percentiles.get("p95"),
            "p99_latency_s": percentiles.get("99") or percentiles.get("p99"),
            "avg_per_token_latency_ms": (avg_per_token_latency * 1000) if avg_per_token_latency else None,
            "avg_per_output_token_latency_ms": (avg_per_output_token_latency * 1000) if avg_per_output_token_latency else None,
            "throughput_tok_s": throughput_tok_s,
            "elapsed_seconds": meta.get("elapsed_seconds"),
        })

    return rows


def print_summary_table(rows: list[dict]):
    """Print a formatted summary table to terminal."""
    print("\n" + "=" * 90)
    print("  MVP Pilot Experiment — Results Summary")
    print("=" * 90)

    header = f"{'Label':<12} {'Pressure':<8} {'K':<4} {'Avg Latency(s)':<16} {'P95 Latency(s)':<16} {'TPOT(ms)':<12} {'Tok/s':<10}"
    print(header)
    print("-" * 90)

    for r in rows:
        avg = f"{r['avg_latency_s']:.3f}" if r['avg_latency_s'] else "N/A"
        p95 = f"{r['p95_latency_s']:.3f}" if r['p95_latency_s'] else "N/A"
        tpot = f"{r['avg_per_output_token_latency_ms']:.2f}" if r['avg_per_output_token_latency_ms'] else "N/A"
        toks = f"{r['throughput_tok_s']:.1f}" if r['throughput_tok_s'] else "N/A"
        print(f"{r['label']:<12} {r['pressure']:<8} {r['K']:<4} {avg:<16} {p95:<16} {tpot:<12} {toks:<10}")

    print("=" * 90)

    # Highlight key comparisons
    print("\n📊 Key Comparisons:")
    low_results = {r["K"]: r for r in rows if r["pressure"] == "low"}
    high_results = {r["K"]: r for r in rows if r["pressure"] == "high"}

    for k in [0, 4, 8]:
        if k in low_results and k in high_results:
            low_lat = low_results[k].get("avg_latency_s")
            high_lat = high_results[k].get("avg_latency_s")
            if low_lat and high_lat:
                ratio = high_lat / low_lat
                print(f"  K={k}: Low={low_lat:.3f}s → High={high_lat:.3f}s (×{ratio:.2f})")

    # Check for non-monotonicity
    print("\n🔍 Non-monotonicity Check:")
    for pressure, results in [("low", low_results), ("high", high_results)]:
        latencies = {}
        for k in [0, 4, 8]:
            if k in results and results[k].get("avg_latency_s"):
                latencies[k] = results[k]["avg_latency_s"]

        if len(latencies) >= 2:
            sorted_k = sorted(latencies.keys())
            monotonic = all(latencies[sorted_k[i]] >= latencies[sorted_k[i+1]]
                          for i in range(len(sorted_k)-1))
            non_mono = not monotonic and any(latencies[sorted_k[i]] < latencies[sorted_k[i+1]]
                                             for i in range(len(sorted_k)-1))

            if non_mono:
                print(f"  ⚠️  {pressure.upper()} pressure: NON-MONOTONIC — SD does NOT always help!")
                for k, lat in sorted(latencies.items()):
                    marker = "← worst" if lat == max(latencies.values()) else ""
                    print(f"      K={k}: {lat:.3f}s {marker}")
            else:
                print(f"  ✅ {pressure.upper()} pressure: monotonic improvement with K")
                for k, lat in sorted(latencies.items()):
                    print(f"      K={k}: {lat:.3f}s")


def save_csv(rows: list[dict], out_path: Path):
    """Save results as CSV."""
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n📄 CSV saved: {out_path}")


def plot_main_figure(rows: list[dict], out_path: Path):
    """Generate the main comparison figure."""
    if not HAS_MPL:
        print("⚠️  matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("MVP Pilot: Speculative Decoding under Memory Pressure\n"
                 "(Qwen3-30B-A3B + EAGLE-3, decode-heavy workload)",
                 fontsize=13, fontweight="bold")

    colors = {"low": "#2196F3", "high": "#F44336"}
    markers = {"low": "o", "high": "s"}

    # ── Left panel: Average Latency ──
    ax = axes[0]
    for pressure in ["low", "high"]:
        subset = [r for r in rows if r["pressure"] == pressure]
        subset.sort(key=lambda x: x["K"])
        ks = [r["K"] for r in subset]
        lats = [r["avg_latency_s"] for r in subset]
        if any(v is None for v in lats):
            continue
        ax.plot(ks, lats, color=colors[pressure], marker=markers[pressure],
                linewidth=2, markersize=8, label=f"{pressure} pressure (gpu_mem={subset[0]['gpu_memory_utilization']})")

    ax.set_xlabel("Speculative Depth K", fontsize=11)
    ax.set_ylabel("Average Latency (seconds)", fontsize=11)
    ax.set_title("Total Generation Latency")
    ax.legend(fontsize=9)
    ax.set_xticks([0, 4, 8])
    ax.grid(True, alpha=0.3)

    # ── Right panel: TPOT ──
    ax = axes[1]
    for pressure in ["low", "high"]:
        subset = [r for r in rows if r["pressure"] == pressure]
        subset.sort(key=lambda x: x["K"])
        ks = [r["K"] for r in subset]
        tpots = [r["avg_per_output_token_latency_ms"] for r in subset]
        if any(v is None for v in tpots):
            continue
        ax.plot(ks, tpots, color=colors[pressure], marker=markers[pressure],
                linewidth=2, markersize=8, label=f"{pressure} pressure (gpu_mem={subset[0]['gpu_memory_utilization']})")

    ax.set_xlabel("Speculative Depth K", fontsize=11)
    ax.set_ylabel("Avg Per-Output-Token Latency (ms)", fontsize=11)
    ax.set_title("Per-Output-Token Latency (TPOT)")
    ax.legend(fontsize=9)
    ax.set_xticks([0, 4, 8])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📈 Figure saved: {out_path}")


def plot_speedup_figure(rows: list[dict], out_path: Path):
    """Generate speedup-over-baseline figure."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Speedup over No-SD Baseline (K=0)\nby Memory Pressure Level",
                 fontsize=12, fontweight="bold")

    colors = {"low": "#2196F3", "high": "#F44336"}
    bar_width = 0.35

    for i, pressure in enumerate(["low", "high"]):
        subset = {r["K"]: r for r in rows if r["pressure"] == pressure}
        baseline = subset.get(0, {}).get("avg_latency_s")
        if not baseline:
            continue

        ks = [4, 8]
        speedups = []
        for k in ks:
            lat = subset.get(k, {}).get("avg_latency_s")
            if lat:
                speedups.append(baseline / lat)
            else:
                speedups.append(0)

        x_pos = [x + i * bar_width for x in range(len(ks))]
        bars = ax.bar(x_pos, speedups, bar_width, color=colors[pressure],
                      label=f"{pressure} pressure", alpha=0.85)

        # Add value labels
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{sp:.2f}×", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="baseline (K=0)")
    ax.set_xlabel("Speculative Depth K")
    ax.set_ylabel("Speedup (×)")
    ax.set_xticks([x + bar_width / 2 for x in range(len([4, 8]))])
    ax.set_xticklabels(["K=4", "K=8"])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Speedup figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MVP pilot results")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    rows = load_results(results_dir)
    if not rows:
        print("ERROR: No result files found")
        sys.exit(1)

    print_summary_table(rows)
    save_csv(rows, results_dir / "summary.csv")
    plot_main_figure(rows, results_dir / "mvp_latency_vs_k.png")
    plot_speedup_figure(rows, results_dir / "mvp_speedup.png")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
