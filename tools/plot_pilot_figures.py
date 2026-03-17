#!/usr/bin/env python3
"""Issue #26: Visualization — 2x2 faceted main figure + speedup chart."""
import argparse, csv, sys
from pathlib import Path

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

COLORS = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
MARKERS = {"low": "o", "medium": "D", "high": "s"}
LABELS = {"low": "Low (0.90)", "medium": "Medium (0.75)", "high": "High (0.60)"}

def load_csv(p):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k in ["K","input_len","output_len"]:
                try: r[k] = int(float(r.get(k,-1)))
                except: r[k] = -1
            for k in ["avg_latency","tpot_ms","throughput_tok_s","speedup_vs_k0","gpu_memory_utilization"]:
                try: r[k] = float(r[k]) if r.get(k) not in ("","None",None) else None
                except: r[k] = None
            rows.append(r)
    return rows

def plot_main(rows, out):
    if not HAS_MPL: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Speculative Decoding under Memory Pressure\nQwen3-30B-A3B MoE + EAGLE-3", fontsize=13, fontweight="bold")
    for i, (key, ylabel) in enumerate([("avg_latency","Latency (s)"),("tpot_ms","TPOT (ms)")]):
        ax = axes[i]
        for prs in ["low","medium","high"]:
            sub = sorted([r for r in rows if r.get("pressure")==prs and r.get(key) is not None], key=lambda x:x["K"])
            if not sub: continue
            ax.plot([r["K"] for r in sub], [r[key] for r in sub], color=COLORS.get(prs,"gray"), marker=MARKERS.get(prs,"o"), linewidth=2, markersize=8, label=LABELS.get(prs,prs))
        ax.set_xlabel("Speculative Depth K"); ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(set(r["K"] for r in rows))); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📈 Main figure: {out}")

def plot_speedup(rows, out):
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Speedup over K=0 Baseline", fontsize=12, fontweight="bold")
    pressures = sorted(set(r["pressure"] for r in rows))
    ks = sorted(set(r["K"] for r in rows if r["K"]>0))
    w = 0.8 / max(len(pressures),1)
    for i, prs in enumerate(pressures):
        vals = []
        for k in ks:
            m = [r for r in rows if r["pressure"]==prs and r["K"]==k and r.get("speedup_vs_k0")]
            vals.append(m[0]["speedup_vs_k0"] if m else 0)
        x = np.arange(len(ks))
        bars = ax.bar(x+i*w, vals, w, color=COLORS.get(prs,"gray"), label=prs, alpha=0.85)
        for b,v in zip(bars, vals):
            if v: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}x", ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(np.arange(len(ks))+w/2); ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Speedup (x)"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"📈 Speedup: {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    cp = Path(args.csv)
    od = Path(args.output_dir) if args.output_dir else cp.parent
    od.mkdir(parents=True, exist_ok=True)
    rows = load_csv(cp)
    print(f"Loaded {len(rows)} results")
    plot_main(rows, od / "fig_main.png")
    plot_speedup(rows, od / "fig_speedup.png")
    print(f"\n✅ Figures saved to {od}")

if __name__ == "__main__":
    main()
