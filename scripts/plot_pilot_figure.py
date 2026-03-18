#!/usr/bin/env python3
"""
Generate the Pilot Experiment Figure: 4 subplots showing the causal chain
  P↑ → KV↓ → AccRate↓ → Speedup↓

Layout: 2x2 grid
  [A] Latency vs P (baseline & SD)   [B] Speedup vs P
  [C] KV Cache vs P                  [D] Acceptance Rate vs P
"""

import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_pilot_figure.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    with open(f"{results_dir}/results.json") as f:
        all_results = json.load(f)

    with open(f"{results_dir}/speedup.json") as f:
        speedups = json.load(f)

    if not speedups:
        print("No speedup data found!")
        sys.exit(1)

    Ps = [s["P"] for s in speedups]
    base_tps = [s["base_tps"] for s in speedups]
    sd_tps = [s["sd_tps"] for s in speedups]
    spd = [s["speedup"] for s in speedups]
    base_lat = [s["base_lat"] for s in speedups]
    sd_lat = [s["sd_lat"] for s in speedups]
    acc_rates = [s.get("acc_rate") for s in speedups]
    mean_acc_lens = [s.get("mean_acc_len") for s in speedups]

    # KV cache from individual results
    kv_base = []
    kv_sd = []
    for p in Ps:
        b = next((r for r in all_results if r["gpu_mem"] == p and not r["use_sd"]), {})
        s = next((r for r in all_results if r["gpu_mem"] == p and r["use_sd"]), {})
        kv_base.append(b.get("kv_cache_gib"))
        kv_sd.append(s.get("kv_cache_gib"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Pilot Experiment: MoE + SD Resource Competition under Memory Pressure\n"
        "Model: Qwen3-30B-A3B (MoE, 128 experts) | EAGLE3 K=3 | CPU offload=30GB",
        fontsize=13, fontweight="bold"
    )

    # Color scheme
    c_base = "#2196F3"  # blue
    c_sd = "#FF5722"    # orange-red
    c_speedup = "#4CAF50"  # green
    c_acc = "#9C27B0"   # purple

    # ── [A] Latency vs P ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(Ps, base_lat, "o-", color=c_base, linewidth=2, markersize=8, label="Baseline (no SD)")
    ax.plot(Ps, sd_lat, "s-", color=c_sd, linewidth=2, markersize=8, label="SD (K=3)")
    ax.set_xlabel("gpu_memory_utilization (P)", fontsize=11)
    ax.set_ylabel("Avg Latency (s)", fontsize=11)
    ax.set_title("(A) Latency vs Memory Pressure", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # ── [B] Speedup vs P ──────────────────────────────────────────────
    ax = axes[0, 1]
    colors = [c_speedup if s >= 1.0 else "#F44336" for s in spd]
    ax.bar(range(len(Ps)), spd, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(Ps)))
    ax.set_xticklabels([f"P={p}" for p in Ps], fontsize=9)
    ax.set_ylabel("Speedup (SD / Baseline)", fontsize=11)
    ax.set_title("(B) SD Speedup vs Memory Pressure", fontsize=11, fontweight="bold")
    for i, s in enumerate(spd):
        ax.text(i, s + 0.02, f"{s:.2f}x", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=min(0.4, min(spd) - 0.1), top=max(spd) + 0.15)
    ax.grid(True, axis="y", alpha=0.3)

    # ── [C] KV Cache vs P ─────────────────────────────────────────────
    ax = axes[1, 0]
    x = np.arange(len(Ps))
    w = 0.35
    if any(v is not None for v in kv_base):
        bars1 = ax.bar(x - w/2, [v or 0 for v in kv_base], w, color=c_base, alpha=0.8, label="Baseline")
        bars2 = ax.bar(x + w/2, [v or 0 for v in kv_sd], w, color=c_sd, alpha=0.8, label="SD (K=3)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P={p}" for p in Ps], fontsize=9)
        ax.set_ylabel("Available KV Cache (GiB)", fontsize=11)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "KV cache data not available", transform=ax.transAxes, ha="center")
    ax.set_title("(C) KV Cache Allocation vs Memory Pressure", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # ── [D] Acceptance Rate vs P ──────────────────────────────────────
    ax = axes[1, 1]
    valid_acc = [(p, a) for p, a in zip(Ps, acc_rates) if a is not None]
    valid_mal = [(p, m) for p, m in zip(Ps, mean_acc_lens) if m is not None]
    if valid_acc:
        ps_a, as_a = zip(*valid_acc)
        ax.plot(ps_a, as_a, "D-", color=c_acc, linewidth=2, markersize=8, label="Acceptance Rate (%)")
        ax.set_ylabel("Acceptance Rate (%)", color=c_acc, fontsize=11)
        ax.tick_params(axis="y", labelcolor=c_acc)
        if valid_mal:
            ax2 = ax.twinx()
            ps_m, as_m = zip(*valid_mal)
            ax2.plot(ps_m, as_m, "^--", color="#FF9800", linewidth=2, markersize=7, label="Mean Acc Length")
            ax2.set_ylabel("Mean Acceptance Length", color="#FF9800", fontsize=11)
            ax2.tick_params(axis="y", labelcolor="#FF9800")
    else:
        ax.text(0.5, 0.5, "Acceptance data not available", transform=ax.transAxes, ha="center")
    ax.set_xlabel("gpu_memory_utilization (P)", fontsize=11)
    ax.set_title("(D) Draft Acceptance vs Memory Pressure", fontsize=11, fontweight="bold")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = f"{results_dir}/pilot_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

    # Also save a text summary
    summary = f"""Pilot Experiment Summary
========================
Model: Qwen3-30B-A3B-Instruct-2507 (MoE, 128 experts, top-8)
Speculator: EAGLE3 (K=3)
CPU Offload: 30 GiB | GPU: RTX A6000 48GB

{'P':>6} | {'Base(tok/s)':>12} | {'SD(tok/s)':>12} | {'Speedup':>8} | {'AccRate':>8} | {'MeanAccLen':>10} | {'KV_base':>8} | {'KV_sd':>8}
{'-'*90}
"""
    for s in speedups:
        ar = f"{s.get('acc_rate',0):.1f}%" if s.get('acc_rate') else "N/A"
        ml = f"{s.get('mean_acc_len',0):.2f}" if s.get('mean_acc_len') else "N/A"
        kb = f"{s.get('kv_gib_base',0):.2f}" if s.get('kv_gib_base') else "N/A"
        ks = f"{s.get('kv_gib_sd',0):.2f}" if s.get('kv_gib_sd') else "N/A"
        summary += f"{s['P']:>6.2f} | {s['base_tps']:>12.2f} | {s['sd_tps']:>12.2f} | {s['speedup']:>7.3f}x | {ar:>8} | {ml:>10} | {kb:>8} | {ks:>8}\n"

    summary += f"""
Key Finding:
  Speedup trend: {' → '.join(f'{s["speedup"]:.2f}x' for s in speedups)}
  {'CONFIRMED: Speedup degrades with increasing pressure!' if len(speedups) >= 2 and speedups[0]["speedup"] > speedups[-1]["speedup"] else 'Trend inconclusive - need more data points.'}
"""
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
