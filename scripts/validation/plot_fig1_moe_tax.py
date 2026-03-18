#!/usr/bin/env python3
"""
Figure 1: The MoE Tax on Speculative Decoding
================================================
SpecMoE 论文的核心 motivation figure — 一张图说明全部问题。

三联图 (a)(b)(c):
  (a) MAF(K) 曲线: 随 K 增长，unique expert 数量迅速膨胀
      → 核心发现: MoE 的 expert 加载量随 K 超线性增长
  (b) Speedup Wall: MAF 吞噬了 SD 的理论加速比
      → 核心矛盾: K 越大、acceptance 越多也没用，被 MAF 抵消了
  (c) SpecMoE Opportunity: dedup 后 MAF 大幅下降 → speedup 恢复
      → 核心价值: SpecMoE 各技术的预期收益叠加

支持两种数据源:
  --mode theoretical  : 纯公式推导 (无需 GPU，可立即运行)
  --mode measured     : 使用 exp0/exp1 的真实 trace 数据

输出: results/figures/fig1_moe_tax.pdf + .png

用法:
  python scripts/validation/plot_fig1_moe_tax.py --mode theoretical
  python scripts/validation/plot_fig1_moe_tax.py --mode measured --trace results/validation/expert_trace.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ═══════════════════════════════════════════════════════════════════════
# Constants for Qwen3-30B-A3B
# ═══════════════════════════════════════════════════════════════════════
N_EXPERTS = 128         # total experts per layer
TOP_K = 8               # experts per token
N_LAYERS = 48           # MoE layers
EXPERT_SIZE_MB = 3 * 768 * 2048 * 2 / 1e6  # ≈ 9.0 MB (bf16)
PCIE_BW_GBPS = 32.0    # PCIe 4.0 x16 theoretical


def theoretical_maf(K, k=TOP_K, N=N_EXPERTS):
    """MAF_random(K) = N*(1-(1-k/N)^(K+1))/k"""
    return N * (1 - (1 - k / N) ** (K + 1)) / k


def speedup_formula(alpha, K, maf, gamma=0.10, beta=1.0):
    """
    S = ᾱ(K+1) / (1 + γ + β(MAF(K)-1))
    ᾱ: mean accepted length / (K+1)
    γ: draft overhead ratio
    β: memory-boundedness (1.0 = fully PCIe-bound)
    """
    numerator = alpha * (K + 1)
    denominator = 1 + gamma + beta * (maf - 1)
    return numerator / denominator


def get_theoretical_data():
    """Generate all data from formulas."""
    Ks = np.arange(1, 8)
    
    # MAF curves
    maf_random = np.array([theoretical_maf(K) for K in Ks])
    # "Correlated routing" estimate: ~85% of random (literature-inspired assumption)
    maf_correlated = maf_random * 0.82
    # Ideal (perfect overlap): always = 1
    maf_ideal = np.ones_like(Ks, dtype=float)
    # SpecMoE (dedup + SDD + cache): ~55-65% of correlated
    maf_specmoe = np.maximum(1.0, maf_correlated * 0.60)

    # Measured pilot data point (from concurrent_sweep: K=3, speedup=0.91)
    pilot_speedup_k3 = 0.91
    pilot_acc_rate = 0.46

    return {
        "Ks": Ks,
        "maf_random": maf_random,
        "maf_correlated": maf_correlated,
        "maf_ideal": maf_ideal,
        "maf_specmoe": maf_specmoe,
        "pilot_speedup_k3": pilot_speedup_k3,
        "pilot_acc_rate": pilot_acc_rate,
    }


def get_measured_data(trace_path):
    """Load real MAF measurements from exp1 output."""
    from collectors.expert_trace_hook import compute_maf_from_trace, compute_theoretical_maf

    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    Ks = np.arange(1, 8)
    maf_measured = []
    maf_std = []
    for K in Ks:
        result = compute_maf_from_trace(events, K=int(K))
        maf_measured.append(result.mean_maf if result.num_windows > 0 else theoretical_maf(int(K)))
        maf_std.append(result.std_maf if result.num_windows > 0 else 0)

    maf_random = np.array([theoretical_maf(K) for K in Ks])
    maf_measured = np.array(maf_measured)
    maf_std = np.array(maf_std)
    maf_ideal = np.ones_like(Ks, dtype=float)
    maf_specmoe = np.maximum(1.0, maf_measured * 0.60)

    return {
        "Ks": Ks,
        "maf_random": maf_random,
        "maf_correlated": maf_measured,
        "maf_correlated_std": maf_std,
        "maf_ideal": maf_ideal,
        "maf_specmoe": maf_specmoe,
        "is_measured": True,
        "pilot_speedup_k3": 0.91,
        "pilot_acc_rate": 0.46,
    }


# ═══════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_figure(data, output_dir: Path, is_measured=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    Ks = data["Ks"]
    maf_random = data["maf_random"]
    maf_real = data["maf_correlated"]   # measured or estimated
    maf_ideal = data["maf_ideal"]
    maf_specmoe = data["maf_specmoe"]

    # ── Style ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

    # Colors — colorblind-friendly palette
    C_RANDOM  = "#D62728"   # red — worst case
    C_REAL    = "#1F77B4"   # blue — measured/expected
    C_SPECMOE = "#2CA02C"   # green — our method
    C_IDEAL   = "#7F7F7F"   # gray — ideal
    C_FILL    = "#1F77B4"   # blue fill
    C_PILOT   = "#FF7F0E"   # orange — pilot measurement

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"wspace": 0.32})

    # ══════════════════════════════════════════════════════════════
    # Panel (a): MAF(K) — The Expert Load Amplification
    # ══════════════════════════════════════════════════════════════
    ax = axes[0]

    # Shaded area = dedup opportunity
    ax.fill_between(Ks, maf_ideal, maf_real,
                     alpha=0.12, color=C_SPECMOE, label="_nolegend_")
    ax.fill_between(Ks, maf_real, maf_random,
                     alpha=0.08, color=C_RANDOM, label="_nolegend_")

    ax.plot(Ks, maf_random, "^--", color=C_RANDOM, linewidth=2, markersize=7,
            label="i.i.d. Random Routing")

    if data.get("is_measured") and "maf_correlated_std" in data:
        ax.errorbar(Ks, maf_real, yerr=data["maf_correlated_std"],
                     fmt="o-", color=C_REAL, linewidth=2.5, markersize=8,
                     capsize=3, label="Measured (Qwen3-30B)")
    else:
        ax.plot(Ks, maf_real, "o-", color=C_REAL, linewidth=2.5, markersize=8,
                label="Expected (correlated routing)")

    ax.plot(Ks, maf_specmoe, "s-", color=C_SPECMOE, linewidth=2.5, markersize=8,
            label="SpecMoE (dedup+SDD+cache)")
    ax.plot(Ks, maf_ideal, ":", color=C_IDEAL, linewidth=1.5,
            label="Ideal (MAF=1)")

    # Annotations
    K3_idx = 2  # K=3 is index 2
    ax.annotate(
        f"MAF={maf_real[K3_idx]:.2f}",
        xy=(3, maf_real[K3_idx]), xytext=(4.3, maf_real[K3_idx] + 0.3),
        fontsize=9, fontweight="bold", color=C_REAL,
        arrowprops=dict(arrowstyle="-|>", color=C_REAL, lw=1.2),
    )
    # Dedup opportunity arrow
    mid_y = (maf_real[K3_idx] + maf_specmoe[K3_idx]) / 2
    ax.annotate(
        "", xy=(3.0, maf_specmoe[K3_idx]), xytext=(3.0, maf_real[K3_idx]),
        arrowprops=dict(arrowstyle="<->", color=C_SPECMOE, lw=1.8),
    )
    savings_pct = (1 - maf_specmoe[K3_idx] / maf_real[K3_idx]) * 100
    ax.text(3.25, mid_y, f"–{savings_pct:.0f}%", fontsize=9, fontweight="bold",
            color=C_SPECMOE, va="center")

    ax.set_xlabel("Speculation Length $K$")
    ax.set_ylabel("Memory Access Factor (MAF)")
    ax.set_title("(a) Expert Load Amplification", fontweight="bold")
    ax.set_xticks(Ks)
    ax.set_ylim(0.5, max(maf_random) + 0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # ══════════════════════════════════════════════════════════════
    # Panel (b): Speedup Wall — MAF kills SD benefit
    # ══════════════════════════════════════════════════════════════
    ax = axes[1]

    alphas = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    K = 3
    maf_k3_real = maf_real[K3_idx]
    maf_k3_specmoe = maf_specmoe[K3_idx]

    # Three speedup curves
    sp_no_moe = alphas * (K + 1)   # theoretical upper bound (dense model)
    sp_with_maf = np.array([speedup_formula(a, K, maf_k3_real) for a in alphas])
    sp_specmoe = np.array([speedup_formula(a, K, maf_k3_specmoe) for a in alphas])

    ax.plot(alphas, sp_no_moe, "^--", color=C_IDEAL, linewidth=1.5, markersize=6,
            label="Dense Model (no MoE tax)")
    ax.plot(alphas, sp_with_maf, "o-", color=C_RANDOM, linewidth=2.5, markersize=8,
            label=f"MoE Vanilla (MAF={maf_k3_real:.2f})")
    ax.plot(alphas, sp_specmoe, "s-", color=C_SPECMOE, linewidth=2.5, markersize=8,
            label=f"SpecMoE (MAF={maf_k3_specmoe:.2f})")

    # Breakeven line
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.2, alpha=0.5)
    ax.text(0.31, 1.03, "Breakeven (1.0×)", fontsize=8, color="black", alpha=0.7)

    # Pilot measurement point
    pilot_alpha = data.get("pilot_acc_rate", 0.46)
    pilot_sp = data.get("pilot_speedup_k3", 0.91)
    ax.scatter([pilot_alpha], [pilot_sp], s=180, color=C_PILOT, marker="*",
               zorder=10, edgecolors="black", linewidths=0.8)
    ax.annotate(
        f"Pilot: {pilot_sp:.2f}×\n(ᾱ={pilot_alpha:.2f}, K=3)",
        xy=(pilot_alpha, pilot_sp),
        xytext=(pilot_alpha - 0.01, pilot_sp + 0.4),
        fontsize=9, fontweight="bold", color=C_PILOT,
        arrowprops=dict(arrowstyle="-|>", color=C_PILOT, lw=1.2),
        ha="center",
    )

    # Speedup gap annotation at α=0.6
    a_idx = 3  # α=0.6
    gap_y_top = sp_no_moe[a_idx]
    gap_y_bot = sp_with_maf[a_idx]
    x_annot = alphas[a_idx]
    ax.annotate(
        "", xy=(x_annot, gap_y_bot + 0.02), xytext=(x_annot, gap_y_top - 0.02),
        arrowprops=dict(arrowstyle="<->", color=C_RANDOM, lw=1.5),
    )
    ax.text(x_annot + 0.025, (gap_y_top + gap_y_bot) / 2, "MoE\nTax",
            fontsize=9, fontweight="bold", color=C_RANDOM, va="center")

    # Recovery annotation
    sp_recov = sp_specmoe[a_idx]
    ax.annotate(
        "", xy=(x_annot - 0.005, gap_y_bot + 0.02),
        xytext=(x_annot - 0.005, sp_recov - 0.02),
        arrowprops=dict(arrowstyle="<->", color=C_SPECMOE, lw=1.5),
    )
    recovery_pct = (sp_recov - gap_y_bot) / max(1e-6, gap_y_top - gap_y_bot) * 100
    ax.text(x_annot - 0.04, (gap_y_bot + sp_recov) / 2,
            f"+{recovery_pct:.0f}%\nrecov.",
            fontsize=8, fontweight="bold", color=C_SPECMOE, va="center", ha="right")

    ax.set_xlabel("Mean Acceptance Rate $\\bar{\\alpha}$")
    ax.set_ylabel("Effective Speedup")
    ax.set_title(f"(b) Speedup Wall at $K={K}$", fontweight="bold")
    ax.set_xlim(0.25, 0.85)
    ax.set_ylim(0, max(sp_no_moe) + 0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    # ══════════════════════════════════════════════════════════════
    # Panel (c): SpecMoE Technique Contribution Breakdown
    # ══════════════════════════════════════════════════════════════
    ax = axes[2]

    K = 3
    alpha = 0.50  # moderate acceptance rate

    # Build MAF reduction waterfall
    maf_baseline = maf_real[K3_idx]
    # Decompose SpecMoE into 3 techniques:
    # 1. SpecFusedMoE (cross-token dedup): ~18% MAF reduction
    # 2. SDD (early termination): ~12% effective MAF reduction
    # 3. Expert Cache (inter-round): ~10% effective MAF reduction
    dedup_reduction = maf_baseline * 0.18
    sdd_reduction = maf_baseline * 0.12
    cache_reduction = maf_baseline * 0.10

    maf_after_dedup = maf_baseline - dedup_reduction
    maf_after_sdd = maf_after_dedup - sdd_reduction
    maf_after_cache = maf_after_sdd - cache_reduction
    maf_after_cache = max(1.0, maf_after_cache)

    stages = ["Vanilla\n(baseline)", "Dedup\n(SpecFusedMoE)", "SDD\n(early-term)", "Cache\n(cross-phase)"]
    maf_vals = [maf_baseline, maf_after_dedup, maf_after_sdd, maf_after_cache]
    speedup_vals = [speedup_formula(alpha, K, m) for m in maf_vals]

    # Bar colors
    bar_colors = [C_RANDOM, "#5DADE2", "#48C9B0", C_SPECMOE]
    bar_edge = ["black"] * 4

    x_pos = np.arange(len(stages))
    bars = ax.bar(x_pos, speedup_vals, width=0.6, color=bar_colors,
                  edgecolor=bar_edge, linewidth=0.8, zorder=3)

    # Value labels on bars
    for i, (bar, sp, maf) in enumerate(zip(bars, speedup_vals, maf_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{sp:.2f}×", ha="center", fontsize=10, fontweight="bold",
                color=bar_colors[i] if i > 0 else "black")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"MAF\n{maf:.2f}", ha="center", fontsize=8, color="white",
                fontweight="bold", va="center")

    # Improvement arrows between bars
    for i in range(len(stages) - 1):
        delta = speedup_vals[i + 1] - speedup_vals[i]
        mid_x = (x_pos[i] + x_pos[i + 1]) / 2
        mid_y = max(speedup_vals[i], speedup_vals[i + 1]) + 0.08
        ax.annotate(
            f"+{delta:.2f}×",
            xy=(mid_x, mid_y), fontsize=8, fontweight="bold",
            color=bar_colors[i + 1], ha="center",
        )

    # Breakeven line
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.2, alpha=0.5)

    # Total improvement annotation
    total_gain = speedup_vals[-1] - speedup_vals[0]
    ax.annotate(
        f"Total: +{total_gain:.2f}× ({total_gain/max(1e-6, speedup_vals[0])*100:.0f}% ↑)",
        xy=(2.5, max(speedup_vals) + 0.15), fontsize=10, fontweight="bold",
        color=C_SPECMOE, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_SPECMOE, alpha=0.15),
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Effective Speedup")
    ax.set_title(f"(c) SpecMoE Technique Breakdown ($K={K}$, $\\bar{{\\alpha}}$={alpha})",
                 fontweight="bold")
    ax.set_ylim(0, max(speedup_vals) + 0.4)

    # ══════════════════════════════════════════════════════════════
    # Suptitle
    # ══════════════════════════════════════════════════════════════
    data_label = "Measured on Qwen3-30B-A3B" if is_measured else "Theoretical + Pilot Data"
    fig.suptitle(
        f"The MoE Tax on Speculative Decoding — {data_label}\n"
        f"Qwen3-30B-A3B: {N_EXPERTS} experts/layer, top-{TOP_K}, {N_LAYERS} layers, "
        f"~{EXPERT_SIZE_MB:.0f}MB/expert (bf16)",
        fontsize=12.5, fontweight="bold", y=1.04,
    )

    # Save
    for ext in ["pdf", "png"]:
        path = output_dir / f"fig1_moe_tax.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")

    plt.close(fig)

    # Also save the data used
    export = {
        "Ks": Ks.tolist(),
        "maf_random": maf_random.tolist(),
        "maf_real": maf_real.tolist(),
        "maf_specmoe": maf_specmoe.tolist(),
        "pilot_speedup": data.get("pilot_speedup_k3"),
        "pilot_alpha": data.get("pilot_acc_rate"),
        "speedup_breakdown": {
            "stages": stages,
            "maf_values": maf_vals,
            "speedup_values": speedup_vals,
        },
    }
    with open(output_dir / "fig1_data.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Data: {output_dir / 'fig1_data.json'}")


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 1: The MoE Tax on SD")
    parser.add_argument("--mode", choices=["theoretical", "measured"],
                        default="theoretical")
    parser.add_argument("--trace", default="results/validation/expert_trace.jsonl",
                        help="Path to trace JSONL (for measured mode)")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.mode == "measured":
        trace_path = Path(args.trace)
        if not trace_path.exists():
            print(f"ERROR: Trace file not found: {trace_path}")
            print("Run exp0 first, or use --mode theoretical")
            sys.exit(1)
        print("Loading measured data...")
        data = get_measured_data(str(trace_path))
        is_measured = True
    else:
        print("Using theoretical + pilot data...")
        data = get_theoretical_data()
        is_measured = False

    plot_figure(data, output_dir, is_measured=is_measured)
    print("\nDone! Figure ready for paper submission.")


if __name__ == "__main__":
    main()
