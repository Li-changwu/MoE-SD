#!/usr/bin/env python3
"""Plot BriskMoE Phase Transition Experiment Results.

Reads results/phase_transition/{ar,sd}_offload_<GB>/result.json
and pcie_dmon.csv to produce:
  - Left panel:  TPS vs CPU Offload  (AR & SD curves)
  - Right panel: SD/AR speedup ratio vs CPU Offload (with 1.0× line)

Output: results/phase_transition/phase_transition.{pdf,png}
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase_transition"
)
OFFLOAD_VALUES = [20, 25, 30, 35, 40, 45]
MODEL_SIZE_GB = 57.0  # Qwen3-30B-A3B in BF16 (approximate)


def load_data():
    """Load TPS and PCIe data for each offload level."""
    offloads, ar_tps, sd_tps = [], [], []
    ar_pcie, sd_pcie = [], []

    for off in OFFLOAD_VALUES:
        ar_path = os.path.join(RESULT_DIR, f"ar_offload_{off}", "result.json")
        sd_path = os.path.join(RESULT_DIR, f"sd_offload_{off}", "result.json")

        if not os.path.exists(ar_path) or not os.path.exists(sd_path):
            print(f"  [WARN] Missing data for offload={off}, skipping")
            continue

        try:
            ar_d = json.load(open(ar_path))
            sd_d = json.load(open(sd_path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Bad JSON for offload={off}: {e}")
            continue

        offloads.append(off)
        ar_tps.append(ar_d["tps_mean"])
        sd_tps.append(sd_d["tps_mean"])

        # PCIe: average rxpci from dmon (optional)
        for tag, lst in [("ar", ar_pcie), ("sd", sd_pcie)]:
            ppath = os.path.join(RESULT_DIR, f"{tag}_offload_{off}", "pcie_dmon.csv")
            avg = parse_pcie_dmon(ppath)
            lst.append(avg)

    return offloads, ar_tps, sd_tps, ar_pcie, sd_pcie


def parse_pcie_dmon(path):
    """Parse nvidia-smi dmon -s t output → average rxpci in MB/s."""
    if not os.path.exists(path):
        return None
    rxvals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Format: gpu rxpci txpci
            if len(parts) >= 2:
                try:
                    rxvals.append(int(parts[1]))
                except ValueError:
                    pass
    return sum(rxvals) / len(rxvals) if rxvals else None


def plot(offloads, ar_tps, sd_tps, ar_pcie, sd_pcie):
    # ---- Style ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    offs = np.array(offloads)
    ar = np.array(ar_tps)
    sd = np.array(sd_tps)
    gpu_pct = (MODEL_SIZE_GB - offs) / MODEL_SIZE_GB * 100.0

    # ================== Left: TPS vs Offload ==================
    ax = axes[0]
    ax.plot(offs, ar, "o-", label="AR (Autoregressive)", color="#1f77b4",
            linewidth=2, markersize=7, zorder=3)
    ax.plot(offs, sd, "s-", label="SD (Speculative Decoding)", color="#ff7f0e",
            linewidth=2, markersize=7, zorder=3)

    # Fill region where SD > AR (green) and SD < AR (red)
    for i in range(len(offs) - 1):
        x_seg = [offs[i], offs[i + 1]]
        ar_seg = [ar[i], ar[i + 1]]
        sd_seg = [sd[i], sd[i + 1]]
        color = "#2ca02c" if sd[i] >= ar[i] else "#d62728"
        ax.fill_between(x_seg, ar_seg, sd_seg, alpha=0.10, color=color)

    # Mark crossover point (if any)
    for i in range(len(offs) - 1):
        if (sd[i] >= ar[i]) != (sd[i + 1] >= ar[i + 1]):
            # Linear interpolation
            denom = (sd[i] - ar[i]) - (sd[i + 1] - ar[i + 1])
            if abs(denom) > 1e-9:
                t = (sd[i] - ar[i]) / denom
                x_cross = offs[i] + t * (offs[i + 1] - offs[i])
                y_cross = ar[i] + t * (ar[i + 1] - ar[i])
                ax.axvline(x=x_cross, color="#d62728", ls="--", alpha=0.6, lw=1.2)
                ax.annotate(
                    f"Phase Transition\n≈{x_cross:.0f} GB",
                    xy=(x_cross, y_cross),
                    xytext=(x_cross + 2.5, y_cross + max(ar) * 0.12),
                    fontsize=10, color="#d62728", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
                )

    ax.set_xlabel("CPU Offload Budget (GB)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_title("MoE Offloading: SD Phase Transition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Secondary x-axis: GPU %
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = offs
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{p:.0f}%" for p in gpu_pct], fontsize=9)
    ax2.set_xlabel("Model on GPU (%)", fontsize=10)

    # ================== Right: Speedup Ratio ==================
    ax = axes[1]
    speedups = sd / ar
    ax.plot(offs, speedups, "D-", color="#2ca02c", linewidth=2, markersize=7, zorder=3)
    ax.axhline(y=1.0, color="#d62728", ls="--", alpha=0.6, lw=1.2, label="Break-even (1.0×)")

    # Annotate each point
    for i, (x, sp) in enumerate(zip(offs, speedups)):
        ax.annotate(f"{sp:.2f}×", (x, sp), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("CPU Offload Budget (GB)", fontsize=12)
    ax.set_ylabel("SD / AR  Speedup", fontsize=12)
    ax.set_title("Speedup Degrades with Offloading", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Secondary x-axis
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([f"{p:.0f}%" for p in gpu_pct], fontsize=9)
    ax3.set_xlabel("Model on GPU (%)", fontsize=10)

    # ================== Save ==================
    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(RESULT_DIR, f"phase_transition.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


def plot_pcie(offloads, ar_tps, sd_tps, ar_pcie, sd_pcie):
    """Optional: PCIe bandwidth per token generation cycle."""
    # Filter out None
    valid = [(o, at, st, ap, sp) for o, at, st, ap, sp
             in zip(offloads, ar_tps, sd_tps, ar_pcie, sd_pcie)
             if ap is not None and sp is not None]
    if not valid:
        print("No PCIe data available — skipping PCIe plot.")
        return

    offs, ats, sts, aps, sps = zip(*valid)
    offs = np.array(offs)

    # PCIe per token = avg_bandwidth_MB / tps → MB per token cycle
    ar_per_tok = np.array(aps) / np.array(ats)  # MB / (tok/s) = MB·s/tok
    sd_per_tok = np.array(sps) / np.array(sts)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(offs, ar_per_tok, "o-", label="AR", color="#1f77b4", lw=2, ms=7)
    ax.plot(offs, sd_per_tok, "s-", label="SD", color="#ff7f0e", lw=2, ms=7)
    ax.set_xlabel("CPU Offload Budget (GB)", fontsize=12)
    ax.set_ylabel("PCIe RX per Token Cycle (MB·s)", fontsize=12)
    ax.set_title("PCIe Transfer Volume per Token", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(RESULT_DIR, f"phase_transition_pcie.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


def main():
    if not os.path.isdir(RESULT_DIR):
        print(f"Result directory not found: {RESULT_DIR}")
        sys.exit(1)

    print(f"Loading data from: {RESULT_DIR}")
    offloads, ar_tps, sd_tps, ar_pcie, sd_pcie = load_data()

    if len(offloads) < 2:
        print(f"Need at least 2 data points, got {len(offloads)}. Run bench first.")
        sys.exit(1)

    print(f"Loaded {len(offloads)} offload levels: {offloads}")
    print(f"  AR TPS: {[f'{t:.2f}' for t in ar_tps]}")
    print(f"  SD TPS: {[f'{t:.2f}' for t in sd_tps]}")

    plot(offloads, ar_tps, sd_tps, ar_pcie, sd_pcie)
    plot_pcie(offloads, ar_tps, sd_tps, ar_pcie, sd_pcie)


if __name__ == "__main__":
    main()
