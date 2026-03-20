#!/usr/bin/env python3
"""
Intuitive Redundancy Visualization for Expert Routing Overlap
==============================================================
Redesigned for maximum visual impact showing redundancy severity.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_PATH = "/root/MoE-SD/results/routing_overlap_analysis.json"
TRACE_PATH = "/tmp/routing_trace.jsonl"
OUTPUT_PATH = "/root/MoE-SD/results/routing_overlap_intuitive.png"

with open(RESULTS_PATH) as f:
    results = json.load(f)

# ── Extract data ─────────────────────────────────────────────────────────────
layers = sorted(int(k) for k in results["per_layer"].keys())
layer_union = [results["per_layer"][str(l)]["union_size"]["mean"] for l in layers]
layer_jaccard = [results["per_layer"][str(l)]["jaccard"]["mean"] for l in layers]

TOP_K = results["top_k"]           # 8
WINDOW = results["window_size"]    # 4
MAX_SLOTS = WINDOW * TOP_K         # 32
NUM_EXPERTS = results["num_experts"]  # 128
mean_union = results["union_size"]["mean"]
redundancy_pct = (1 - results["dedup_ratio"]["mean"]) * 100

# Load raw trace for detailed analysis
events = []
with open(TRACE_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            events.append(json.loads(line))

by_req_layer = defaultdict(list)
for e in events:
    by_req_layer[(e["request_id"], e["layer_id"])].append(e)
for key in by_req_layer:
    by_req_layer[key].sort(key=lambda x: x["token_idx"])

# Compute all union sizes
all_union_sizes = []
for (req_id, layer_id), token_events in by_req_layer.items():
    if len(token_events) < WINDOW:
        continue
    for start in range(len(token_events) - WINDOW + 1):
        window = token_events[start:start + WINDOW]
        expert_sets = [set(e["experts"]) for e in window]
        union_all = set()
        for s in expert_sets:
            union_all |= s
        all_union_sizes.append(len(union_all))

# ── COLOR SCHEME ─────────────────────────────────────────────────────────────
C_RED = "#E74C3C"       # redundant
C_RED_LIGHT = "#FADBD8"
C_GREEN = "#27AE60"     # unique/effective
C_GREEN_LIGHT = "#D5F5E3"
C_BLUE = "#2980B9"
C_GRAY = "#BDC3C7"
C_DARK = "#2C3E50"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor="white")
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                       left=0.06, right=0.96, top=0.90, bottom=0.07)

fig.suptitle("Expert Routing Redundancy in EAGLE-3 Verify Batch\n"
             "Qwen3-30B-A3B  |  128 experts, top-8  |  K=3 (4-token batch)",
             fontsize=16, fontweight="bold", y=0.97)

# ══════════════════════════════════════════════════════════════════════════════
# (A) GIANT DONUT — overall redundancy
# ══════════════════════════════════════════════════════════════════════════════
ax_donut = fig.add_subplot(gs[0, 0])

unique_pct = 100 - redundancy_pct
sizes = [redundancy_pct, unique_pct]
colors = [C_RED, C_GREEN]
explode = (0.05, 0)

wedges, texts, autotexts = ax_donut.pie(
    sizes, explode=explode, colors=colors, autopct="",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))

# Big center text
ax_donut.text(0, 0.08, f"{redundancy_pct:.0f}%", ha="center", va="center",
              fontsize=42, fontweight="bold", color=C_RED)
ax_donut.text(0, -0.22, "REDUNDANT", ha="center", va="center",
              fontsize=13, fontweight="bold", color=C_RED)

# Labels outside
ax_donut.annotate(f"Redundant\n{redundancy_pct:.1f}%\n({MAX_SLOTS - mean_union:.0f} experts)",
                  xy=(0.3, 0.6), fontsize=11, color=C_RED, fontweight="bold",
                  ha="center")
ax_donut.annotate(f"Unique\n{unique_pct:.1f}%\n({mean_union:.0f} experts)",
                  xy=(-0.3, -0.6), fontsize=11, color=C_GREEN, fontweight="bold",
                  ha="center")

ax_donut.set_title("(a) Overall Redundancy Rate", fontsize=13, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════════════════
# (B) STACKED BAR — per-layer redundant vs unique
# ══════════════════════════════════════════════════════════════════════════════
ax_stack = fig.add_subplot(gs[0, 1:])

layer_redundant = [MAX_SLOTS - u for u in layer_union]

bars_unique = ax_stack.bar(layers, layer_union, color=C_GREEN, edgecolor="white",
                            linewidth=0.3, label="Unique experts")
bars_redundant = ax_stack.bar(layers, layer_redundant, bottom=layer_union,
                               color=C_RED, edgecolor="white", linewidth=0.3,
                               alpha=0.85, label="Redundant (duplicated)")

ax_stack.axhline(y=mean_union, color=C_DARK, linestyle="--", linewidth=1.2, alpha=0.7)
ax_stack.text(max(layers) + 1.5, mean_union, f"avg={mean_union:.0f}",
              fontsize=10, color=C_DARK, va="center")
ax_stack.axhline(y=MAX_SLOTS, color=C_GRAY, linestyle=":", linewidth=0.8)
ax_stack.text(max(layers) + 1.5, MAX_SLOTS, f"total={MAX_SLOTS}",
              fontsize=9, color="gray", va="center")

# Annotate extreme layers
max_red_idx = np.argmax(layer_redundant)
min_red_idx = np.argmin(layer_redundant)
ax_stack.annotate(f"L{layers[max_red_idx]}: {layer_redundant[max_red_idx]:.0f} redundant\n({layer_redundant[max_red_idx]/MAX_SLOTS*100:.0f}%)",
                  xy=(layers[max_red_idx], MAX_SLOTS),
                  xytext=(layers[max_red_idx] + 5, MAX_SLOTS + 2),
                  arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
                  fontsize=9, color=C_RED, fontweight="bold")

ax_stack.set_xlabel("MoE Layer Index", fontsize=11)
ax_stack.set_ylabel("Expert Count (per 4-token batch)", fontsize=11)
ax_stack.set_title("(b) Per-Layer: Unique vs Redundant Expert Computations", fontsize=13, fontweight="bold")
ax_stack.set_xlim(-1, max(layers) + 6)
ax_stack.set_ylim(0, MAX_SLOTS + 6)
ax_stack.legend(loc="upper right", fontsize=10)

# ══════════════════════════════════════════════════════════════════════════════
# (C) WATERFALL — shows how 4 tokens' experts combine
# ══════════════════════════════════════════════════════════════════════════════
ax_water = fig.add_subplot(gs[1, 0])

# Pick a representative window from trace (one from the middle layer 31, which has most overlap)
example_window = None
for (req_id, layer_id), token_events in by_req_layer.items():
    if layer_id == 31 and len(token_events) >= WINDOW:
        example_window = token_events[:WINDOW]
        break

if example_window:
    token_labels = [f"Token {i}" for i in range(WINDOW)]
    # Cumulative union as we add each token
    cumul_union = []
    cumul_new = []
    running_set = set()
    for i, ev in enumerate(example_window):
        experts_i = set(ev["experts"])
        new_experts = experts_i - running_set
        running_set |= experts_i
        cumul_union.append(len(running_set))
        cumul_new.append(len(new_experts))

    # Stacked waterfall: each bar shows "previously seen" (gray) + "new" (green) + "wasted" (red)
    x = np.arange(WINDOW)
    previously_seen = [TOP_K - n for n, u in zip(cumul_new, cumul_union)]
    previously_seen[0] = 0  # first token has no previous

    bar_width = 0.6
    ax_water.bar(x, cumul_new, bar_width, bottom=[0]*WINDOW,
                 color=C_GREEN, edgecolor="white", label="New unique experts")
    ax_water.bar(x, previously_seen, bar_width, bottom=cumul_new,
                 color=C_RED, edgecolor="white", alpha=0.8, label="Already seen (redundant)")

    for i in range(WINDOW):
        ax_water.text(i, cumul_new[i]/2, f"+{cumul_new[i]}",
                      ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        if previously_seen[i] > 0:
            ax_water.text(i, cumul_new[i] + previously_seen[i]/2,
                          f"{previously_seen[i]}",
                          ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    # Show running union on top
    for i in range(WINDOW):
        ax_water.text(i, TOP_K + 0.4, f"Union={cumul_union[i]}",
                      ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    ax_water.set_xticks(x)
    ax_water.set_xticklabels(token_labels, fontsize=11)
    ax_water.set_ylabel("Experts per Token", fontsize=11)
    ax_water.set_ylim(0, TOP_K + 2)
    ax_water.set_title(f"(c) Example: How Experts Accumulate (Layer 31)",
                       fontsize=13, fontweight="bold")
    ax_water.legend(loc="upper left", fontsize=9)
    ax_water.axhline(y=TOP_K, color=C_GRAY, linestyle=":", linewidth=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# (D) REDUNDANCY RATE PER LAYER — line chart with danger zone
# ══════════════════════════════════════════════════════════════════════════════
ax_rate = fig.add_subplot(gs[1, 1])

layer_redundancy_rate = [(MAX_SLOTS - u) / MAX_SLOTS * 100 for u in layer_union]

# Color by severity
colors_line = []
for r in layer_redundancy_rate:
    if r >= 50:
        colors_line.append(C_RED)
    elif r >= 35:
        colors_line.append("#E67E22")  # orange
    else:
        colors_line.append(C_GREEN)

ax_rate.bar(layers, layer_redundancy_rate, color=colors_line, edgecolor="white", linewidth=0.3)

# Danger zones
ax_rate.axhspan(50, 60, alpha=0.08, color="red")
ax_rate.axhspan(35, 50, alpha=0.05, color="orange")
ax_rate.axhspan(0, 35, alpha=0.05, color="green")

ax_rate.axhline(y=redundancy_pct, color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Overall avg={redundancy_pct:.1f}%")
ax_rate.axhline(y=50, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
ax_rate.text(max(layers) + 1, 51, "50%", fontsize=9, color="red", va="bottom")

# Top layers annotation
sorted_by_red = sorted(zip(layers, layer_redundancy_rate), key=lambda x: x[1], reverse=True)
for lid, rate in sorted_by_red[:3]:
    ax_rate.annotate(f"L{lid}: {rate:.0f}%", xy=(lid, rate),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=8, fontweight="bold", color=C_RED)

ax_rate.set_xlabel("MoE Layer Index", fontsize=11)
ax_rate.set_ylabel("Redundancy Rate (%)", fontsize=11)
ax_rate.set_title("(d) Per-Layer Redundancy Rate", fontsize=13, fontweight="bold")
ax_rate.set_xlim(-1, max(layers) + 4)
ax_rate.set_ylim(0, max(layer_redundancy_rate) * 1.15)
ax_rate.legend(loc="upper right", fontsize=10)

# ══════════════════════════════════════════════════════════════════════════════
# (E) KEY TAKEAWAY BOX
# ══════════════════════════════════════════════════════════════════════════════
ax_box = fig.add_subplot(gs[1, 2])
ax_box.axis("off")

# Compute stats for the box
layers_above_50 = sum(1 for r in layer_redundancy_rate if r >= 50)
layers_above_40 = sum(1 for r in layer_redundancy_rate if r >= 40)
max_red = max(layer_redundancy_rate)
max_red_layer = layers[layer_redundancy_rate.index(max_red)]

box_text = (
    "KEY FINDINGS\n"
    "=" * 36 + "\n\n"
    f"4 tokens x top-8 = 32 expert slots\n"
    f"Only {mean_union:.0f} unique experts needed\n"
    f"{MAX_SLOTS - mean_union:.0f} slots ({redundancy_pct:.0f}%) WASTED\n\n"
    "-" * 36 + "\n\n"
    f"Worst layer: L{max_red_layer} ({max_red:.0f}% redundant)\n"
    f"Layers >50% redundant: {layers_above_50}/{len(layers)}\n"
    f"Layers >40% redundant: {layers_above_40}/{len(layers)}\n\n"
    "-" * 36 + "\n\n"
    "Avg shared experts per pair:\n"
    f"  {results['raw_overlap_per_pair']['mean']:.1f} / {TOP_K} "
    f"({results['raw_overlap_per_pair']['mean']/TOP_K*100:.0f}%)\n\n"
    "Adjacent tokens share:\n"
    f"  {results['adjacent_raw_overlap']['mean']:.1f} / {TOP_K} "
    f"({results['adjacent_raw_overlap']['mean']/TOP_K*100:.0f}%)\n\n"
    "-" * 36 + "\n\n"
    "=> If dedup is applied in\n"
    "   fused_moe verify kernel,\n"
    f"   ~{redundancy_pct:.0f}% compute can be saved"
)

ax_box.text(0.05, 0.95, box_text, transform=ax_box.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=C_RED_LIGHT, edgecolor=C_RED,
                      alpha=0.9, linewidth=2))

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved to {OUTPUT_PATH}")
