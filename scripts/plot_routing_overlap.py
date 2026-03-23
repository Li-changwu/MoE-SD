#!/usr/bin/env python3
"""
Visualize Expert Routing Overlap Analysis Results
==================================================

Generates a multi-panel figure showing:
  (a) Per-layer Jaccard overlap + shared expert count
  (b) Union size distribution (histogram)
  (c) Redundancy breakdown (stacked bar)
  (d) Pairwise shared expert count distribution (histogram)

Also prints out the redundancy calculation formula.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
RESULTS_PATH = "/root/MoE-SD/results/routing_overlap_analysis.json"
TRACE_PATH = "/tmp/routing_trace.jsonl"
OUTPUT_PATH = "/root/MoE-SD/results/routing_overlap_figure.png"

with open(RESULTS_PATH) as f:
    results = json.load(f)

# ── Extract per-layer data ───────────────────────────────────────────────────
layers = sorted(int(k) for k in results["per_layer"].keys())
layer_jaccard = [results["per_layer"][str(l)]["jaccard"]["mean"] for l in layers]
layer_shared = [results["per_layer"][str(l)]["raw_overlap"]["mean"] for l in layers]
layer_union = [results["per_layer"][str(l)]["union_size"]["mean"] for l in layers]
layer_adj_shared = [results["per_layer"][str(l)]["adjacent_raw"]["mean"] for l in layers]

TOP_K = results["top_k"]           # 8
WINDOW = results["window_size"]    # 4
MAX_SLOTS = WINDOW * TOP_K         # 32
NUM_EXPERTS = results["num_experts"]  # 128

# ── Compute union-size distribution from trace ───────────────────────────────
# Re-derive union size histogram from the trace for a finer view
union_sizes_per_layer = {}
for l in layers:
    mean = results["per_layer"][str(l)]["union_size"]["mean"]
    union_sizes_per_layer[l] = mean

# ── Also load raw trace data for histograms ──────────────────────────────────
from collections import defaultdict

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

# Compute per-window union sizes and pairwise overlaps
all_union_sizes = []
all_pairwise_shared = []
all_pairwise_jaccard_vals = []

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
        for ii in range(WINDOW):
            for jj in range(ii + 1, WINDOW):
                shared = len(expert_sets[ii] & expert_sets[jj])
                all_pairwise_shared.append(shared)
                union_pair = len(expert_sets[ii] | expert_sets[jj])
                all_pairwise_jaccard_vals.append(shared / union_pair if union_pair > 0 else 0)

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                       left=0.07, right=0.95, top=0.93, bottom=0.06)

# ── (a) Per-layer Jaccard overlap (bar chart) ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
colors = plt.cm.RdYlGn_r(np.array(layer_jaccard) / max(layer_jaccard))
bars = ax1.bar(layers, [j * 100 for j in layer_jaccard], color=colors, edgecolor="gray", linewidth=0.3)
ax1.set_xlabel("MoE Layer Index")
ax1.set_ylabel("Pairwise Jaccard (%)")
ax1.set_title("(a) Per-Layer Expert Routing Overlap (4-token verify window, top-8 from 128 experts)")
ax1.axhline(y=results["pairwise_jaccard"]["mean"] * 100, color="red", linestyle="--",
            linewidth=1.5, label=f'Overall mean = {results["pairwise_jaccard"]["mean"]*100:.1f}%')
ax1.set_xlim(-1, max(layers) + 1)
ax1.set_ylim(0, max(layer_jaccard) * 100 * 1.15)

# Annotate top layers
top5 = sorted(range(len(layer_jaccard)), key=lambda i: layer_jaccard[i], reverse=True)[:3]
for idx in top5:
    ax1.annotate(f"L{layers[idx]}\n{layer_jaccard[idx]*100:.1f}%",
                 xy=(layers[idx], layer_jaccard[idx] * 100),
                 xytext=(0, 12), textcoords="offset points",
                 ha="center", fontsize=9, fontweight="bold", color="darkred")
ax1.legend(loc="upper right", fontsize=10)

# ── (b) Per-layer union size (line chart with fill) ─────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(layers, layer_union, MAX_SLOTS, alpha=0.3, color="red",
                 label=f"Redundant (avg {MAX_SLOTS - np.mean(layer_union):.1f})")
ax2.fill_between(layers, TOP_K, layer_union, alpha=0.3, color="green",
                 label=f"Unique extra (beyond {TOP_K})")
ax2.fill_between(layers, 0, [TOP_K] * len(layers), alpha=0.2, color="blue",
                 label=f"Min required ({TOP_K})")
ax2.plot(layers, layer_union, "ko-", markersize=3, linewidth=1.2, label=f"Union size (avg={np.mean(layer_union):.1f})")
ax2.axhline(y=MAX_SLOTS, color="red", linestyle=":", linewidth=1, alpha=0.5)
ax2.axhline(y=TOP_K, color="blue", linestyle=":", linewidth=1, alpha=0.5)
ax2.set_xlabel("MoE Layer Index")
ax2.set_ylabel("Expert Count")
ax2.set_title(f"(b) Union Size per Verify Batch (4 tokens × top-8 = 32 slots)")
ax2.set_xlim(-1, max(layers) + 1)
ax2.set_ylim(0, MAX_SLOTS + 2)
ax2.legend(loc="upper right", fontsize=8)

# Add text annotations for bounds
ax2.text(max(layers) + 0.5, MAX_SLOTS + 0.5, f"{MAX_SLOTS}", fontsize=9, color="red", ha="left")
ax2.text(max(layers) + 0.5, TOP_K + 0.5, f"{TOP_K}", fontsize=9, color="blue", ha="left")

# ── (c) Union size histogram ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
bins = np.arange(TOP_K - 0.5, MAX_SLOTS + 1.5, 1)
counts, bin_edges, patches = ax3.hist(all_union_sizes, bins=bins, color="steelblue",
                                       edgecolor="white", linewidth=0.5, density=True)
ax3.axvline(x=np.mean(all_union_sizes), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {np.mean(all_union_sizes):.1f}")
ax3.axvline(x=MAX_SLOTS, color="gray", linestyle=":", linewidth=1,
            label=f"Max = {MAX_SLOTS} (zero overlap)")
ax3.axvline(x=TOP_K, color="green", linestyle=":", linewidth=1,
            label=f"Min = {TOP_K} (full overlap)")
ax3.set_xlabel("Union Size (unique experts per 4-token batch)")
ax3.set_ylabel("Density")
ax3.set_title("(c) Distribution of Union Sizes")
ax3.legend(fontsize=9)

# ── (d) Pairwise shared expert count histogram ──────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
bins_shared = np.arange(-0.5, TOP_K + 1.5, 1)
ax4.hist(all_pairwise_shared, bins=bins_shared, color="coral", edgecolor="white",
         linewidth=0.5, density=True)
ax4.axvline(x=np.mean(all_pairwise_shared), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {np.mean(all_pairwise_shared):.2f}")
ax4.set_xlabel("Shared Expert Count (per token pair)")
ax4.set_ylabel("Density")
ax4.set_title(f"(d) Pairwise Shared Experts (each token selects top-{TOP_K})")
ax4.set_xticks(range(TOP_K + 1))
ax4.legend(fontsize=10)

# ── (e) Redundancy calculation explanation ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis("off")

mean_union = results["union_size"]["mean"]
mean_dedup = results["dedup_ratio"]["mean"]
redundancy = (1 - mean_dedup) * 100

text = (
    "Redundancy Calculation\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"EAGLE-3 verify batch: K=3 → {WINDOW} tokens\n"
    f"Each token: top-{TOP_K} from {NUM_EXPERTS} experts\n"
    f"Total dispatch slots: {WINDOW} × {TOP_K} = {MAX_SLOTS}\n\n"
    "For each 4-token window:\n"
    f"  union_size = |E₁ ∪ E₂ ∪ E₃ ∪ E₄|\n"
    f"            (= unique experts actually needed)\n\n"
    f"  dedup_ratio = union_size / total_slots\n"
    f"             = union_size / {MAX_SLOTS}\n\n"
    f"  redundancy  = 1 − dedup_ratio\n"
    f"             = ({MAX_SLOTS} − union_size) / {MAX_SLOTS}\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Measured Results:\n\n"
    f"  Avg union_size  = {mean_union:.1f}\n"
    f"  Avg dedup_ratio = {mean_union:.1f}/{MAX_SLOTS} = {mean_dedup:.4f}\n"
    f"  Avg redundancy  = ({MAX_SLOTS}−{mean_union:.1f})/{MAX_SLOTS}\n"
    f"                  = {redundancy:.1f}%\n\n"
    "=> Per verify batch:\n"
    f"  {MAX_SLOTS} expert dispatches need only {mean_union:.0f} unique\n"
    f"  {MAX_SLOTS - mean_union:.0f} dispatches ({redundancy:.0f}%) are REDUNDANT"
)

ax5.text(0.05, 0.95, text, transform=ax5.transAxes,
         fontsize=11, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

# ── Title ────────────────────────────────────────────────────────────────────
fig.suptitle("Expert Routing Overlap Analysis — Qwen3-30B-A3B (128E, top-8, 48 layers)\n"
             "EAGLE-3 K=3 verify batch = 4 tokens",
             fontsize=15, fontweight="bold", y=0.98)

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved to {OUTPUT_PATH}")

# Also save a per-layer detailed CSV
csv_path = Path(OUTPUT_PATH).with_suffix(".csv")
with open(csv_path, "w") as f:
    f.write("layer,jaccard,shared_experts,union_size,adjacent_shared\n")
    for i, l in enumerate(layers):
        f.write(f"{l},{layer_jaccard[i]:.4f},{layer_shared[i]:.2f},"
                f"{layer_union[i]:.1f},{layer_adj_shared[i]:.2f}\n")
print(f"CSV saved to {csv_path}")
