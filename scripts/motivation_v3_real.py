#!/usr/bin/env python3
"""
BriskMoE §2 Motivation — v3 (Chronic Overload Story, Real Data)
================================================================
Based on real Qwen3-30B-A3B expert routing traces from HumanEval.

Structure:
  Fig 2: Naive SD helps, but pushes expert memory into chronic overload
         Left:  Throughput vs memory budget (AR / naive SD)
         Right: Cache hit rate vs memory budget (AR η / SD η)

  Fig 3: Obs 1 — SD turns occasional overflow into chronic overload
         Left:  AR vs SD working-set CDF with cache capacity line
         Right: P(W>S) bar chart: AR vs SD

  Fig 4: Obs 2 — Under chronic overload, LRU cannot preserve useful residency
         Main:  LRU hit-rate time series + burst shading
         Inset: Cross-layer burst fraction

  Fig 5: Obs 3 — Lookahead oversubscribes transfer budget
         Left:  Demand distribution vs PCIe budget
         Right: FIFO / Random / Oracle urgency-weighted coverage

No solution previews (SACR / ELP / DIPP / BriskMoE) in any figure.

Run:
    cd /root/MoE-SD && python scripts/motivation_v3_real.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
TRACE_PATH  = ROOT / "results" / "real_trace" / "expert_trace_humaneval.jsonl"
HITRATE_CSV = ROOT / "results" / "obs_experiments" / "hitrate_sweep.csv"
OUT_DIR     = ROOT / "results" / "motivation_figures_v3"

# ── Model / SD parameters ───────────────────────────────────────────
NUM_LAYERS        = 48
NUM_EXPERTS       = 128
TOP_K_EXPERTS     = 8
DRAFT_K           = 3
ACCEPT_RATE       = 0.625
SEED              = 42

# ── Hardware ─────────────────────────────────────────────────────────
EXPERT_SIZE_BYTES = 9_437_184   # ~9.44 MB per expert
PCIE_BW_BPS       = 25e9       # PCIe Gen4 x16
DRAFT_LATENCY_S   = 0.030      # ~30 ms

# ── Publication style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# Colors: AR = blue palette, SD = red/orange palette
C_AR   = "#2196F3"  # blue
C_SD   = "#FF5722"  # red-orange
C_MISS = "#F44336"  # red
C_HIT  = "#4CAF50"  # green


# ====================================================================
# Load & convert real trace
# ====================================================================

def load_trace():
    """Load real expert trace and return raw token-level data."""
    events = []
    with open(TRACE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    print(f"  Loaded {len(events)} events from {TRACE_PATH.name}")

    # Group: (request_id, token_idx) -> {layer_id: [expert_ids]}
    token_data: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(dict)
    for ev in events:
        key = (ev["request_id"], ev["token_idx"])
        token_data[key][ev["layer_id"]] = ev["experts"]

    all_tokens = sorted(token_data.keys())
    print(f"  Total decode tokens: {len(all_tokens)}")
    return token_data, all_tokens


def build_sd_trace(token_data, all_tokens):
    """Group AR tokens into SD steps (K+1 per step) with acceptance mask."""
    req_tokens: dict[str, list] = defaultdict(list)
    for key in all_tokens:
        req_tokens[key[0]].append(key)

    rng = random.Random(SEED)
    sd_trace = []
    global_step = 0
    step_size = DRAFT_K + 1

    for _, tokens in sorted(req_tokens.items()):
        for start in range(0, len(tokens), step_size):
            chunk = tokens[start: start + step_size]
            if len(chunk) < 2:
                continue

            accepted_mask = [True]
            for _ in range(1, len(chunk)):
                if accepted_mask[-1] and rng.random() < ACCEPT_RATE:
                    accepted_mask.append(True)
                else:
                    accepted_mask.append(False)

            for layer_id in range(NUM_LAYERS):
                token_expert_map = {}
                for tok_pos, tk in enumerate(chunk):
                    experts = token_data[tk].get(layer_id, [])
                    if experts:
                        token_expert_map[tok_pos] = experts

                if token_expert_map:
                    sd_trace.append({
                        "step": global_step,
                        "layer_id": layer_id,
                        "token_expert_map": token_expert_map,
                        "accepted_mask": accepted_mask,
                    })
            global_step += 1

    print(f"  SD trace: {global_step} steps × {NUM_LAYERS} layers = {len(sd_trace)} entries")
    return sd_trace


# ====================================================================
# Fig 2: Naive SD helps, but enters chronic overload
# ====================================================================

def plot_fig2():
    if not HITRATE_CSV.exists():
        print("[Fig 2] SKIP — hitrate_sweep.csv not found")
        return

    rows = list(csv.DictReader(open(HITRATE_CSV)))
    ar_rows = [r for r in rows if r["config"] == "AR"]
    sd_rows = [r for r in rows if r["config"] == "SD"]

    mem_ar  = [float(r["memory_gib"]) for r in ar_rows]
    tps_ar  = [float(r["avg_tps"])    for r in ar_rows]
    eta_ar  = [float(r["hit_rate"]) * 100 for r in ar_rows]
    mem_sd  = [float(r["memory_gib"]) for r in sd_rows]
    tps_sd  = [float(r["avg_tps"])    for r in sd_rows]
    eta_sd  = [float(r["hit_rate"]) * 100 for r in sd_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # ── Left: Throughput ──
    ax1.plot(mem_ar, tps_ar, "o-", color=C_AR, lw=2, ms=6, label="AR")
    ax1.plot(mem_sd, tps_sd, "s-", color=C_SD, lw=2, ms=6, label="Naive SD")
    ax1.set_xlabel("GPU Memory Budget (GB)")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_xlim(22, 46)
    ax1.set_ylim(0, 8)
    ax1.legend(loc="lower right")
    ax1.set_title("Throughput", pad=6)

    ratio_max = tps_sd[-1] / tps_ar[-1]
    ax1.annotate(f"{ratio_max:.1f}× speedup",
                 xy=(44, tps_sd[-1]), xytext=(34, 7.2), fontsize=8.5,
                 color="#BF360C",
                 arrowprops=dict(arrowstyle="->", color="#BF360C", lw=0.8))
    ax1.annotate("Diminishing\nreturns",
                 xy=(36, tps_ar[3]), xytext=(26, 5.2), fontsize=8,
                 color="#1565C0",
                 arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8))

    # ── Right: Hit rate ──
    ax2.plot(mem_ar, eta_ar, "o-", color=C_AR, lw=2, ms=6, label=r"AR $\eta$")
    ax2.plot(mem_sd, eta_sd, "s-", color=C_SD, lw=2, ms=6, label=r"SD $\eta$")
    ax2.set_xlabel("GPU Memory Budget (GB)")
    ax2.set_ylabel("Cache Hit Rate (%)")
    ax2.set_xlim(22, 46)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")
    ax2.set_title("Cache Hit Rate", pad=6)
    ax2.fill_between(mem_ar, eta_ar,
                     eta_sd[:len(mem_ar)] if len(eta_sd) >= len(mem_ar) else eta_sd,
                     alpha=0.12, color=C_SD)
    ax2.annotate("SD pushes cache\ninto chronic overload",
                 xy=(32, (eta_ar[2] + eta_sd[2]) / 2),
                 xytext=(26, 28), fontsize=8.5, color="#BF360C",
                 arrowprops=dict(arrowstyle="->", color="#BF360C", lw=0.8))

    fig.suptitle("Fig. 2: Naive SD improves throughput but drives expert memory into chronic overload",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_naive_sd.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_naive_sd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] AR@{mem_ar[-1]:.0f}G={tps_ar[-1]:.2f} | SD={tps_sd[-1]:.2f} ({ratio_max:.2f}×) | "
          f"AR η={eta_ar[-1]:.1f}% SD η={eta_sd[-1]:.1f}%")


# ====================================================================
# Fig 3: Obs 1 — SD turns occasional overflow into chronic overload
# ====================================================================

def plot_fig3(token_data, all_tokens, cache_size: int):
    """
    Left:  CDF of per-layer per-step working set under AR vs SD
    Right: P(W>S) bar chart
    """
    # ── AR working set: 1 token per step → W = |experts| = top-K ──
    ar_ws = []
    for key in all_tokens:
        for l in range(NUM_LAYERS):
            experts = token_data[key].get(l, [])
            ar_ws.append(len(experts))

    # ── SD working set: K+1 tokens per step → W = |union| ──
    req_tokens: dict[str, list] = defaultdict(list)
    for key in all_tokens:
        req_tokens[key[0]].append(key)

    sd_ws = []
    step_size = DRAFT_K + 1
    for _, tokens in sorted(req_tokens.items()):
        for start in range(0, len(tokens), step_size):
            chunk = tokens[start: start + step_size]
            if len(chunk) < 2:
                continue
            for l in range(NUM_LAYERS):
                expert_union = set()
                for tk in chunk:
                    expert_union.update(token_data[tk].get(l, []))
                sd_ws.append(len(expert_union))

    ar_ws = np.array(ar_ws)
    sd_ws = np.array(sd_ws)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # ── Left: CDF ──
    ar_sorted = np.sort(ar_ws)
    sd_sorted = np.sort(sd_ws)
    ar_cdf = np.arange(1, len(ar_sorted) + 1) / len(ar_sorted)
    sd_cdf = np.arange(1, len(sd_sorted) + 1) / len(sd_sorted)

    ax1.plot(ar_sorted, ar_cdf, "-", color=C_AR, lw=2.2, label="AR")
    ax1.plot(sd_sorted, sd_cdf, "-", color=C_SD, lw=2.2, label="SD (K=3)")
    ax1.axvline(x=cache_size, color="gray", ls="--", lw=1.5,
                label=f"Cache capacity S={cache_size}")
    ax1.set_xlabel("Working Set Size per Layer per Step")
    ax1.set_ylabel("CDF")
    ax1.set_xlim(0, max(sd_ws.max(), cache_size + 5) + 2)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title("Working-Set Distribution", pad=6)

    # Shade the overflow region
    ax1.axvspan(cache_size, ax1.get_xlim()[1], alpha=0.08, color=C_MISS,
                zorder=0, label="_")
    ax1.text(cache_size + 1, 0.15, "Overflow\nregion", fontsize=8, color="#B71C1C")

    # ── Right: P(W>S) bar chart ──
    cache_sizes_to_show = [cache_size]
    ar_over = (ar_ws > cache_size).mean() * 100
    sd_over = (sd_ws > cache_size).mean() * 100

    bars = ax2.bar(
        ["AR", "SD (K=3)"],
        [ar_over, sd_over],
        color=[C_AR, C_SD],
        width=0.45,
        edgecolor="black",
        lw=0.7,
    )
    ax2.set_ylabel("P(W > S) (%)")
    ax2.set_title(f"Overflow Probability (S={cache_size})", pad=6)
    ax2.set_ylim(0, 115)
    for bar, val in zip(bars, [ar_over, sd_over]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    # Stats annotation
    ax2.text(0.5, 0.55,
             f"AR: mean W = {ar_ws.mean():.0f}\n"
             f"SD: mean W = {sd_ws.mean():.1f}\n"
             f"SD/AR ratio = {sd_ws.mean()/ar_ws.mean():.1f}×",
             transform=ax2.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#FBC02D", lw=0.8))

    fig.suptitle("Fig. 3: SD turns occasional overflow into chronic overload",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_obs1_overload.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_obs1_overload.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig 3] AR: W mean={ar_ws.mean():.1f} | P(W>S)={ar_over:.1f}%")
    print(f"[Fig 3] SD: W mean={sd_ws.mean():.1f} | P(W>S)={sd_over:.1f}% | "
          f"expansion={sd_ws.mean()/ar_ws.mean():.1f}×")


# ====================================================================
# Fig 4: Obs 2 — Under chronic overload, LRU cannot preserve residency
# ====================================================================

def plot_fig4(sd_trace: list[dict], cache_size: int):
    max_step   = max(e["step"] for e in sd_trace)
    num_layers = NUM_LAYERS

    lru:     dict[int, list[int]] = defaultdict(list)
    lru_set: dict[int, set[int]]  = defaultdict(set)

    step_layer_hits:   dict[tuple[int, int], int] = defaultdict(int)
    step_layer_misses: dict[tuple[int, int], int] = defaultdict(int)
    step_layer_ws:     dict[tuple[int, int], int] = defaultdict(int)

    for entry in sd_trace:
        step     = entry["step"]
        layer_id = entry["layer_id"]
        all_experts = set()
        for _, experts in entry["token_expert_map"].items():
            all_experts.update(experts)
        step_layer_ws[(step, layer_id)] = len(all_experts)

        s_h = 0; s_m = 0
        for e in all_experts:
            if e in lru_set[layer_id]:
                s_h += 1
                lru[layer_id].remove(e)
                lru[layer_id].append(e)
            else:
                s_m += 1
                if len(lru[layer_id]) >= cache_size:
                    victim = lru[layer_id].pop(0)
                    lru_set[layer_id].discard(victim)
                lru[layer_id].append(e)
                lru_set[layer_id].add(e)
        step_layer_hits[(step, layer_id)]   = s_h
        step_layer_misses[(step, layer_id)] = s_m

    target_layer = num_layers // 2  # Layer 24

    per_step_hr = []
    per_step_ws = []
    for s in range(max_step + 1):
        h = step_layer_hits.get((s, target_layer), 0)
        m = step_layer_misses.get((s, target_layer), 0)
        per_step_hr.append(h / max(1, h + m))
        per_step_ws.append(step_layer_ws.get((s, target_layer), 0))

    per_step_burst_frac = []
    for s in range(max_step + 1):
        nb = sum(1 for l in range(num_layers)
                 if step_layer_ws.get((s, l), 0) > cache_size)
        per_step_burst_frac.append(nb / num_layers)

    # Cross-layer average HR per step
    per_step_avg_hr = []
    for s in range(max_step + 1):
        th = sum(step_layer_hits.get((s, l), 0) for l in range(num_layers))
        tm = sum(step_layer_misses.get((s, l), 0) for l in range(num_layers))
        per_step_avg_hr.append(th / max(1, th + tm))

    # Display window
    warmup = min(15, max_step // 5)
    s_end  = min(max_step + 1, warmup + 85)
    steps_w = list(range(warmup, s_end))
    hr_w    = per_step_hr[warmup:s_end]
    ws_w    = per_step_ws[warmup:s_end]
    burst_w = per_step_burst_frac[warmup:s_end]

    if not hr_w:
        print("[Fig 4] Not enough steps")
        return

    fig, ax = plt.subplots(figsize=(9, 4.0))

    def smooth(data, w=5):
        kernel = np.ones(w) / w
        return np.convolve(data, kernel, mode="same")

    # Raw (faint)
    ax.plot(steps_w, [h * 100 for h in hr_w], "-", color=C_MISS, lw=0.5, alpha=0.25)
    # Smoothed
    hr_smooth = smooth([h * 100 for h in hr_w])
    ax.plot(steps_w, hr_smooth, "-", color=C_MISS, lw=2.2, alpha=0.9,
            label=f"LRU (Layer {target_layer})")

    # Burst shading
    burst_steps = [s for s, w in zip(steps_w, ws_w) if w > cache_size]
    for bs in burst_steps:
        ax.axvspan(bs - 0.4, bs + 0.4, color="#FFCDD2", alpha=0.30, zorder=0)

    n_burst = len(burst_steps)
    burst_pct = n_burst / max(1, len(steps_w)) * 100

    baseline = cache_size / (cache_size + TOP_K_EXPERTS) * 100
    ax.axhline(y=baseline, color="gray", ls=":", lw=0.8, label=f"Steady-state baseline ({baseline:.0f}%)")

    ax.set_xlabel("SD Step")
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Fig. 4: Under chronic overload, LRU cannot preserve useful residency",
                 fontsize=10.5, pad=8)

    lru_avg = np.mean(hr_w) * 100
    info = (f"LRU avg HR = {lru_avg:.1f}%  (baseline = {baseline:.0f}%)\n"
            f"Burst steps: {n_burst}/{len(steps_w)} ({burst_pct:.0f}%)\n"
            f"Cache S={cache_size}, W mean={np.mean(ws_w):.0f}")
    ax.text(0.02, 0.05, info, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#FBC02D", lw=0.8))

    # ── Inset: Cross-layer burst fraction ──
    ax_in = ax.inset_axes([0.62, 0.55, 0.35, 0.38])
    avg_burst_smooth = smooth([b * 100 for b in burst_w], w=3)
    ax_in.bar(steps_w, [b * 100 for b in burst_w], color="#FFCDD2",
              edgecolor="none", width=1.0)
    ax_in.plot(steps_w, avg_burst_smooth, "-", color="#D32F2F", lw=1.5)
    ax_in.set_ylabel("Layers\nw/ burst (%)", fontsize=7)
    ax_in.set_xlabel("Step", fontsize=7)
    ax_in.set_title("Cross-Layer Burst Fraction", fontsize=8, pad=3)
    ax_in.tick_params(labelsize=7)
    ax_in.set_ylim(0, 105)
    avg_burst = np.mean(burst_w) * 100
    ax_in.axhline(y=avg_burst, color="#D32F2F", ls="--", lw=0.8)
    ax_in.text(0.95, 0.12, f"avg={avg_burst:.0f}%", transform=ax_in.transAxes,
               ha="right", fontsize=7, color="#D32F2F",
               bbox=dict(fc="white", ec="none", alpha=0.7))

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_obs2_residency.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_obs2_residency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig 4] Layer {target_layer}: LRU avg HR={lru_avg:.1f}% | "
          f"burst={n_burst}/{len(steps_w)} ({burst_pct:.0f}%) | "
          f"cross-layer burst={avg_burst:.1f}%")


# ====================================================================
# Fig 5: Obs 3 — Lookahead oversubscribes transfer budget
# ====================================================================

def plot_fig5(sd_trace: list[dict], cache_size: int):
    budget = int(DRAFT_LATENCY_S * PCIE_BW_BPS / EXPERT_SIZE_BYTES)
    max_step = max(e["step"] for e in sd_trace)

    steps_data: dict[int, list[dict]] = defaultdict(list)
    for entry in sd_trace:
        steps_data[entry["step"]].append(entry)

    lru:     dict[int, list[int]] = defaultdict(list)
    lru_set: dict[int, set[int]]  = defaultdict(set)

    per_step_demand  = []
    fifo_coverages   = []
    random_coverages = []
    oracle_coverages = []

    rng = random.Random(SEED + 1)

    for step in range(max_step + 1):
        entries = steps_data.get(step, [])
        cache_snapshot = {l: set(lru_set[l]) for l in range(NUM_LAYERS)}

        step_miss_experts: dict[int, set[int]] = defaultdict(set)
        for entry in entries:
            layer_id = entry["layer_id"]
            for _, experts in entry["token_expert_map"].items():
                for e in experts:
                    if e not in cache_snapshot.get(layer_id, set()):
                        step_miss_experts[layer_id].add(e)

        total_misses = sum(len(v) for v in step_miss_experts.values())
        per_step_demand.append(total_misses)

        miss_list = [(l, e) for l, exps in step_miss_experts.items() for e in exps]

        if total_misses > 0:
            expert_demand_count: dict[tuple[int, int], int] = defaultdict(int)
            expert_urgency: dict[tuple[int, int], float] = {}
            miss_set = set(miss_list)
            for entry in entries:
                layer_id = entry["layer_id"]
                for _, experts in entry["token_expert_map"].items():
                    for e in experts:
                        key = (layer_id, e)
                        if key in miss_set:
                            expert_demand_count[key] += 1
                            expert_urgency[key] = 1.0 / (layer_id + 1)

            total_value = sum(
                expert_urgency.get(k, 0) * expert_demand_count.get(k, 0)
                for k in miss_list
            )

            def wcov(selected: set) -> float:
                return sum(
                    expert_urgency.get(k, 0) * expert_demand_count.get(k, 0)
                    for k in selected
                ) / max(1e-9, total_value)

            # FIFO (interleaved token-first order)
            fifo_order = []
            tps = sorted(set(tp for ent in entries for tp in ent["token_expert_map"]))
            fifo_seen = set()
            for tp in tps:
                for ent in entries:
                    lid = ent["layer_id"]
                    if tp in ent["token_expert_map"]:
                        for e in ent["token_expert_map"][tp]:
                            key = (lid, e)
                            if key in miss_set and key not in fifo_seen:
                                fifo_order.append(key)
                                fifo_seen.add(key)
            fifo_coverages.append(wcov(set(fifo_order[:budget])))

            # Random
            random_sel = set(rng.sample(miss_list, min(budget, len(miss_list))))
            random_coverages.append(wcov(random_sel))

            # Oracle
            oracle_order = sorted(
                miss_list,
                key=lambda k: expert_urgency.get(k, 0) * expert_demand_count.get(k, 0),
                reverse=True,
            )
            oracle_coverages.append(wcov(set(oracle_order[:budget])))

        # Step LRU forward
        for entry in entries:
            layer_id = entry["layer_id"]
            all_experts = set()
            for _, experts in entry["token_expert_map"].items():
                all_experts.update(experts)
            for e in all_experts:
                if e in lru_set[layer_id]:
                    lru[layer_id].remove(e)
                    lru[layer_id].append(e)
                else:
                    if len(lru[layer_id]) >= cache_size:
                        victim = lru[layer_id].pop(0)
                        lru_set[layer_id].discard(victim)
                    lru[layer_id].append(e)
                    lru_set[layer_id].add(e)

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    warmup = min(10, max_step // 5)
    demand_skip = per_step_demand[warmup:]

    ax1.hist(demand_skip, bins=25, color="#FF8A65", edgecolor="white", lw=0.5,
             alpha=0.85, label="Per-step demand")
    ax1.axvline(x=budget, color="#1565C0", ls="--", lw=2,
                label=f"PCIe budget ({budget})")
    ax1.set_xlabel("Miss Expert Count per SD Step")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Demand vs. Budget", pad=6)
    ax1.legend(fontsize=8)

    avg_demand = np.mean(demand_skip) if demand_skip else 0
    pct_over = sum(1 for d in demand_skip if d > budget) / max(1, len(demand_skip)) * 100
    ax1.text(0.95, 0.85,
             f"Avg: {avg_demand:.0f} experts\n"
             f"Budget: {budget}\n"
             f"Ratio: {avg_demand / max(1, budget):.1f}×\n"
             f"{pct_over:.0f}% steps over budget",
             transform=ax1.transAxes, ha="right", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFECB3", ec="#FF8F00", lw=0.8))

    # Right: FIFO / Random / Oracle
    strategies = ["FIFO", "Random", "Oracle"]
    skip = min(5, len(fifo_coverages) // 3)
    avg_covs = [
        np.mean(fifo_coverages[skip:])   * 100 if fifo_coverages else 0,
        np.mean(random_coverages[skip:]) * 100 if random_coverages else 0,
        np.mean(oracle_coverages[skip:]) * 100 if oracle_coverages else 0,
    ]
    colors = ["#FFA726", "#BDBDBD", "#66BB6A"]
    bars = ax2.bar(strategies, avg_covs, color=colors, width=0.5,
                   edgecolor="black", lw=0.7)
    ax2.set_ylabel("Urgency-Weighted\nValue Coverage (%)")
    ax2.set_title("Scheduling Strategy", pad=6)
    ax2.set_ylim(0, max(avg_covs) * 1.35 if avg_covs and max(avg_covs) > 0 else 100)
    for bar, val in zip(bars, avg_covs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    gap = avg_covs[2] - avg_covs[0]
    ax2.text(0.5, 0.90,
             f"FIFO → Oracle: +{gap:.0f}pp\n→ Priority scheduling matters",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color="#1B5E20",
             bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#4CAF50", lw=0.8))

    fig.suptitle("Fig. 5: Lookahead oversubscribes transfer budget",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_obs3_budget.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_obs3_budget.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig 5] demand={avg_demand:.0f} | budget={budget} | "
          f"ratio={avg_demand/max(1,budget):.1f}× | {pct_over:.0f}% over budget")
    print(f"[Fig 5] FIFO={avg_covs[0]:.1f}% Random={avg_covs[1]:.1f}% "
          f"Oracle={avg_covs[2]:.1f}%")


# ====================================================================
# Main
# ====================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BriskMoE §2 Motivation v3 — Chronic Overload Story (Real Data)")
    print(f"  Trace: {TRACE_PATH}")
    print(f"  Model: Qwen3-30B-A3B, {NUM_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K_EXPERTS}")
    print(f"  SD: K={DRAFT_K}, α={ACCEPT_RATE}")
    print("=" * 70)

    # ── Load trace ──
    print("\n── Loading real trace ──")
    token_data, all_tokens = load_trace()
    sd_trace = build_sd_trace(token_data, all_tokens)

    max_step = max(e["step"] for e in sd_trace)

    # Auto cache size: ~70% of mean working set (tight, realistic)
    sample_ws = []
    for s in range(min(20, max_step)):
        for l in range(NUM_LAYERS):
            experts = set()
            for entry in sd_trace:
                if entry["step"] == s and entry["layer_id"] == l:
                    for _, ex in entry["token_expert_map"].items():
                        experts.update(ex)
            if experts:
                sample_ws.append(len(experts))
    avg_ws = int(np.mean(sample_ws)) if sample_ws else 20
    cache_size = max(8, int(avg_ws * 0.7))
    print(f"  Avg working set = {avg_ws}, cache_size S = {cache_size}")

    # ── Fig 2 ──
    print("\n── Fig 2: Naive SD helps, but chronic overload ──")
    plot_fig2()

    # ── Fig 3 ──
    print("\n── Fig 3: Obs 1 — Chronic overload ──")
    plot_fig3(token_data, all_tokens, cache_size)

    # ── Fig 4 ──
    print("\n── Fig 4: Obs 2 — LRU residency instability ──")
    plot_fig4(sd_trace, cache_size)

    # ── Fig 5 ──
    print("\n── Fig 5: Obs 3 — Bandwidth oversubscription ──")
    plot_fig5(sd_trace, cache_size)

    # ── Summary ──
    summary = {
        "data_source": "REAL — Qwen3-30B-A3B on HumanEval (10 prompts, 1953 tokens)",
        "trace_file": str(TRACE_PATH),
        "num_layers": NUM_LAYERS,
        "num_sd_steps": max_step + 1,
        "cache_size": cache_size,
        "avg_working_set": avg_ws,
    }
    with open(OUT_DIR / "motivation_v3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Summary → {OUT_DIR / 'motivation_v3_summary.json'} ──")
    print("\n" + "=" * 70)
    print("ALL v3 MOTIVATION EXPERIMENTS COMPLETE")
    print(f"Figures: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
