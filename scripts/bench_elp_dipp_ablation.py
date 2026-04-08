#!/usr/bin/env python3
"""
BriskMoE ELP+DIPP Ablation Benchmark on Real HumanEval Trace
==============================================================
Uses the collected real expert routing trace (93,264 events, 1,953 tokens)
from Qwen3-30B-A3B on HumanEval to validate:

  1. LRU baseline (SD, no optimization)
  2. SACR only (cache replacement, no partitioning)
  3. ELP only (Pin/Flex partition, LRU eviction)
  4. DIPP only (priority prefetch, LRU eviction)
  5. SACR + ELP
  6. SACR + DIPP
  7. ELP + DIPP
  8. Full BriskMoE (SACR + ELP + DIPP)

Metrics:
  - Cache hit rate (η)
  - Estimated throughput (tok/s) via Bandwidth Regime Model
  - Prefetch coverage (DIPP configs only)
  - Rejected-expert cache residence fraction
  - Cascade eviction length

Hardware assumptions:
  - PCIe Gen4 x16: 25 GB/s
  - HBM bandwidth: 768 GB/s
  - Expert size: 9.44 MB
  - Draft latency: ~30 ms → PCIe budget = 79 experts
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.accept_reject_tracker import AcceptRejectTracker, AcceptRejectTrackerConfig
from adapters.sacr import SACREvictionPolicy, SACRConfig
from adapters.elp import ExpertLifecyclePartition, ELPConfig
from adapters.dipp import DraftInformedPrioritizedPreloader, DIPPConfig
from adapters.briskmoe_cache import BriskMoECache, BriskMoECacheConfig

# ── Constants ──
TRACE_PATH = ROOT / "results" / "real_trace" / "expert_trace_humaneval.jsonl"
RESULT_DIR = ROOT / "results" / "elp_dipp_ablation"

# Hardware parameters (A6000 + PCIe Gen4 x16)
EXPERT_SIZE_BYTES = 9_437_184   # ~9.44 MB
PCIE_BW = 25e9                  # 25 GB/s
HBM_BW = 768e9                  # 768 GB/s (A6000)
DRAFT_LATENCY_S = 0.030         # 30 ms
PCIE_BUDGET = int(DRAFT_LATENCY_S * PCIE_BW / EXPERT_SIZE_BYTES)  # ~79

# SD parameters
DRAFT_K = 3
ACCEPT_RATE = 0.625
SEED = 42

# Cache sizes to sweep (in number of expert slots per layer)
CACHE_SIZES = [10, 14, 17, 21, 25]


# ══════════════════════════════════════════════════════════════════
# Phase 1: Load and convert real trace
# ══════════════════════════════════════════════════════════════════

def load_real_trace(trace_path: Path) -> list[dict]:
    """Load JSONL trace → SD step format (same as motivation_v3_real.py)."""
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Group by (request_id, token_idx)
    token_data: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(dict)
    for ev in events:
        key = (ev["request_id"], ev["token_idx"])
        token_data[key][ev["layer_id"]] = ev["experts"]

    all_tokens = sorted(token_data.keys(), key=lambda x: (x[0], x[1]))

    # Group by request_id
    req_tokens: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in all_tokens:
        req_tokens[key[0]].append(key)

    # Convert to SD steps
    rng = random.Random(SEED)
    sd_trace = []
    global_step = 0
    step_size = DRAFT_K + 1

    for req_id, tokens in sorted(req_tokens.items()):
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

            layer_ids = sorted(token_data[chunk[0]].keys())

            for layer_id in layer_ids:
                token_expert_map = {}
                for tok_pos, tok_key in enumerate(chunk):
                    experts = token_data[tok_key].get(layer_id, [])
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

    return sd_trace


# ══════════════════════════════════════════════════════════════════
# Phase 2: Simulation engines
# ══════════════════════════════════════════════════════════════════

def simulate_lru(trace: list[dict], cache_size: int) -> dict:
    """Pure LRU baseline."""
    lru: dict[int, list[int]] = defaultdict(list)
    lru_set: dict[int, set[int]] = defaultdict(set)

    hits = misses = 0
    expert_accepted: dict[tuple[int, int], int] = defaultdict(int)
    expert_total: dict[tuple[int, int], int] = defaultdict(int)
    miss_streak: dict[int, int] = defaultdict(int)
    cascade_lengths: list[int] = []

    for entry in trace:
        layer_id = entry["layer_id"]
        accepted_mask = entry["accepted_mask"]
        token_expert_map = entry["token_expert_map"]

        all_experts = set()
        for tok_pos, experts in token_expert_map.items():
            is_acc = accepted_mask[int(tok_pos)] if int(tok_pos) < len(accepted_mask) else False
            for e in experts:
                all_experts.add(e)
                expert_total[(layer_id, e)] += 1
                if is_acc:
                    expert_accepted[(layer_id, e)] += 1

        step_misses = 0
        for e in all_experts:
            if e in lru_set[layer_id]:
                hits += 1
                lru[layer_id].remove(e)
                lru[layer_id].append(e)
            else:
                misses += 1
                step_misses += 1
                if len(lru[layer_id]) >= cache_size:
                    victim = lru[layer_id].pop(0)
                    lru_set[layer_id].discard(victim)
                lru[layer_id].append(e)
                lru_set[layer_id].add(e)

        if step_misses > max(2, cache_size * 0.15):
            miss_streak[layer_id] += 1
        else:
            if miss_streak[layer_id] > 1:
                cascade_lengths.append(miss_streak[layer_id])
            miss_streak[layer_id] = 0

    # Rejected expert residence
    rej_res = []
    for layer_id in lru_set:
        low_ar = sum(1 for e in lru_set[layer_id]
                     if expert_accepted.get((layer_id, e), 0) / max(1, expert_total.get((layer_id, e), 0)) < 0.3)
        rej_res.append(low_ar / max(1, len(lru_set[layer_id])))

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(1, total),
        "rejected_residence": sum(rej_res) / max(1, len(rej_res)),
        "avg_cascade": sum(cascade_lengths) / max(1, len(cascade_lengths)) if cascade_lengths else 0,
        "num_cascades": len(cascade_lengths),
    }


def simulate_briskmoe(
    trace: list[dict],
    cache_size: int,
    enable_sacr: bool = True,
    enable_elp: bool = True,
    enable_dipp: bool = False,
) -> dict:
    """Simulate BriskMoE cache with configurable components."""
    config = BriskMoECacheConfig(
        total_slots_per_layer=cache_size,
        sacr=SACRConfig(
            alpha=0.3 if enable_sacr else 0.6,
            beta=0.2 if enable_sacr else 0.4,
            gamma=0.5 if enable_sacr else 0.0,
        ),
        elp=ELPConfig(
            pin_ratio=0.7 if enable_elp else 0.0,
            promotion_threshold=5 if enable_elp else 999999,
        ),
        dipp=DIPPConfig(
            max_prefetch_experts=PCIE_BUDGET if enable_dipp else 0,
        ),
        tracker=AcceptRejectTrackerConfig(ema_alpha=0.15, min_observations=3),
        rebalance_interval=10,
    )
    cache = BriskMoECache(config)

    hits = misses = 0
    expert_accepted: dict[tuple[int, int], int] = defaultdict(int)
    expert_total: dict[tuple[int, int], int] = defaultdict(int)
    miss_streak: dict[int, int] = defaultdict(int)
    cascade_lengths: list[int] = []

    # DIPP stats
    dipp_prefetch_hits = 0
    dipp_total_scheduled = 0

    # Group trace by step for DIPP
    steps_data: dict[int, list[dict]] = defaultdict(list)
    for entry in trace:
        steps_data[entry["step"]].append(entry)

    max_step = max(e["step"] for e in trace)

    for step in range(max_step + 1):
        entries = steps_data.get(step, [])
        if not entries:
            continue

        # --- DIPP: compute prefetch schedule BEFORE verify ---
        prefetched: set[tuple[int, int]] = set()
        if enable_dipp:
            cache.begin_draft_round()
            # Build predictions from current step's token-expert map
            predictions: dict[int, dict[int, list[int]]] = {}
            for entry in entries:
                lid = entry["layer_id"]
                predictions[lid] = {}
                for tok_pos, experts in entry["token_expert_map"].items():
                    predictions[lid][int(tok_pos)] = experts

            schedule = cache.compute_full_prefetch_schedule(predictions)
            dipp_total_scheduled += len(schedule)

            # Simulate prefetching: add to cache if not already there
            for lid, eid, val in schedule:
                if eid not in cache.get_cache_state(lid):
                    # Prefetch: insert into cache (may evict)
                    is_hit, victim = cache.access_expert(lid, eid, step=step)
                    if not is_hit:
                        prefetched.add((lid, eid))

        # --- Verify phase: access all experts ---
        for entry in entries:
            layer_id = entry["layer_id"]
            accepted_mask = entry["accepted_mask"]
            token_expert_map = entry["token_expert_map"]

            all_experts = set()
            for tok_pos, experts in token_expert_map.items():
                is_acc = accepted_mask[int(tok_pos)] if int(tok_pos) < len(accepted_mask) else False
                for e in experts:
                    all_experts.add(e)
                    expert_total[(layer_id, e)] += 1
                    if is_acc:
                        expert_accepted[(layer_id, e)] += 1

            step_misses = 0
            for e in all_experts:
                is_hit, victim = cache.access_expert(layer_id, e, step=step)
                if is_hit:
                    hits += 1
                    if (layer_id, e) in prefetched:
                        dipp_prefetch_hits += 1
                else:
                    misses += 1
                    step_misses += 1

            # Update tracker
            cache.on_verify_complete(
                layer_id=layer_id,
                token_expert_map=token_expert_map,
                accepted_mask=accepted_mask,
                step=step,
            )

            if step_misses > max(2, cache_size * 0.15):
                miss_streak[layer_id] += 1
            else:
                if miss_streak[layer_id] > 1:
                    cascade_lengths.append(miss_streak[layer_id])
                miss_streak[layer_id] = 0

    # Rejected expert residence
    num_layers = max(e["layer_id"] for e in trace) + 1
    rej_res = []
    for layer_id in range(num_layers):
        cached = cache.get_cache_state(layer_id)
        if not cached:
            continue
        low_ar = sum(1 for e in cached
                     if expert_accepted.get((layer_id, e), 0) / max(1, expert_total.get((layer_id, e), 0)) < 0.3)
        rej_res.append(low_ar / max(1, len(cached)))

    total = hits + misses
    result = {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(1, total),
        "rejected_residence": sum(rej_res) / max(1, len(rej_res)),
        "avg_cascade": sum(cascade_lengths) / max(1, len(cascade_lengths)) if cascade_lengths else 0,
        "num_cascades": len(cascade_lengths),
    }
    if enable_dipp:
        result["dipp_prefetch_hits"] = dipp_prefetch_hits
        result["dipp_total_scheduled"] = dipp_total_scheduled
        result["dipp_prefetch_accuracy"] = dipp_prefetch_hits / max(1, dipp_total_scheduled)

    return result


# ══════════════════════════════════════════════════════════════════
# Phase 3: Throughput estimation via Bandwidth Regime Model
# ══════════════════════════════════════════════════════════════════

def estimate_throughput(hit_rate: float, num_verify_tokens: float = 2.5) -> dict:
    """
    Bandwidth Regime Model:
      B_eff(η) = η · B_HBM + (1-η) · B_PCIe
      TPOT = expert_bytes × top_k / B_eff
      throughput = num_verify_tokens / TPOT  (for SD)
    """
    top_k = 8
    total_bytes_per_token = EXPERT_SIZE_BYTES * top_k  # single layer

    b_eff = hit_rate * HBM_BW + (1 - hit_rate) * PCIE_BW
    tpot_per_layer = total_bytes_per_token / b_eff
    num_layers = 48
    tpot_total = tpot_per_layer * num_layers

    # For SD: effective throughput = accepted_tokens / total_time_per_step
    # total_time_per_step = draft_time + verify_time
    verify_time = tpot_total
    draft_time = DRAFT_LATENCY_S
    total_step_time = draft_time + verify_time

    # Average accepted tokens per step = K * accept_rate + 1 (bonus)
    avg_accepted = DRAFT_K * ACCEPT_RATE + 1  # ~2.875

    throughput_sd = avg_accepted / total_step_time

    # AR throughput (no draft overhead, 1 token per step, same cache behavior)
    throughput_ar = 1.0 / verify_time if verify_time > 0 else 0

    return {
        "b_eff_gbps": b_eff / 1e9,
        "tpot_ms": tpot_total * 1000,
        "throughput_sd": throughput_sd,
        "throughput_ar": throughput_ar,
        "sd_over_ar": throughput_sd / max(0.001, throughput_ar),
    }


# ══════════════════════════════════════════════════════════════════
# Phase 4: Main benchmark
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    label: str
    sacr: bool
    elp: bool
    dipp: bool


CONFIGS = [
    Config("LRU (Base)",           sacr=False, elp=False, dipp=False),
    Config("SACR only",            sacr=True,  elp=False, dipp=False),
    Config("ELP only",             sacr=False, elp=True,  dipp=False),
    Config("DIPP only",            sacr=False, elp=False, dipp=True),
    Config("SACR+ELP",             sacr=True,  elp=True,  dipp=False),
    Config("SACR+DIPP",            sacr=True,  elp=False, dipp=True),
    Config("ELP+DIPP",             sacr=False, elp=True,  dipp=True),
    Config("Full (SACR+ELP+DIPP)", sacr=True,  elp=True,  dipp=True),
]


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("BriskMoE ELP+DIPP Ablation Benchmark")
    print(f"  Trace: {TRACE_PATH}")
    print(f"  Hardware: A6000 48GB, PCIe Gen4 x16 ({PCIE_BW/1e9:.0f} GB/s)")
    print(f"  Expert size: {EXPERT_SIZE_BYTES/1e6:.2f} MB")
    print(f"  SD params: K={DRAFT_K}, α={ACCEPT_RATE}")
    print(f"  PCIe budget: {PCIE_BUDGET} experts/step")
    print(f"  Cache sizes: {CACHE_SIZES}")
    print("=" * 90)

    # Load trace
    print("\n[1/3] Loading real trace...")
    t0 = time.time()
    sd_trace = load_real_trace(TRACE_PATH)
    num_layers = max(e["layer_id"] for e in sd_trace) + 1
    max_step = max(e["step"] for e in sd_trace)
    print(f"  Loaded: {len(sd_trace)} entries, {max_step+1} steps, {num_layers} layers")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Run ablation across cache sizes
    print("\n[2/3] Running ablation...")
    all_results = {}

    for cache_size in CACHE_SIZES:
        print(f"\n{'─'*80}")
        print(f"  Cache Size = {cache_size} slots/layer "
              f"({cache_size * EXPERT_SIZE_BYTES / 1e9:.2f} GB for {num_layers}L)")
        print(f"{'─'*80}")

        results_for_size = {}

        for cfg in CONFIGS:
            t1 = time.time()
            if not cfg.sacr and not cfg.elp and not cfg.dipp:
                stats = simulate_lru(sd_trace, cache_size)
            else:
                stats = simulate_briskmoe(
                    sd_trace, cache_size,
                    enable_sacr=cfg.sacr,
                    enable_elp=cfg.elp,
                    enable_dipp=cfg.dipp,
                )
            elapsed = time.time() - t1

            tp = estimate_throughput(stats["hit_rate"])

            print(f"  {cfg.label:<25} η={stats['hit_rate']:.4f}  "
                  f"rej_res={stats['rejected_residence']:.3f}  "
                  f"cascade={stats['avg_cascade']:.1f}  "
                  f"est_SD={tp['throughput_sd']:.2f} tok/s  "
                  f"({elapsed:.1f}s)")

            results_for_size[cfg.label] = {
                **stats,
                **tp,
                "cache_size": cache_size,
                "config": cfg.label,
            }

        all_results[cache_size] = results_for_size

    # ── Summary table ──
    print("\n\n" + "=" * 100)
    print("ABLATION SUMMARY — Real Qwen3-30B-A3B HumanEval Trace")
    print("=" * 100)

    # Header
    print(f"\n{'Config':<25}", end="")
    for cs in CACHE_SIZES:
        print(f"  S={cs:>2} η     tps", end="")
    print()
    print("-" * (25 + len(CACHE_SIZES) * 18))

    for cfg in CONFIGS:
        print(f"{cfg.label:<25}", end="")
        for cs in CACHE_SIZES:
            r = all_results[cs][cfg.label]
            print(f"  {r['hit_rate']:.4f}  {r['throughput_sd']:5.2f}", end="")
        print()

    # ── Improvement over LRU ──
    print(f"\n{'─'*100}")
    print("IMPROVEMENT OVER LRU BASELINE (hit rate delta, throughput ratio)")
    print(f"{'─'*100}")
    print(f"{'Config':<25}", end="")
    for cs in CACHE_SIZES:
        print(f"  S={cs:>2} Δη      ×", end="")
    print()
    print("-" * (25 + len(CACHE_SIZES) * 18))

    for cfg in CONFIGS:
        if cfg.label == "LRU (Base)":
            continue
        print(f"{cfg.label:<25}", end="")
        for cs in CACHE_SIZES:
            lru_r = all_results[cs]["LRU (Base)"]
            r = all_results[cs][cfg.label]
            delta_eta = r["hit_rate"] - lru_r["hit_rate"]
            ratio = r["throughput_sd"] / max(0.01, lru_r["throughput_sd"])
            print(f"  {delta_eta:+.4f}  {ratio:.2f}×", end="")
        print()

    # ── Key insight: ELP+DIPP contribution ──
    print(f"\n{'═'*100}")
    print("KEY RESULTS (S=14, the operating point)")
    print(f"{'═'*100}")
    if 14 in all_results:
        r14 = all_results[14]
        lru = r14["LRU (Base)"]
        full = r14["Full (SACR+ELP+DIPP)"]
        elp_only = r14["ELP only"]
        dipp_only = r14["DIPP only"]
        sacr_only = r14["SACR only"]
        sacr_elp = r14["SACR+ELP"]
        elp_dipp = r14["ELP+DIPP"]

        print(f"  LRU baseline:          η={lru['hit_rate']:.4f}  → {lru['throughput_sd']:.2f} tok/s")
        print(f"  SACR only:             η={sacr_only['hit_rate']:.4f}  → {sacr_only['throughput_sd']:.2f} tok/s  "
              f"(+{sacr_only['hit_rate']-lru['hit_rate']:.4f})")
        print(f"  ELP only:              η={elp_only['hit_rate']:.4f}  → {elp_only['throughput_sd']:.2f} tok/s  "
              f"(+{elp_only['hit_rate']-lru['hit_rate']:.4f})")
        print(f"  DIPP only:             η={dipp_only['hit_rate']:.4f}  → {dipp_only['throughput_sd']:.2f} tok/s  "
              f"(+{dipp_only['hit_rate']-lru['hit_rate']:.4f})")
        print(f"  SACR+ELP:              η={sacr_elp['hit_rate']:.4f}  → {sacr_elp['throughput_sd']:.2f} tok/s  "
              f"(+{sacr_elp['hit_rate']-lru['hit_rate']:.4f})")
        print(f"  ELP+DIPP:              η={elp_dipp['hit_rate']:.4f}  → {elp_dipp['throughput_sd']:.2f} tok/s  "
              f"(+{elp_dipp['hit_rate']-lru['hit_rate']:.4f})")
        print(f"  Full (SACR+ELP+DIPP):  η={full['hit_rate']:.4f}  → {full['throughput_sd']:.2f} tok/s  "
              f"(+{full['hit_rate']-lru['hit_rate']:.4f})")
        print(f"\n  SD/AR speedup (Full): {full['sd_over_ar']:.2f}×")
        print(f"  Full / LRU throughput: {full['throughput_sd']/max(0.01,lru['throughput_sd']):.2f}×")

    # ── Save results ──
    summary = {
        "experiment": "elp_dipp_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trace": str(TRACE_PATH),
        "hardware": {
            "gpu": "NVIDIA RTX A6000 48GB",
            "pcie_bw_gbps": PCIE_BW / 1e9,
            "hbm_bw_gbps": HBM_BW / 1e9,
        },
        "sd_params": {"K": DRAFT_K, "accept_rate": ACCEPT_RATE},
        "cache_sizes": CACHE_SIZES,
        "configs": [c.label for c in CONFIGS],
        "results": {str(k): v for k, v in all_results.items()},
    }

    out_path = RESULT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
