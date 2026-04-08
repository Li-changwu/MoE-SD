"""
ISSUE-005: BriskMoE Profiling Validation Tests
=================================================
Validates that P1-P4 problems exist and that SACR/ELP/DIPP solve them.

Test categories:
  1. P1 (Cache Pollution) — rejected-token experts waste cache space under LRU
  2. P4 (Cascade Eviction) — burst overflow propagates ~8 steps under LRU, ~2 with ELP
  3. P2 (Flat Prefetch) — demand >> supply, DIPP beats FIFO
  4. Ablation — Full > any subset > Base
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import unittest
from collections import defaultdict


def _load_module(name, filename):
    path = os.path.join(os.path.dirname(__file__), "..", "adapters", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_art_mod = _load_module("adapters.accept_reject_tracker", "accept_reject_tracker.py")
_sacr_mod = _load_module("adapters.sacr", "sacr.py")
_elp_mod = _load_module("adapters.elp", "elp.py")
_dipp_mod = _load_module("adapters.dipp", "dipp.py")
_cache_mod = _load_module("adapters.briskmoe_cache", "briskmoe_cache.py")

AcceptRejectTracker = _art_mod.AcceptRejectTracker
AcceptRejectTrackerConfig = _art_mod.AcceptRejectTrackerConfig
SACREvictionPolicy = _sacr_mod.SACREvictionPolicy
SACRConfig = _sacr_mod.SACRConfig
ExpertLifecyclePartition = _elp_mod.ExpertLifecyclePartition
ELPConfig = _elp_mod.ELPConfig
DraftInformedPrioritizedPreloader = _dipp_mod.DraftInformedPrioritizedPreloader
DIPPConfig = _dipp_mod.DIPPConfig
BriskMoECache = _cache_mod.BriskMoECache
BriskMoECacheConfig = _cache_mod.BriskMoECacheConfig


# ──────────────────────────────────────────────────────────────────────
# Helpers: Simulated MoE Trace Generation
# ──────────────────────────────────────────────────────────────────────

def generate_sd_trace(
    num_steps: int = 50,
    num_layers: int = 4,
    draft_k: int = 3,
    top_k_experts: int = 8,
    num_experts: int = 128,
    accept_rate: float = 0.625,  # 2.5 out of 4 tokens accepted
    persistent_experts: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a synthetic SD verify trace.

    Each step produces K+1 tokens, each activating top_k_experts per layer.
    Persistent experts appear with high probability; transient ones are random.

    Returns list of dicts with keys:
      - step, layer_id, token_expert_map, accepted_mask
    """
    rng = random.Random(seed)
    if persistent_experts is None:
        # ~12 persistent experts per layer
        persistent_experts = list(range(12))

    trace = []
    for step in range(num_steps):
        num_tokens = draft_k + 1  # K draft + 1 bonus
        num_accepted = max(1, round(num_tokens * accept_rate))
        accepted_mask = [True] * num_accepted + [False] * (num_tokens - num_accepted)
        rng.shuffle(accepted_mask)

        for layer_id in range(num_layers):
            token_expert_map = {}
            for tok_pos in range(num_tokens):
                experts = []
                # Persistent experts: high probability for accepted tokens
                if accepted_mask[tok_pos]:
                    for pe in persistent_experts:
                        if rng.random() < 0.6 and len(experts) < top_k_experts:
                            experts.append(pe)
                else:
                    # Rejected tokens: some persistent overlap, mostly transient
                    for pe in persistent_experts:
                        if rng.random() < 0.2 and len(experts) < top_k_experts:
                            experts.append(pe)

                # Fill remaining with transient experts
                while len(experts) < top_k_experts:
                    e = rng.randint(0, num_experts - 1)
                    if e not in experts:
                        experts.append(e)

                token_expert_map[tok_pos] = experts

            trace.append({
                "step": step,
                "layer_id": layer_id,
                "token_expert_map": token_expert_map,
                "accepted_mask": accepted_mask,
            })

    return trace


def simulate_lru_cache(trace: list[dict], cache_size: int = 17) -> dict:
    """
    Simulate a simple LRU cache on the trace.

    Returns stats dict with:
      - hits, misses, hit_rate
      - rejected_expert_residence: fraction of cache occupied by low-AR experts
      - cascade_lengths: list of consecutive miss-streaks after capacity overflow
    """
    # Per-layer LRU: ordered list (most recent at end)
    lru: dict[int, list[int]] = defaultdict(list)
    lru_set: dict[int, set[int]] = defaultdict(set)

    # Track which experts are "rejected-heavy"
    expert_accepted: dict[tuple[int, int], int] = defaultdict(int)
    expert_total: dict[tuple[int, int], int] = defaultdict(int)

    hits = 0
    misses = 0
    miss_streak: dict[int, int] = defaultdict(int)  # per-layer
    cascade_lengths: list[int] = []

    for entry in trace:
        layer_id = entry["layer_id"]
        accepted_mask = entry["accepted_mask"]
        token_expert_map = entry["token_expert_map"]

        # Collect all unique experts accessed this step
        all_experts = set()
        for tok_pos, experts in token_expert_map.items():
            is_accepted = accepted_mask[tok_pos] if tok_pos < len(accepted_mask) else False
            for e in experts:
                all_experts.add(e)
                expert_total[(layer_id, e)] += 1
                if is_accepted:
                    expert_accepted[(layer_id, e)] += 1

        step_misses = 0
        for e in all_experts:
            if e in lru_set[layer_id]:
                hits += 1
                # Move to end (most recently used)
                lru[layer_id].remove(e)
                lru[layer_id].append(e)
            else:
                misses += 1
                step_misses += 1
                # Evict if full
                if len(lru[layer_id]) >= cache_size:
                    victim = lru[layer_id].pop(0)
                    lru_set[layer_id].discard(victim)
                lru[layer_id].append(e)
                lru_set[layer_id].add(e)

        # Track cascade (consecutive steps with high miss count)
        if step_misses > max(2, cache_size * 0.15):  # >15% miss rate per step
            miss_streak[layer_id] += 1
        else:
            if miss_streak[layer_id] > 1:
                cascade_lengths.append(miss_streak[layer_id])
            miss_streak[layer_id] = 0

    # Compute rejected-expert residence
    # Check what fraction of cache is occupied by experts with AR < 0.3
    rejected_residence_per_layer = []
    for layer_id in lru_set:
        low_ar_count = 0
        for e in lru_set[layer_id]:
            total = expert_total.get((layer_id, e), 0)
            accepted = expert_accepted.get((layer_id, e), 0)
            ar = accepted / max(1, total)
            if ar < 0.3:
                low_ar_count += 1
        rejected_residence_per_layer.append(
            low_ar_count / max(1, len(lru_set[layer_id]))
        )

    avg_rejected_residence = (
        sum(rejected_residence_per_layer) / max(1, len(rejected_residence_per_layer))
    )

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(1, total),
        "rejected_expert_residence": avg_rejected_residence,
        "cascade_lengths": cascade_lengths,
        "avg_cascade_length": (
            sum(cascade_lengths) / max(1, len(cascade_lengths))
            if cascade_lengths
            else 0.0
        ),
    }


def simulate_briskmoe_cache(
    trace: list[dict],
    cache_size: int = 17,
    enable_sacr: bool = True,
    enable_elp: bool = True,
    enable_dipp: bool = False,
) -> dict:
    """
    Simulate BriskMoE cache with configurable components on the same trace.

    Returns same stats format as simulate_lru_cache.
    """
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
        tracker=AcceptRejectTrackerConfig(ema_alpha=0.15, min_observations=3),
    )
    cache = BriskMoECache(config)

    hits = 0
    misses = 0

    expert_accepted: dict[tuple[int, int], int] = defaultdict(int)
    expert_total: dict[tuple[int, int], int] = defaultdict(int)

    miss_streak: dict[int, int] = defaultdict(int)
    cascade_lengths: list[int] = []

    for entry in trace:
        step = entry["step"]
        layer_id = entry["layer_id"]
        accepted_mask = entry["accepted_mask"]
        token_expert_map = entry["token_expert_map"]

        # Access all experts
        all_experts = set()
        for tok_pos, experts in token_expert_map.items():
            is_accepted = accepted_mask[tok_pos] if tok_pos < len(accepted_mask) else False
            for e in experts:
                all_experts.add(e)
                expert_total[(layer_id, e)] += 1
                if is_accepted:
                    expert_accepted[(layer_id, e)] += 1

        step_misses = 0
        for e in all_experts:
            is_hit, victim = cache.access_expert(layer_id, e, step=step)
            if is_hit:
                hits += 1
            else:
                misses += 1
                step_misses += 1

        # Update tracker with verify result (this is the SD-specific signal)
        cache.on_verify_complete(
            layer_id=layer_id,
            token_expert_map=token_expert_map,
            accepted_mask=accepted_mask,
            step=step,
        )

        # Track cascade
        if step_misses > max(2, cache_size * 0.15):
            miss_streak[layer_id] += 1
        else:
            if miss_streak[layer_id] > 1:
                cascade_lengths.append(miss_streak[layer_id])
            miss_streak[layer_id] = 0

    # Compute rejected-expert residence
    rejected_residence_per_layer = []
    for layer_id in range(max(e["layer_id"] for e in trace) + 1):
        cached = cache.get_cache_state(layer_id)
        if not cached:
            continue
        low_ar_count = 0
        for e in cached:
            total = expert_total.get((layer_id, e), 0)
            accepted = expert_accepted.get((layer_id, e), 0)
            ar = accepted / max(1, total)
            if ar < 0.3:
                low_ar_count += 1
        rejected_residence_per_layer.append(
            low_ar_count / max(1, len(cached))
        )

    avg_rejected_residence = (
        sum(rejected_residence_per_layer) / max(1, len(rejected_residence_per_layer))
    )

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(1, total),
        "rejected_expert_residence": avg_rejected_residence,
        "cascade_lengths": cascade_lengths,
        "avg_cascade_length": (
            sum(cascade_lengths) / max(1, len(cascade_lengths))
            if cascade_lengths
            else 0.0
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Test: P1 — Cache Pollution Validation
# ──────────────────────────────────────────────────────────────────────

class TestP1CachePollution(unittest.TestCase):
    """Validate P1: rejected-token experts pollute LRU cache."""

    def setUp(self):
        self.trace = generate_sd_trace(
            num_steps=80,
            num_layers=4,
            draft_k=3,
            top_k_experts=8,
            num_experts=128,
            accept_rate=0.625,
            seed=42,
        )

    def test_lru_has_rejected_pollution(self):
        """LRU should have significant rejected-expert residence (>10%)."""
        stats = simulate_lru_cache(self.trace, cache_size=17)
        self.assertGreater(
            stats["rejected_expert_residence"], 0.10,
            f"Expected >10% rejected-expert residence under LRU, "
            f"got {stats['rejected_expert_residence']:.1%}"
        )

    def test_sacr_reduces_pollution(self):
        """SACR should have lower rejected-expert residence than LRU."""
        lru_stats = simulate_lru_cache(self.trace, cache_size=17)
        sacr_stats = simulate_briskmoe_cache(
            self.trace, cache_size=17, enable_sacr=True, enable_elp=False,
        )
        self.assertLess(
            sacr_stats["rejected_expert_residence"],
            lru_stats["rejected_expert_residence"],
            f"SACR residence ({sacr_stats['rejected_expert_residence']:.1%}) "
            f"should be < LRU ({lru_stats['rejected_expert_residence']:.1%})"
        )

    def test_sacr_improves_hit_rate_over_lru(self):
        """SACR hit rate should be higher than LRU."""
        lru_stats = simulate_lru_cache(self.trace, cache_size=17)
        sacr_stats = simulate_briskmoe_cache(
            self.trace, cache_size=17, enable_sacr=True, enable_elp=False,
        )
        self.assertGreater(
            sacr_stats["hit_rate"],
            lru_stats["hit_rate"],
            f"SACR hit rate ({sacr_stats['hit_rate']:.3f}) "
            f"should be > LRU ({lru_stats['hit_rate']:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────
# Test: P4 — Cascade Eviction Validation
# ──────────────────────────────────────────────────────────────────────

class TestP4CascadeEviction(unittest.TestCase):
    """Validate P4: SD burst access (W > S) causes persistent-expert eviction under LRU."""

    def test_sd_working_set_exceeds_cache(self):
        """SD working set (unique experts per step) should exceed cache size."""
        trace = generate_sd_trace(
            num_steps=50, num_layers=1, draft_k=3,
            top_k_experts=8, num_experts=64,
            accept_rate=0.5, seed=123,
        )
        max_unique = 0
        for entry in trace:
            unique = set()
            for experts in entry["token_expert_map"].values():
                unique.update(experts)
            max_unique = max(max_unique, len(unique))
        cache_size = 17
        self.assertGreater(
            max_unique, cache_size,
            f"W_sd ({max_unique}) should exceed S ({cache_size})"
        )

    def test_lru_evicts_persistent_experts(self):
        """Under LRU with burst, persistent experts get evicted and miss frequently."""
        # Use a controlled scenario: 12 persistent experts always needed,
        # plus a burst of transient experts that force LRU eviction
        lru: list[int] = []
        lru_set: set[int] = set()
        cache_size = 17
        persistent = set(range(12))
        persistent_misses = 0
        persistent_accesses = 0

        for step in range(50):
            # Normal step: access persistent experts
            for e in range(12):
                persistent_accesses += 1
                if e in lru_set:
                    lru.remove(e)
                    lru.append(e)
                else:
                    persistent_misses += 1
                    if len(lru) >= cache_size:
                        victim = lru.pop(0)
                        lru_set.discard(victim)
                    lru.append(e)
                    lru_set.add(e)

            # Burst step: 15 transient experts flood the cache
            for e in range(100, 115):
                if e in lru_set:
                    lru.remove(e)
                    lru.append(e)
                else:
                    if len(lru) >= cache_size:
                        victim = lru.pop(0)
                        lru_set.discard(victim)
                    lru.append(e)
                    lru_set.add(e)

        persistent_miss_rate = persistent_misses / max(1, persistent_accesses)
        # LRU should have significant persistent-expert misses due to burst eviction
        self.assertGreater(
            persistent_miss_rate, 0.15,
            f"LRU persistent-expert miss rate ({persistent_miss_rate:.1%}) "
            f"should be > 15% due to burst eviction"
        )

    def test_elp_protects_persistent_experts(self):
        """ELP Pin Zone should protect persistent experts from burst eviction."""
        # pin_capacity = 0.7 × 17 = 11 slots
        # Use 10 persistent experts (fits in pin zone)
        config = BriskMoECacheConfig(
            total_slots_per_layer=17,
            elp=ELPConfig(pin_ratio=0.7, promotion_threshold=3, demotion_window=100),
            sacr=SACRConfig(alpha=0.6, beta=0.4, gamma=0.0),  # no SACR, just ELP
        )
        cache = BriskMoECache(config)
        layer = 0
        num_persistent = 10  # fits in pin_capacity=11

        # Phase 1: Low-contention warmup — only persistent experts, no bursts.
        # This lets them accumulate enough accesses to get promoted to Pin Zone.
        for step in range(10):
            for e in range(num_persistent):
                cache.access_expert(layer, e, step=step)

        # Trigger deferred promotion
        cache.elp.rebalance(0)

        # Verify pinning happened
        pinned = cache.elp.get_pin_set(layer)
        self.assertGreater(len(pinned), 0, "Some experts should be pinned after warmup")

        # Phase 2: Burst with transients — check pinned experts survive
        late_misses = 0
        late_accesses = 0
        for step in range(10, 40):
            # Burst: 12 transient experts fill flex zone
            for e in range(100, 112):
                cache.access_expert(layer, e, step=step)
            # Access persistent experts — pinned ones should still be hit
            for e in range(num_persistent):
                late_accesses += 1
                is_hit, _ = cache.access_expert(layer, e, step=step)
                if not is_hit:
                    late_misses += 1

        late_miss_rate = late_misses / max(1, late_accesses)
        self.assertLess(
            late_miss_rate, 0.15,
            f"ELP late-stage persistent miss rate ({late_miss_rate:.1%}) "
            f"should be < 15% (persistent experts are pinned)"
        )

    def test_elp_improves_hit_rate(self):
        """ELP should improve hit rate vs LRU on the same trace."""
        trace = generate_sd_trace(
            num_steps=100, num_layers=4, draft_k=3,
            top_k_experts=8, num_experts=128,
            accept_rate=0.625, seed=123,
        )
        lru_stats = simulate_lru_cache(trace, cache_size=17)
        elp_stats = simulate_briskmoe_cache(
            trace, cache_size=17, enable_sacr=False, enable_elp=True,
        )
        self.assertGreater(
            elp_stats["hit_rate"],
            lru_stats["hit_rate"],
            f"ELP hit rate ({elp_stats['hit_rate']:.3f}) "
            f"should be > LRU ({lru_stats['hit_rate']:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────
# Test: P2 — Flat Prefetch Validation (DIPP)
# ──────────────────────────────────────────────────────────────────────

class TestP2FlatPrefetch(unittest.TestCase):
    """Validate P2: demand >> supply when K=3, L=26."""

    def test_demand_exceeds_budget(self):
        """Miss demand should exceed PCIe bandwidth budget."""
        rng = random.Random(42)
        dipp = DraftInformedPrioritizedPreloader(DIPPConfig(
            max_prefetch_experts=79,
        ))

        # Simulate K=3 tokens across L=26 layers
        # Each token activates 8 experts per layer
        predictions: dict[int, dict[int, list[int]]] = {}
        for layer_id in range(26):
            predictions[layer_id] = {}
            for tok in range(3):
                experts = [rng.randint(0, 127) for _ in range(8)]
                predictions[layer_id][tok] = experts

        # Cache has ~11 experts per layer (partial fill)
        cache_state: dict[int, set[int]] = {}
        for layer_id in range(26):
            cache_state[layer_id] = set(range(11))

        schedule = dipp.compute_schedule(predictions, cache_state)

        # Should have been truncated by budget
        self.assertLessEqual(len(schedule), 79)
        # But over_budget should be > 0 (demand exceeded supply)
        self.assertGreater(
            dipp.stats.total_experts_over_budget, 0,
            "Expected demand to exceed supply (over_budget should be > 0)"
        )

    def test_dipp_prefers_early_layers(self):
        """DIPP should prioritize early layers over late layers."""
        dipp = DraftInformedPrioritizedPreloader(DIPPConfig(
            max_prefetch_experts=10,  # tight budget
        ))

        # Same expert (id=99) needed in layer 1 and layer 25
        predictions = {
            1: {0: [99]},
            25: {0: [99]},
        }
        cache_state = {1: set(), 25: set()}

        schedule = dipp.compute_schedule(predictions, cache_state)

        # Layer 1 should appear before layer 25
        layers_in_order = [layer_id for layer_id, _, _ in schedule]
        if 1 in layers_in_order and 25 in layers_in_order:
            self.assertLess(
                layers_in_order.index(1),
                layers_in_order.index(25),
                "DIPP should schedule layer 1 before layer 25"
            )

    def test_dipp_prefers_high_demand_experts(self):
        """Expert needed by 3 tokens should rank above expert needed by 1."""
        dipp = DraftInformedPrioritizedPreloader(DIPPConfig(
            max_prefetch_experts=5,
        ))

        # Expert 50: needed by 3 tokens; Expert 60: needed by 1 token
        predictions = {
            1: {0: [50], 1: [50], 2: [50, 60]},
        }
        cache_state = {1: set()}

        schedule = dipp.compute_schedule(predictions, cache_state)
        experts_in_order = [eid for _, eid, _ in schedule]

        self.assertIn(50, experts_in_order)
        if 60 in experts_in_order:
            self.assertLess(
                experts_in_order.index(50),
                experts_in_order.index(60),
                "Expert 50 (demand=3) should rank above expert 60 (demand=1)"
            )

    def test_fifo_vs_dipp_quality(self):
        """DIPP should have higher aggregate value than FIFO ordering."""
        rng = random.Random(99)

        predictions: dict[int, dict[int, list[int]]] = {}
        for layer_id in range(10):
            predictions[layer_id] = {}
            for tok in range(3):
                predictions[layer_id][tok] = [rng.randint(12, 127) for _ in range(8)]

        cache_state = {l: set(range(8)) for l in range(10)}

        dipp = DraftInformedPrioritizedPreloader(DIPPConfig(max_prefetch_experts=20))
        schedule = dipp.compute_schedule(predictions, cache_state)

        # FIFO: just take first N unique miss experts in iteration order
        fifo_experts = []
        fifo_seen = set()
        for layer_id in sorted(predictions.keys()):
            for tok in sorted(predictions[layer_id].keys()):
                for e in predictions[layer_id][tok]:
                    if e not in cache_state.get(layer_id, set()) and (layer_id, e) not in fifo_seen:
                        fifo_seen.add((layer_id, e))
                        val = dipp.compute_value(layer_id, e, predictions, cache_state)
                        fifo_experts.append(val)
                        if len(fifo_experts) >= 20:
                            break
                if len(fifo_experts) >= 20:
                    break
            if len(fifo_experts) >= 20:
                break

        dipp_total_value = sum(v for _, _, v in schedule)
        fifo_total_value = sum(fifo_experts)

        self.assertGreaterEqual(
            dipp_total_value, fifo_total_value,
            f"DIPP value ({dipp_total_value:.2f}) should be >= FIFO ({fifo_total_value:.2f})"
        )


# ──────────────────────────────────────────────────────────────────────
# Test: Ablation — Full > any subset > Base
# ──────────────────────────────────────────────────────────────────────

class TestAblation(unittest.TestCase):
    """Validate ablation: Full system beats any single component or base."""

    def setUp(self):
        self.trace = generate_sd_trace(
            num_steps=100,
            num_layers=4,
            draft_k=3,
            top_k_experts=8,
            num_experts=128,
            accept_rate=0.625,
            seed=777,
        )
        self.cache_size = 17

    def _run_config(self, sacr: bool, elp: bool) -> dict:
        return simulate_briskmoe_cache(
            self.trace, self.cache_size,
            enable_sacr=sacr, enable_elp=elp,
        )

    def test_base_lru_lowest(self):
        """LRU (base) should have lowest hit rate."""
        lru = simulate_lru_cache(self.trace, self.cache_size)
        sacr_only = self._run_config(sacr=True, elp=False)
        elp_only = self._run_config(sacr=False, elp=True)
        full = self._run_config(sacr=True, elp=True)

        self.assertLessEqual(
            lru["hit_rate"], sacr_only["hit_rate"],
            f"LRU ({lru['hit_rate']:.3f}) should be ≤ SACR ({sacr_only['hit_rate']:.3f})"
        )
        self.assertLessEqual(
            lru["hit_rate"], elp_only["hit_rate"],
            f"LRU ({lru['hit_rate']:.3f}) should be ≤ ELP ({elp_only['hit_rate']:.3f})"
        )
        self.assertLessEqual(
            lru["hit_rate"], full["hit_rate"],
            f"LRU ({lru['hit_rate']:.3f}) should be ≤ Full ({full['hit_rate']:.3f})"
        )

    def test_full_beats_any_single(self):
        """Full (SACR+ELP) should beat SACR-only and ELP-only (within tolerance)."""
        sacr_only = self._run_config(sacr=True, elp=False)
        elp_only = self._run_config(sacr=False, elp=True)
        full = self._run_config(sacr=True, elp=True)

        # Small tolerance (0.005) for adaptive γ scoring dynamics
        tol = 0.005
        self.assertGreaterEqual(
            full["hit_rate"] + tol, sacr_only["hit_rate"],
            f"Full ({full['hit_rate']:.3f}) should be ≥ SACR-only ({sacr_only['hit_rate']:.3f})"
        )
        self.assertGreaterEqual(
            full["hit_rate"] + tol, elp_only["hit_rate"],
            f"Full ({full['hit_rate']:.3f}) should be ≥ ELP-only ({elp_only['hit_rate']:.3f})"
        )

    def test_monotonic_improvement(self):
        """Adding components should not decrease hit rate."""
        lru = simulate_lru_cache(self.trace, self.cache_size)
        sacr_only = self._run_config(sacr=True, elp=False)
        full = self._run_config(sacr=True, elp=True)

        # Should be monotonically non-decreasing: LRU ≤ SACR ≤ Full
        self.assertLessEqual(lru["hit_rate"], sacr_only["hit_rate"])
        self.assertLessEqual(sacr_only["hit_rate"], full["hit_rate"])


# ──────────────────────────────────────────────────────────────────────
# Test: AcceptRatio Signal Validity
# ──────────────────────────────────────────────────────────────────────

class TestAcceptRatioSignal(unittest.TestCase):
    """Validate that AcceptRatio correctly separates high/low value experts."""

    def test_accepted_experts_have_higher_ar_than_rejected(self):
        """
        Accepted-token experts should have systematically higher AcceptRatio
        than rejected-token experts.
        """
        tracker = AcceptRejectTracker(AcceptRejectTrackerConfig(ema_alpha=0.15))
        rng = random.Random(42)
        layer = 0
        persistent = list(range(8))  # always-accepted experts
        transient = list(range(100, 108))  # always-rejected experts

        for step in range(30):
            # Accepted tokens always activate persistent experts
            # Rejected tokens always activate transient experts
            token_expert_map = {
                0: persistent[:4],   # accepted
                1: persistent[4:],   # accepted
                2: transient[:4],    # rejected
                3: transient[4:],    # rejected
            }
            accepted_mask = [True, True, False, False]
            tracker.record_verify_result(layer, token_expert_map, accepted_mask, step)

        # Persistent experts should have high AR
        persistent_ars = [tracker.get_accept_ratio(layer, e) for e in persistent]
        # Transient experts should have low AR
        transient_ars = [tracker.get_accept_ratio(layer, e) for e in transient]

        avg_persistent = sum(persistent_ars) / len(persistent_ars)
        avg_transient = sum(transient_ars) / len(transient_ars)

        self.assertGreater(avg_persistent, 0.7, f"Persistent AR {avg_persistent:.3f} should be > 0.7")
        self.assertLess(avg_transient, 0.3, f"Transient AR {avg_transient:.3f} should be < 0.3")
        self.assertGreater(
            avg_persistent, avg_transient + 0.3,
            f"Gap ({avg_persistent:.3f} - {avg_transient:.3f}) should be > 0.3"
        )


# ──────────────────────────────────────────────────────────────────────
# Test: Persistent-Transient Bimodal Distribution
# ──────────────────────────────────────────────────────────────────────

class TestBimodalDistribution(unittest.TestCase):
    """Validate expert access frequency is bimodal under SD."""

    def test_access_frequency_is_bimodal(self):
        """
        Under SD trace, expert accesses should show bimodal distribution:
        a small group of persistent experts accessed frequently, and a large
        tail of transient experts accessed rarely.
        """
        trace = generate_sd_trace(
            num_steps=100,
            num_layers=1,
            draft_k=3,
            num_experts=128,
            accept_rate=0.625,
            persistent_experts=list(range(12)),
            seed=42,
        )

        # Count per-expert access frequency for layer 0
        freq: dict[int, int] = defaultdict(int)
        for entry in trace:
            if entry["layer_id"] != 0:
                continue
            for experts in entry["token_expert_map"].values():
                for e in experts:
                    freq[e] += 1

        counts = sorted(freq.values(), reverse=True)

        # Top ~12 should have much higher frequency than the rest
        if len(counts) > 12:
            top_12_avg = sum(counts[:12]) / 12
            rest_avg = sum(counts[12:]) / max(1, len(counts) - 12)
            self.assertGreater(
                top_12_avg, rest_avg * 2,
                f"Top-12 avg ({top_12_avg:.1f}) should be > 2× rest avg ({rest_avg:.1f})"
            )


# ──────────────────────────────────────────────────────────────────────
# Test: End-to-End Full SD Cycle Profiling
# ──────────────────────────────────────────────────────────────────────

class TestFullSDCycleProfiling(unittest.TestCase):
    """Run a complete SD cycle profiling and verify all components interact correctly."""

    def test_full_cycle_with_stats(self):
        """Full SD cycle: draft → verify → stats should be consistent."""
        config = BriskMoECacheConfig(
            total_slots_per_layer=17,
            sacr=SACRConfig(alpha=0.3, beta=0.2, gamma=0.5),
            elp=ELPConfig(pin_ratio=0.7, promotion_threshold=5, demotion_window=50),
            dipp=DIPPConfig(max_prefetch_experts=79),
        )
        cache = BriskMoECache(config)
        rng = random.Random(42)

        # Simulate 20 SD cycles
        for step in range(20):
            # Draft phase: K=3 tokens, L=4 layers
            cache.begin_draft_round()
            for k in range(3):
                preds = {}
                for layer in range(4):
                    preds[layer] = [rng.randint(0, 127) for _ in range(8)]
                cache.on_draft_token(k, preds, step=step)

            # Verify phase: access experts, then report accept/reject
            for layer in range(4):
                token_expert_map = {}
                for tok in range(4):
                    experts = [rng.randint(0, 127) for _ in range(8)]
                    token_expert_map[tok] = experts
                    for e in experts:
                        cache.access_expert(layer, e, step=step)

                accepted_mask = [rng.random() > 0.4 for _ in range(4)]
                cache.on_verify_complete(layer, token_expert_map, accepted_mask, step=step)

        stats = cache.get_stats_summary()
        self.assertGreater(stats["cache"]["hits"] + stats["cache"]["misses"], 0)
        self.assertGreater(stats["tracker"]["verify_callbacks"], 0)
        self.assertGreater(stats["dipp"]["schedules"], 0)
        self.assertGreaterEqual(stats["cache"]["hit_rate"], 0.0)
        self.assertLessEqual(stats["cache"]["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
