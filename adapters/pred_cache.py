"""
PredCache — Predictive Expert Cache Management
================================================
Forward-looking eviction + prefetch using speculation-provided future demand.

Core insight: SD's draft phase provides "future knowledge" about expert
demand — the exact information Belady's OPT algorithm needs. PredCache
uses this to approximate OPT online:

  - PredEvict: evict experts with lowest predicted future demand
  - PredLoad:  prioritize prefetching by predicted demand × urgency
  - PredReserve: protect cached experts with high predicted demand

PredScore(e) = D_hat(e) + λ·R(e)

  D_hat(e) = predicted demand count (from last verify's router logits)
  R(e)     = normalized LRU recency (fallback for unpredicted experts)
  λ        = fallback weight (decreases as prediction confidence grows)

When D_hat is unavailable (cold start), PredCache degrades gracefully to LRU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class PredCacheConfig:
    """Configuration for PredCache."""
    # Fallback weight for LRU recency when prediction is uncertain
    lru_fallback_weight: float = 10.0
    # Top-k used by router (for demand counting)
    top_k: int = 8
    # Number of experts per layer
    num_experts: int = 128
    # Max prefetch experts per round (PCIe bandwidth budget)
    max_prefetch_experts: int = 79
    # Urgency decay for cross-layer prefetch priority
    urgency_decay: str = "inverse"  # "inverse", "linear", "exp"
    # EMA decay factor for demand signal (0 = only latest, 1 = never forget)
    demand_decay: float = 0.5
    # EAMC capacity (number of historical request-level traces).
    eamc_capacity: int = 120
    # Number of matched traces used to aggregate pEAM.
    eam_match_topk: int = 3
    # Max cosine distance for a trace to be considered a match.
    eam_match_max_distance: float = 0.35
    # Max decode iterations aggregated into one request-level trace.
    request_trace_max_iters: int = 256
    # Relative weight of trace likelihood in final score.
    trace_likelihood_weight: float = 1.0
    # Relative weight of EMA-demand fallback in final score.
    demand_fallback_weight: float = 0.25
    # Keep prediction focused on current/future layers only.
    future_layer_only: bool = True


class PredictiveExpertCacheManager:
    """
    Unified forward-looking cache manager for SD-aware MoE inference.

    Uses router logits from the previous verification step as predictions
    for the next step's expert demand. This is a lightweight approximation
    that exploits temporal locality in routing patterns.

    Phase A (real draft routing): In future, can be upgraded to use
    draft model's hidden states → target router gate for true K×L prediction.
    """

    def __init__(self, config: Optional[PredCacheConfig] = None):
        self.config = config or PredCacheConfig()
        # Per-layer predicted demand: layer_id → list[float] of length num_experts
        # demand[e] = raw demand (before lazy decay)
        self._predicted_demand: dict[int, list[float]] = {}
        # Per-layer demand update step: layer_id → list[int]
        # Tracks when demand was last updated for lazy decay computation
        self._demand_step: dict[int, list[int]] = {}
        # Per-layer LRU timestamps: layer_id → list[int] (last access step per expert)
        self._last_access: dict[int, list[int]] = {}
        # Global step counter
        self._step: int = 0
        # Stats
        self._evictions_pred: int = 0     # evictions guided by prediction
        self._evictions_lru: int = 0      # evictions falling back to LRU
        self._prefetch_scheduled: int = 0
        # Layer metadata (discovered online).
        self._max_layer_seen: int = -1
        # Request/iteration traces.
        self._iter_trace: dict[int, dict[int, float]] = {}
        self._request_trace: dict[int, list[float]] = {}
        self._request_iters: int = 0
        self._last_layer_seen: Optional[int] = None
        # EAMC: historical request-level traces.
        # Each item keeps:
        #   - flat_unit: unit-normalized flattened request trace
        #   - row_probs: per-layer normalized expert activation probabilities
        self._eamc: list[dict[str, list]] = []
        # Latest predicted pEAM likelihood (from matched EAMC traces).
        self._trace_likelihood: dict[int, list[float]] = {}
        self._trace_confidence: float = 0.0

    def _ensure_layer(self, layer_id: int) -> None:
        """Lazy-init arrays for a new layer."""
        if layer_id not in self._predicted_demand:
            n = self.config.num_experts
            self._predicted_demand[layer_id] = [0.0] * n
            self._demand_step[layer_id] = [0] * n
            self._last_access[layer_id] = [0] * n
        if layer_id > self._max_layer_seen:
            self._max_layer_seen = layer_id
        if layer_id not in self._request_trace:
            self._request_trace[layer_id] = [0.0] * self.config.num_experts

    def _get_demand(self, layer_id: int, expert_id: int) -> float:
        """Get lazily-decayed demand for an expert. O(1)."""
        raw = self._predicted_demand[layer_id][expert_id]
        if raw == 0.0:
            return 0.0
        age = self._step - self._demand_step[layer_id][expert_id]
        if age <= 0:
            return raw
        return raw * (self.config.demand_decay ** age)

    def record_access(self, layer_id: int, expert_id: int) -> None:
        """Record that expert was accessed at current step."""
        self._ensure_layer(layer_id)
        self._last_access[layer_id][expert_id] = self._step

    def record_access_batch(self, layer_id: int, expert_ids: list[int]) -> None:
        """Record batch of expert accesses."""
        self._ensure_layer(layer_id)
        la = self._last_access[layer_id]
        step = self._step
        for eid in expert_ids:
            la[eid] = step

    def update_predictions_from_logits(
        self, layer_id: int, router_logits_topk_ids: list[list[int]]
    ) -> None:
        """
        Update predicted demand from router's topk_ids.

        Uses lazy decay — only touches the experts that appear in routing.
        O(top_k × num_tokens) instead of O(num_experts).
        """
        self._ensure_layer(layer_id)
        demand = self._predicted_demand[layer_id]
        ds = self._demand_step[layer_id]
        step = self._step
        decay = self.config.demand_decay
        n = self.config.num_experts
        # Update only touched experts (lazy decay + add)
        for token_experts in router_logits_topk_ids:
            for eid in token_experts:
                if 0 <= eid < n:
                    age = step - ds[eid]
                    demand[eid] = demand[eid] * (decay ** age) + 1.0
                    ds[eid] = step

    def update_predictions_from_flat(
        self, layer_id: int, topk_ids_flat: list[int]
    ) -> None:
        """
        Update predicted demand from a flat list of expert IDs.
        Uses lazy decay — O(len(topk_ids_flat)) not O(num_experts).
        """
        self._ensure_layer(layer_id)
        demand = self._predicted_demand[layer_id]
        ds = self._demand_step[layer_id]
        step = self._step
        decay = self.config.demand_decay
        n = self.config.num_experts
        for eid in topk_ids_flat:
            if 0 <= eid < n:
                age = step - ds[eid]
                demand[eid] = demand[eid] * (decay ** age) + 1.0
                ds[eid] = step

    def _num_layers(self) -> int:
        if self._max_layer_seen < 0:
            return 1
        return self._max_layer_seen + 1

    def _finalize_iteration(self) -> None:
        if not self._iter_trace:
            return
        for layer_id, cnts in self._iter_trace.items():
            self._ensure_layer(layer_id)
            row = self._request_trace[layer_id]
            for eid, c in cnts.items():
                row[eid] += c
        self._iter_trace.clear()
        self._request_iters += 1
        if self._request_iters >= self.config.request_trace_max_iters:
            self.finalize_request_trace()

    def _flatten_request_trace(self) -> list[float]:
        n_layers = self._num_layers()
        n_experts = self.config.num_experts
        vec = [0.0] * (n_layers * n_experts)
        for lid in range(n_layers):
            row = self._request_trace.get(lid)
            if row is None:
                continue
            base = lid * n_experts
            for eid in range(n_experts):
                vec[base + eid] = float(row[eid])
        return vec

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm_sq = 0.0
        for v in vec:
            norm_sq += v * v
        if norm_sq <= 0.0:
            return [0.0] * len(vec)
        inv = 1.0 / math.sqrt(norm_sq)
        return [v * inv for v in vec]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        s = 0.0
        for i in range(n):
            s += a[i] * b[i]
        return s

    def _row_normalized_request_trace(self) -> list[list[float]]:
        n_layers = self._num_layers()
        n_experts = self.config.num_experts
        out: list[list[float]] = []
        for lid in range(n_layers):
            row = self._request_trace.get(lid)
            if row is None:
                out.append([0.0] * n_experts)
                continue
            s = sum(row)
            if s <= 0.0:
                out.append([0.0] * n_experts)
            else:
                inv = 1.0 / s
                out.append([float(v) * inv for v in row])
        return out

    def finalize_request_trace(self) -> None:
        """
        Archive current request-level trace into EAMC.

        Replacement policy follows MoE-Infinity's practical design:
        when full, replace the most similar stored trace.
        """
        self._finalize_iteration()
        flat = self._flatten_request_trace()
        flat_unit = self._l2_normalize(flat)
        if not any(flat_unit):
            self._request_trace.clear()
            self._request_iters = 0
            return

        row_probs = self._row_normalized_request_trace()
        entry = {
            "flat_unit": flat_unit,
            "row_probs": row_probs,
        }

        if len(self._eamc) < self.config.eamc_capacity:
            self._eamc.append(entry)
        else:
            # Replace the most similar existing trace to keep diversity.
            best_sim = -2.0
            best_idx = 0
            for i, old in enumerate(self._eamc):
                sim = self._dot(flat_unit, old["flat_unit"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            self._eamc[best_idx] = entry

        self._request_trace.clear()
        self._request_iters = 0

    def _flatten_iter_trace(self) -> list[float]:
        n_layers = self._num_layers()
        n_experts = self.config.num_experts
        vec = [0.0] * (n_layers * n_experts)
        for lid, row in self._iter_trace.items():
            if lid < 0 or lid >= n_layers:
                continue
            base = lid * n_experts
            for eid, c in row.items():
                if 0 <= eid < n_experts:
                    vec[base + eid] = float(c)
        return vec

    def _layer_proximity_weight(self, layer_id: int, current_layer: int) -> float:
        n_layers = self._num_layers()
        if n_layers <= 0:
            return 1.0
        if self.config.future_layer_only and layer_id < current_layer:
            return 0.0
        # MoE-Infinity style layer proximity: (1 - (i-l)/L)
        return max(0.0, 1.0 - ((layer_id - current_layer) / float(n_layers)))

    def _update_trace_likelihood(self, current_layer: int) -> None:
        if not self._eamc:
            self._trace_likelihood.clear()
            self._trace_confidence = 0.0
            return

        cur_unit = self._l2_normalize(self._flatten_iter_trace())
        if not any(cur_unit):
            self._trace_likelihood.clear()
            self._trace_confidence = 0.0
            return

        scored: list[tuple[float, dict[str, list]]] = []
        for e in self._eamc:
            sim = self._dot(cur_unit, e["flat_unit"])
            dist = 1.0 - sim
            if dist <= self.config.eam_match_max_distance:
                scored.append((sim, e))

        if not scored:
            # Fallback: still pick nearest trace to avoid empty predictor.
            best_sim = -2.0
            best_entry = self._eamc[0]
            for e in self._eamc:
                sim = self._dot(cur_unit, e["flat_unit"])
                if sim > best_sim:
                    best_sim = sim
                    best_entry = e
            scored = [(best_sim, best_entry)]

        scored.sort(key=lambda x: x[0], reverse=True)
        topk = max(1, self.config.eam_match_topk)
        picked = scored[:topk]
        self._trace_confidence = max(0.0, min(1.0, picked[0][0]))

        n_layers = self._num_layers()
        n_experts = self.config.num_experts
        agg: dict[int, list[float]] = {}
        total_w = 0.0
        for sim, e in picked:
            w = max(1e-6, sim)
            total_w += w
            rows = e["row_probs"]
            for lid in range(min(n_layers, len(rows))):
                row = rows[lid]
                out_row = agg.get(lid)
                if out_row is None:
                    out_row = [0.0] * n_experts
                    agg[lid] = out_row
                for eid in range(min(n_experts, len(row))):
                    out_row[eid] += w * float(row[eid])

        if total_w <= 0.0:
            self._trace_likelihood.clear()
            return

        inv = 1.0 / total_w
        self._trace_likelihood = {}
        for lid, row in agg.items():
            prox = self._layer_proximity_weight(lid, current_layer)
            if prox <= 0.0:
                continue
            self._trace_likelihood[lid] = [v * inv * prox for v in row]

    def observe_layer_routing(self, layer_id: int, topk_ids_flat: list[int]) -> None:
        """
        Update iteration/request traces and online pEAM prediction.

        Layer wrap-around (e.g., L-1 -> 0) is used as a lightweight
        signal for iteration boundary during decoding.
        """
        self._ensure_layer(layer_id)
        if self._last_layer_seen is not None and layer_id < self._last_layer_seen:
            self._finalize_iteration()
        self._last_layer_seen = layer_id

        row = self._iter_trace.get(layer_id)
        if row is None:
            row = {}
            self._iter_trace[layer_id] = row
        n = self.config.num_experts
        for eid in topk_ids_flat:
            if 0 <= eid < n:
                row[eid] = row.get(eid, 0.0) + 1.0

        # Keep EMA demand as a robust fallback signal.
        self.update_predictions_from_flat(layer_id, topk_ids_flat)
        self._update_trace_likelihood(layer_id)

    def pred_score(self, layer_id: int, expert_id: int) -> float:
        """
        Compute PredScore for a cached expert.

        PredScore = D_hat(e) + λ·R(e)

        Higher score = more valuable = should NOT be evicted.
        """
        self._ensure_layer(layer_id)
        demand = self._get_demand(layer_id, expert_id)
        trace_like = 0.0
        tr = self._trace_likelihood.get(layer_id)
        if tr is not None and 0 <= expert_id < len(tr):
            trace_like = tr[expert_id]
        # Normalized recency: recent access → high R
        age = self._step - self._last_access[layer_id][expert_id]
        recency = 1.0 / (1.0 + age)
        return (
            self.config.trace_likelihood_weight * trace_like
            + self.config.demand_fallback_weight * demand
            + self.config.lru_fallback_weight * recency
        )

    def select_victim(
        self, layer_id: int, candidates: list[int]
    ) -> int:
        """
        Select the best eviction victim from candidates.

        Evicts the expert with the lowest PredScore
        (= lowest predicted future demand + lowest recency).
        """
        if not candidates:
            raise ValueError("No candidates for eviction")

        self._ensure_layer(layer_id)
        best_victim = candidates[0]
        best_score = self.pred_score(layer_id, best_victim)

        for i in range(1, len(candidates)):
            eid = candidates[i]
            score = self.pred_score(layer_id, eid)
            if score < best_score:
                best_score = score
                best_victim = eid

        return best_victim

    def get_demand_boost(self, layer_id: int, expert_id: int) -> float:
        """
        Get demand-based priority boost for an expert. O(1).

        Returns the lazily-decayed EMA demand. Used by DIPP to enhance
        its Value function: V(l,e) = (1 + demand_boost) × urgency(l).
        """
        if layer_id not in self._predicted_demand:
            return 0.0
        n = self.config.num_experts
        if expert_id < 0 or expert_id >= n:
            return 0.0
        return self._get_demand(layer_id, expert_id)

    def compute_prefetch_schedule(
        self,
        cache_states: dict[int, set[int]],
        num_layers: int = 48,
    ) -> list[tuple[int, int, float]]:
        """
        Compute cross-layer prefetch schedule using predicted demand.

        Returns list of (layer_id, expert_id, priority) sorted by
        priority descending, truncated to bandwidth budget.

        Priority = demand(e) × urgency(layer) for miss experts.
        """
        candidates: list[tuple[float, int, int]] = []
        current_layer = self._last_layer_seen if self._last_layer_seen is not None else 0

        for layer_id in range(num_layers):
            if layer_id not in self._predicted_demand and layer_id not in self._trace_likelihood:
                continue
            cached = cache_states.get(layer_id, set())
            urgency = self._urgency(layer_id)
            prox = self._layer_proximity_weight(layer_id, current_layer)
            if prox <= 0.0:
                continue

            trace_row = self._trace_likelihood.get(layer_id)

            for eid in range(self.config.num_experts):
                if eid in cached:
                    continue
                trace_like = 0.0
                if trace_row is not None and eid < len(trace_row):
                    trace_like = trace_row[eid]
                d = self._get_demand(layer_id, eid)
                p = (
                    self.config.trace_likelihood_weight * trace_like
                    + self.config.demand_fallback_weight * d
                )
                if p > 0:
                    value = p * urgency * prox
                    candidates.append((value, layer_id, eid))

        candidates.sort(reverse=True)
        budget = self.config.max_prefetch_experts
        selected = candidates[:budget]
        self._prefetch_scheduled += len(selected)

        return [(lid, eid, val) for val, lid, eid in selected]

    def advance_step(self) -> None:
        """Advance the global step counter."""
        self._step += 1

    def get_stats(self) -> dict:
        return {
            "step": self._step,
            "evictions_pred_guided": self._evictions_pred,
            "evictions_lru_fallback": self._evictions_lru,
            "prefetch_scheduled": self._prefetch_scheduled,
            "eamc_size": len(self._eamc),
            "trace_confidence": self._trace_confidence,
        }

    def _urgency(self, layer_id: int) -> float:
        adjusted = layer_id + 1
        mode = self.config.urgency_decay
        if mode == "inverse":
            return 1.0 / adjusted
        elif mode == "linear":
            return max(0.01, 1.0 - (adjusted - 1) * 0.03)
        elif mode == "exp":
            return 0.95 ** (adjusted - 1)
        return 1.0 / adjusted
