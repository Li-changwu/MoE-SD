"""MoE-Infinity style request-level expert prefetch planner.

This module implements the key idea from MoE-Infinity (OSDI'24):
predict future expert activations using historical sequence-level routing
patterns, then prefetch experts ahead of use.

Design for vLLM ELMM:
- Observe routing online during layer-by-layer execution.
- Segment traces by detecting layer index wrap-around (new request).
- Match current partial trace against a history bank.
- Emit per-layer expert prefetch plans for a short horizon.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class MoEInfinityConfig:
    history_size: int = 64
    horizon: int = 2
    max_prefetch_k: int = 4
    min_similarity: float = 0.05


class SequenceTracePrefetcher:
    """Request-level trace matcher and prefetch planner."""

    def __init__(self, num_layers: int, config: MoEInfinityConfig):
        self.num_layers = max(1, num_layers)
        self.config = config

        # Current request partial trace: layer_id -> {expert_id: count}
        self._cur_trace: dict[int, dict[int, float]] = {}
        self._last_layer_seen: int = -1

        # History bank of completed traces.
        # Each item: dict[layer_id] -> dict[expert_id] -> weight
        self._history: deque[dict[int, dict[int, float]]] = deque(maxlen=max(1, config.history_size))

    def _finalize_current_trace(self):
        if self._cur_trace:
            self._history.append({
                lid: dict(edict)
                for lid, edict in self._cur_trace.items()
            })
            self._cur_trace.clear()

    def observe_layer(self, layer_id: int, topk_ids: torch.Tensor):
        """Ingest routing for one layer of current request."""
        if self._last_layer_seen >= 0 and layer_id <= self._last_layer_seen:
            # New request begins; archive previous one.
            self._finalize_current_trace()
        self._last_layer_seen = layer_id

        if topk_ids.numel() == 0:
            return

        flat = topk_ids.reshape(-1).to(torch.long)
        uniq, cnt = torch.unique(flat, sorted=False, return_counts=True)
        layer_map = self._cur_trace.setdefault(layer_id, {})
        for e_t, c_t in zip(uniq, cnt):
            e = int(e_t.item())
            layer_map[e] = layer_map.get(e, 0.0) + float(c_t.item())

    def _trace_similarity(self, hist: dict[int, dict[int, float]], up_to_layer: int) -> float:
        """Cosine similarity on observed prefix layers."""
        dot = 0.0
        na = 0.0
        nb = 0.0
        for lid, cur_map in self._cur_trace.items():
            if lid > up_to_layer:
                continue
            hist_map = hist.get(lid)
            if not hist_map:
                continue
            for e, w in cur_map.items():
                dot += w * hist_map.get(e, 0.0)
                na += w * w
            for w in hist_map.values():
                nb += w * w

        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    def _predict_layer_scores(self, target_layer: int, current_layer: int) -> dict[int, float]:
        scores: dict[int, float] = {}
        if not self._history:
            return scores

        for hist in self._history:
            sim = self._trace_similarity(hist, up_to_layer=current_layer)
            if sim < self.config.min_similarity:
                continue
            hmap = hist.get(target_layer)
            if not hmap:
                continue
            for e, w in hmap.items():
                scores[e] = scores.get(e, 0.0) + sim * w

        return scores

    def plan_prefetch(
        self,
        current_layer_id: int,
        topk_ids: torch.Tensor,
        targets: list[dict],
    ) -> dict[int, list[int]]:
        """Return target_layer_id -> expert_id list for prefetch."""
        self.observe_layer(current_layer_id, topk_ids)
        if not targets:
            return {}

        plan: dict[int, list[int]] = {}
        for t in sorted(targets, key=lambda x: int(x["layer_id"])):
            layer_id = int(t["layer_id"])
            if layer_id <= current_layer_id:
                continue
            if layer_id - current_layer_id > max(1, self.config.horizon):
                continue

            cached = t["cached"]
            width_cap = max(1, int(t["width_cap"]))
            num_experts = max(1, int(t["num_experts"]))

            scores = self._predict_layer_scores(layer_id, current_layer_id)
            if not scores:
                continue

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            out: list[int] = []
            for eid, _ in ranked:
                if eid < 0 or eid >= num_experts:
                    continue
                if eid in cached:
                    continue
                out.append(eid)
                if len(out) >= min(width_cap, self.config.max_prefetch_k):
                    break

            if out:
                plan[layer_id] = out

        return plan
