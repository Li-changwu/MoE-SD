"""AdapMoE + MoE-Infinity inspired prefetch planner for vLLM ELMM.

This module replaces the earlier heuristic-only prefetcher with a hybrid
design aligned with both upstream implementations:

- AdapMoE side: confidence-aware adaptive prefetch width.
- MoE-Infinity side: trace-based cross-layer transition prediction.

The planner learns expert transitions online from routing traces and emits
multi-layer (horizon > 1) prefetch plans. It is intentionally lightweight and
does not require model surgery.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass
class AdapMoEConfig:
    # Lower/upper bound of experts prefetched per target layer.
    min_prefetch_k: int = 1
    max_prefetch_k: int = 8
    # Max cross-layer planning distance.
    horizon: int = 2
    # EMA factor for per-layer confidence tracking.
    confidence_ema_alpha: float = 0.10
    # Normalize confidence into [0, 1] by this expected margin.
    expected_margin: float = 3.0
    # Skip prefetch when confidence is too low.
    min_confidence: float = 0.25
    # EMA factor for transition table updates.
    transition_ema_alpha: float = 0.20
    # Decay farther layers in horizon planning.
    horizon_decay: float = 0.75


class AdaptiveGatingPrefetcher:
    """Predict future-layer experts from confidence + learned transitions."""

    def __init__(self, config: AdapMoEConfig):
        self.config = config
        self._layer_confidence_ema: dict[int, float] = {}
        # (src_layer, src_expert) -> {dst_expert: score}
        self._transition: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
        # Per-step trace for online transition learning.
        self._step_trace: dict[int, torch.Tensor] = {}
        self._last_layer_seen: int = -1

    def _begin_new_step_if_needed(self, layer_id: int):
        # Decode/prefill steps typically walk layers in ascending order.
        # If we observe a non-increasing layer id, start a fresh trace window.
        if self._last_layer_seen >= 0 and layer_id <= self._last_layer_seen:
            self._step_trace.clear()
        self._last_layer_seen = layer_id

    def _record_transition(self, src_layer: int, src_ids: torch.Tensor, dst_ids: torch.Tensor):
        if src_ids.numel() == 0 or dst_ids.numel() == 0:
            return
        a = self.config.transition_ema_alpha
        src_flat = src_ids.reshape(-1).to(torch.long)
        dst_flat = dst_ids.reshape(-1).to(torch.long)

        src_unique, src_cnt = torch.unique(src_flat, sorted=False, return_counts=True)
        dst_unique, dst_cnt = torch.unique(dst_flat, sorted=False, return_counts=True)
        src_weight = src_cnt.float() / float(max(1, src_cnt.sum().item()))
        dst_weight = dst_cnt.float() / float(max(1, dst_cnt.sum().item()))

        for i, src_t in enumerate(src_unique):
            src_eid = int(src_t.item())
            key = (src_layer, src_eid)
            row = self._transition[key]
            s_w = float(src_weight[i].item())
            for j, dst_t in enumerate(dst_unique):
                dst_eid = int(dst_t.item())
                contrib = s_w * float(dst_weight[j].item())
                prev = row.get(dst_eid, 0.0)
                row[dst_eid] = (1.0 - a) * prev + a * contrib

    def observe_layer(self, layer_id: int, topk_ids: torch.Tensor):
        """Ingest current layer routing and update transition traces online."""
        self._begin_new_step_if_needed(layer_id)
        cur = topk_ids.detach()
        prev_ids = self._step_trace.get(layer_id - 1)
        if prev_ids is not None:
            self._record_transition(layer_id - 1, prev_ids, cur)
        self._step_trace[layer_id] = cur

    def _compute_confidence(
        self,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> float:
        """
        Combine routing margin and hidden-state norm as confidence score.

        - routing margin: top1 - top2 on router logits
        - hidden norm: RMS of hidden vectors (attention-output proxy)
        """
        if router_logits.numel() == 0:
            return 0.0

        logits = router_logits.float()
        top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            margin = top2[..., 0]
        else:
            margin = top2[..., 0] - top2[..., 1]
        margin_mean = float(margin.mean().item())

        hs = hidden_states.float()
        hs_rms = float(torch.sqrt(torch.clamp((hs * hs).mean(), min=1e-12)).item())

        margin_norm = max(0.0, min(1.0, margin_mean / self.config.expected_margin))
        # Compress hidden norm into a stable range [0, 1).
        hs_norm = hs_rms / (1.0 + hs_rms)

        # Weighted fusion: gating signal dominates.
        return 0.8 * margin_norm + 0.2 * hs_norm

    def _adaptive_width(self, layer_id: int, conf: float) -> int:
        ema_old = self._layer_confidence_ema.get(layer_id, conf)
        a = self.config.confidence_ema_alpha
        ema_new = (1.0 - a) * ema_old + a * conf
        self._layer_confidence_ema[layer_id] = ema_new

        span = max(0, self.config.max_prefetch_k - self.config.min_prefetch_k)
        return self.config.min_prefetch_k + int(round(span * ema_new))

    def predict_layer_experts(
        self,
        current_layer_id: int,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cached: set[int],
        layer_num_experts: int,
        width_cap: int,
    ) -> list[int]:
        """Compatibility wrapper: predict experts for immediate next layer."""
        plan = self.plan_prefetch(
            current_layer_id=current_layer_id,
            topk_ids=topk_ids,
            router_logits=router_logits,
            hidden_states=hidden_states,
            targets=[
                {
                    "layer_id": current_layer_id + 1,
                    "cached": layer_cached,
                    "num_experts": layer_num_experts,
                    "width_cap": width_cap,
                }
            ],
        )
        return plan.get(current_layer_id + 1, [])

    def _predict_next_distribution(
        self,
        src_layer: int,
        src_ids: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Predict next-layer expert scores from learned transition table."""
        src_flat = src_ids.reshape(-1).to(torch.long)
        out = torch.zeros(num_experts, dtype=torch.float32)
        if src_flat.numel() == 0:
            return out

        src_unique, src_cnt = torch.unique(src_flat, sorted=False, return_counts=True)
        src_weight = src_cnt.float() / float(max(1, src_cnt.sum().item()))
        for i, src_t in enumerate(src_unique):
            src_eid = int(src_t.item())
            row = self._transition.get((src_layer, src_eid))
            if not row:
                continue
            w = float(src_weight[i].item())
            for dst_eid, score in row.items():
                if 0 <= dst_eid < num_experts:
                    out[dst_eid] += w * float(score)
        return out

    def plan_prefetch(
        self,
        current_layer_id: int,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        targets: list[dict],
    ) -> dict[int, list[int]]:
        """
        Build multi-layer prefetch plan.

        Args:
            targets: list of dicts with keys:
                - layer_id: absolute target layer id
                - cached: set[int] currently cached experts for that layer
                - num_experts: expert count of target layer
                - width_cap: hard cap for selected experts
        Returns:
            mapping target_layer_id -> sorted expert ids to prefetch
        """
        self.observe_layer(current_layer_id, topk_ids)
        if topk_ids.numel() == 0 or not targets:
            return {}

        conf = self._compute_confidence(router_logits, hidden_states)
        if conf < self.config.min_confidence:
            return {}

        base_k = self._adaptive_width(current_layer_id, conf)
        if base_k <= 0:
            return {}

        # Seed distribution from current routing frequencies.
        seed_flat = topk_ids.reshape(-1).to(torch.long)
        seed_unique, seed_cnt = torch.unique(seed_flat, sorted=False, return_counts=True)
        seed_dist = {
            int(e.item()): float(c.item())
            for e, c in zip(seed_unique, seed_cnt)
        }

        # Predict layer by layer; each hop reuses the previous hop distribution.
        targets_sorted = sorted(targets, key=lambda x: int(x["layer_id"]))
        plan: dict[int, list[int]] = {}

        prev_layer = current_layer_id
        prev_ids = torch.tensor(list(seed_dist.keys()), dtype=torch.long)

        for t in targets_sorted:
            layer_id = int(t["layer_id"])
            if layer_id <= prev_layer:
                continue

            num_experts = max(1, int(t["num_experts"]))
            cached = t["cached"]
            width_cap = max(1, int(t["width_cap"]))

            # Advance from prev_layer -> layer_id via repeated transition lookups.
            dist = None
            hop = prev_layer
            cur_ids = prev_ids
            while hop < layer_id:
                dist_t = self._predict_next_distribution(
                    src_layer=hop,
                    src_ids=cur_ids,
                    num_experts=num_experts,
                )
                if dist is None:
                    dist = dist_t
                else:
                    dist = dist + dist_t
                cur_ids = torch.nonzero(dist_t > 0, as_tuple=False).reshape(-1)
                hop += 1

            if dist is None or dist.numel() == 0:
                prev_layer = layer_id
                prev_ids = torch.empty(0, dtype=torch.long)
                continue

            # Combine predicted transition score with confidence and distance decay.
            dist_hops = max(1, layer_id - current_layer_id)
            decay = self.config.horizon_decay ** (dist_hops - 1)
            score = dist * (0.5 + conf) * decay
            sorted_ids = torch.argsort(score, descending=True)

            # Farther layers use narrower widths to avoid over-prefetching.
            layer_k = max(1, int(round(base_k * decay)))
            k = min(width_cap, layer_k)

            selected: list[int] = []
            for eid_t in sorted_ids:
                eid = int(eid_t.item())
                if score[eid] <= 0:
                    break
                if eid in cached:
                    continue
                selected.append(eid)
                if len(selected) >= k:
                    break

            if selected:
                plan[layer_id] = selected

            prev_layer = layer_id
            prev_ids = torch.tensor(selected, dtype=torch.long)

        return plan
