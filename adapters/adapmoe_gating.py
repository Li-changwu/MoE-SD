"""AdapMoE sensitivity-based adaptive gating for vLLM ELMM.

This implementation removes the previous coarse approximation and follows the
paper's core control loop more closely:

- Layer-wise sensitivity-aware single-expert target.
- Token confidence thresholding using top-1/top-2 routing margin.
- Online stability control with EMA-smoothed thresholds and sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdapMoEGatingConfig:
    # Global target single-expert selection ratio.
    target_single_ratio: float = 0.24
    # Warmup decode calls before enabling adaptive single-expert routing.
    warmup_steps: int = 128
    # EMA for per-layer confidence threshold.
    threshold_ema_alpha: float = 0.10
    # Minimum target ratio on sensitive (early) layers.
    min_layer_scale: float = 0.35
    # EMA for online layer sensitivity score.
    sensitivity_ema_alpha: float = 0.08
    # Weight for entropy in sensitivity estimate.
    entropy_weight: float = 0.55


class SensitivityAdaptiveGating:
    """Adjust active expert count per token with layer sensitivity awareness."""

    def __init__(self, num_layers: int, config: AdapMoEGatingConfig):
        self.num_layers = max(1, num_layers)
        self.config = config
        self._layer_threshold_ema: dict[int, float] = {}
        self._layer_sensitivity_ema: dict[int, float] = {}
        self._step: int = 0

    def _layer_target_ratio(self, layer_rank: int, layer_id: int) -> float:
        # Early layers are more sensitive; preserve more multi-expert routing.
        if self.num_layers <= 1:
            return self.config.target_single_ratio
        pos = float(layer_rank) / float(self.num_layers - 1)
        base_scale = self.config.min_layer_scale + (1.0 - self.config.min_layer_scale) * pos

        # Additional adaptive scaling from online sensitivity estimate.
        sens = self._layer_sensitivity_ema.get(layer_id, 0.5)
        adaptive_scale = 1.0 - 0.6 * sens
        scale = max(self.config.min_layer_scale, base_scale * adaptive_scale)
        return max(0.0, min(1.0, self.config.target_single_ratio * scale))

    def _update_layer_sensitivity(
        self,
        layer_id: int,
        topk_weights: torch.Tensor,
    ) -> float:
        if topk_weights.numel() == 0:
            return self._layer_sensitivity_ema.get(layer_id, 0.5)

        w = torch.clamp(topk_weights.float(), min=1e-8)
        # Lower margin and higher entropy imply higher sensitivity.
        margin = torch.clamp(w[:, 0] - w[:, 1], min=0.0)
        margin_norm = margin / (margin + 1.0)
        margin_sens = 1.0 - margin_norm.mean()

        ent = -(w * torch.log(w)).sum(dim=-1)
        ent_norm = ent / torch.log(torch.tensor(float(w.shape[-1]), device=w.device))
        ent_norm = torch.clamp(ent_norm, 0.0, 1.0).mean()

        sens_now = (1.0 - self.config.entropy_weight) * margin_sens + self.config.entropy_weight * ent_norm
        sens_now_f = float(torch.clamp(sens_now, 0.0, 1.0).item())

        old = self._layer_sensitivity_ema.get(layer_id, sens_now_f)
        a = self.config.sensitivity_ema_alpha
        new = (1.0 - a) * old + a * sens_now_f
        self._layer_sensitivity_ema[layer_id] = new
        return new

    def apply(
        self,
        layer_id: int,
        layer_rank: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Return (new_weights, new_ids, single_ratio).

        Only changes routing when k >= 2 and warmup is over.
        """
        self._step += 1

        if topk_weights.ndim != 2 or topk_weights.shape[-1] < 2:
            return topk_weights, topk_ids, 0.0
        if self._step <= self.config.warmup_steps:
            return topk_weights, topk_ids, 0.0

        self._update_layer_sensitivity(layer_id, topk_weights)
        target_ratio = self._layer_target_ratio(layer_rank, layer_id)
        if target_ratio <= 0.0:
            return topk_weights, topk_ids, 0.0

        w = topk_weights
        margin = (w[:, 0] - w[:, 1]).float()
        # Quantile threshold: keep top confident tokens as single-expert.
        q = max(0.0, min(1.0, 1.0 - target_ratio))
        th_now = float(torch.quantile(margin, q).item())
        th_old = self._layer_threshold_ema.get(layer_id, th_now)
        a = self.config.threshold_ema_alpha
        th = (1.0 - a) * th_old + a * th_now
        self._layer_threshold_ema[layer_id] = th

        mask = margin >= th
        if not bool(mask.any()):
            return topk_weights, topk_ids, 0.0

        new_w = topk_weights.clone()
        new_ids = topk_ids.clone()

        # For selected tokens, collapse to one expert.
        new_w[mask, 0] = 1.0
        new_w[mask, 1:] = 0.0
        new_ids[mask, 1:] = new_ids[mask, 0].unsqueeze(1)

        single_ratio = float(mask.float().mean().item())
        return new_w, new_ids, single_ratio
