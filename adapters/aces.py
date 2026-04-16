"""
ACES — Adaptive Cache with Expert Scoring.

Uses the MoE router's full softmax probability distribution (not just
the top-K selection) to maintain an EMA priority score per expert per
layer.  Cache eviction picks the cached expert with the lowest EMA
score, naturally fusing recency, frequency, and routing confidence
into a single signal.

Key insight: traditional caches (LRU/LFU) only observe binary
access events. MoE routers produce a *continuous* probability
distribution over all experts — including "near-miss" experts that
almost entered top-K. ACES exploits this richer signal.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ACESConfig:
    beta: float = 0.9         # EMA momentum (higher = more history)
    num_experts: int = 128    # experts per layer


class ACESPolicy:
    """Per-layer EMA priority tracker for expert cache eviction."""

    __slots__ = ("config", "_ema",)

    def __init__(self, config: ACESConfig):
        self.config = config
        # layer_id -> float tensor of shape [num_experts]
        self._ema: dict[int, torch.Tensor] = {}

    def _get_ema(self, layer_id: int) -> torch.Tensor:
        if layer_id not in self._ema:
            self._ema[layer_id] = torch.zeros(
                self.config.num_experts, dtype=torch.float32
            )
        return self._ema[layer_id]

    def update(self, layer_id: int, router_logits: torch.Tensor) -> None:
        """
        Update EMA from router logits.

        Args:
            layer_id: integer layer index
            router_logits: shape [num_tokens, num_experts] raw logits from gate
        """
        # Average across tokens in this step, then softmax to get probs
        # For decode (1 token), this is just softmax of the single row
        with torch.no_grad():
            mean_logits = router_logits.float().mean(dim=0)  # [num_experts]
            probs = torch.softmax(mean_logits, dim=0).cpu()  # move to CPU

        ema = self._get_ema(layer_id)
        beta = self.config.beta
        ema.mul_(beta).add_(probs, alpha=(1.0 - beta))

    def select_victim(self, layer_id: int, cached_eids: list[int],
                      protected_eids: set[int] | None = None) -> int:
        """
        Select the cached expert with the lowest EMA priority to evict.

        Args:
            layer_id: layer index
            cached_eids: list of expert IDs currently in cache
            protected_eids: experts needed this step (do not evict)

        Returns:
            expert_id to evict
        """
        ema = self._get_ema(layer_id)
        best_victim = None
        best_score = float("inf")
        for eid in cached_eids:
            if protected_eids and eid in protected_eids:
                continue
            score = ema[eid].item()
            if score < best_score:
                best_score = score
                best_victim = eid
        # Fallback: if all are protected, pick global lowest
        if best_victim is None:
            for eid in cached_eids:
                score = ema[eid].item()
                if score < best_score:
                    best_score = score
                    best_victim = eid
        return best_victim if best_victim is not None else cached_eids[0]

    def get_top_k(self, layer_id: int, k: int) -> list[int]:
        """Return expert IDs with the K highest EMA scores for this layer."""
        ema = self._get_ema(layer_id)
        _, indices = ema.topk(min(k, ema.shape[0]))
        return indices.tolist()

    def get_priority(self, layer_id: int, expert_id: int) -> float:
        """Get current EMA priority for an expert."""
        ema = self._get_ema(layer_id)
        return ema[expert_id].item()
