"""
Speculation Divergence Detector (SDD) — Layer-wise Early Termination
====================================================================
Detects draft token rejection early by monitoring router logits divergence
across MoE layers during target model verify.

When a draft token's router distribution significantly diverges from
expected patterns, it's likely to be rejected. We can stop processing
MoE FFN for that token in remaining layers, reducing effective MAF.

Metrics used:
  - KL divergence of router logits between consecutive layers
  - Top-k expert overlap between draft routing prediction and target routing
  - Router entropy anomaly (low entropy → concentrated routing → confident)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SDDConfig:
    """Configuration for Speculation Divergence Detector."""
    # Minimum layer to start checking (skip early layers)
    min_check_layer: int = 8
    # KL divergence threshold for detecting divergence
    kl_threshold: float = 2.0
    # Number of consecutive divergent layers before termination
    consecutive_threshold: int = 3
    # Entropy threshold (low entropy = confident routing, unlikely to match draft)
    entropy_threshold: float = 1.5
    # Top-k overlap threshold (low overlap = high divergence)
    topk_overlap_threshold: float = 0.25
    # Method: "kl", "overlap", "entropy", "combined"
    method: str = "combined"


@dataclass
class TokenDivergenceState:
    """Track divergence state for a single draft token across layers."""
    token_idx: int
    consecutive_divergent: int = 0
    total_divergent: int = 0
    layers_checked: int = 0
    frozen: bool = False
    frozen_at_layer: int = -1
    divergence_scores: list = field(default_factory=list)


class SpeculationDivergenceDetector:
    """
    Monitors router logits during target model verify to detect
    divergent draft tokens that should be terminated early.
    """

    def __init__(self, config: Optional[SDDConfig] = None, num_layers: int = 48):
        self.config = config or SDDConfig()
        self.num_layers = num_layers
        self._token_states: dict[int, TokenDivergenceState] = {}
        self._draft_routing: Optional[dict] = None  # layer -> token -> top_k experts

        # Statistics
        self._total_checks = 0
        self._total_freezes = 0
        self._freeze_layer_sum = 0
        self._true_positive = 0
        self._false_positive = 0
        self._true_negative = 0
        self._false_negative = 0

    def init_verify_round(self, num_draft_tokens: int):
        """Initialize state for a new verify round with K draft tokens."""
        self._token_states = {
            i: TokenDivergenceState(token_idx=i)
            for i in range(num_draft_tokens)
        }

    def set_draft_routing(self, draft_routing: dict):
        """
        Set the draft model's predicted routing for comparison.

        Args:
            draft_routing: {layer_id: {token_idx: [expert_ids]}}
        """
        self._draft_routing = draft_routing

    def check_layer(
        self,
        layer_id: int,
        router_logits: torch.Tensor,  # [batch_size, num_experts] — target model logits
        token_indices: list[int],      # which tokens these logits correspond to
    ) -> torch.Tensor:
        """
        Check divergence at a given layer for all draft tokens.

        Args:
            layer_id: Current layer index
            router_logits: [batch, num_experts] target model router output
            token_indices: Token indices corresponding to each batch element

        Returns:
            frozen_mask: [batch] bool tensor, True = should freeze this token
        """
        batch_size = router_logits.shape[0]
        frozen_mask = torch.zeros(batch_size, dtype=torch.bool, device=router_logits.device)

        if layer_id < self.config.min_check_layer:
            return frozen_mask

        target_probs = F.softmax(router_logits.float(), dim=-1)

        for b_idx, t_idx in enumerate(token_indices):
            if t_idx not in self._token_states:
                continue

            state = self._token_states[t_idx]
            if state.frozen:
                frozen_mask[b_idx] = True
                continue

            # Compute divergence score
            score = self._compute_divergence(
                target_probs[b_idx],
                layer_id,
                t_idx,
            )

            state.divergence_scores.append((layer_id, score))
            state.layers_checked += 1
            self._total_checks += 1

            # Check thresholds
            is_divergent = self._is_divergent(score)

            if is_divergent:
                state.consecutive_divergent += 1
                state.total_divergent += 1
            else:
                state.consecutive_divergent = 0

            # Freeze decision
            if state.consecutive_divergent >= self.config.consecutive_threshold:
                state.frozen = True
                state.frozen_at_layer = layer_id
                frozen_mask[b_idx] = True
                self._total_freezes += 1
                self._freeze_layer_sum += layer_id
                logger.debug(f"Token {t_idx} frozen at layer {layer_id} "
                             f"(consecutive={state.consecutive_divergent})")

        return frozen_mask

    def _compute_divergence(
        self,
        target_probs: torch.Tensor,  # [num_experts]
        layer_id: int,
        token_idx: int,
    ) -> float:
        """Compute divergence score for a token at a specific layer."""
        method = self.config.method

        if method == "entropy":
            return self._entropy_divergence(target_probs)
        elif method == "overlap":
            return self._overlap_divergence(target_probs, layer_id, token_idx)
        elif method == "kl":
            return self._kl_divergence(target_probs, layer_id, token_idx)
        elif method == "combined":
            scores = []
            scores.append(self._entropy_divergence(target_probs))
            if self._draft_routing:
                scores.append(self._overlap_divergence(target_probs, layer_id, token_idx))
            # Normalize and average
            return sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown SDD method: {method}")

    def _entropy_divergence(self, target_probs: torch.Tensor) -> float:
        """
        Low entropy in target routing means concentrated distribution.
        If the target is very confident, it's more likely to diverge from draft.
        We use negative entropy as divergence signal (higher = more divergent).
        """
        entropy = -(target_probs * (target_probs + 1e-10).log()).sum().item()
        # Normalize: uniform distribution has max entropy
        max_entropy = torch.tensor(target_probs.shape[0], dtype=torch.float).log().item()
        normalized_entropy = entropy / max_entropy
        # Invert: low entropy → high divergence score
        return 1.0 - normalized_entropy

    def _overlap_divergence(
        self,
        target_probs: torch.Tensor,
        layer_id: int,
        token_idx: int,
    ) -> float:
        """
        Compare top-k experts between draft prediction and target routing.
        Low overlap → high divergence.
        """
        if not self._draft_routing or layer_id not in self._draft_routing:
            return 0.0

        layer_routing = self._draft_routing[layer_id]
        if token_idx not in layer_routing:
            return 0.0

        draft_experts = set(layer_routing[token_idx])
        target_top_k = target_probs.topk(len(draft_experts)).indices.tolist()
        target_experts = set(target_top_k)

        if not draft_experts or not target_experts:
            return 1.0

        overlap = len(draft_experts & target_experts) / len(draft_experts | target_experts)
        return 1.0 - overlap  # High divergence = low overlap

    def _kl_divergence(
        self,
        target_probs: torch.Tensor,
        layer_id: int,
        token_idx: int,
    ) -> float:
        """
        KL divergence between target and draft routing distributions.
        Requires draft routing probabilities (not just indices).
        Falls back to entropy if draft probs unavailable.
        """
        # For now, use entropy as proxy since we may not have draft probs
        return self._entropy_divergence(target_probs)

    def _is_divergent(self, score: float) -> bool:
        """Determine if a score indicates divergence."""
        method = self.config.method
        if method == "entropy":
            return score > (1.0 - self.config.entropy_threshold / 5.0)
        elif method == "overlap":
            return score > (1.0 - self.config.topk_overlap_threshold)
        elif method == "kl":
            return score > 0.5
        elif method == "combined":
            return score > 0.6
        return False

    def get_frozen_mask(self) -> dict[int, bool]:
        """Get current frozen state for all tracked tokens."""
        return {t_idx: state.frozen for t_idx, state in self._token_states.items()}

    def report_acceptance(self, token_idx: int, accepted: bool):
        """
        Report actual acceptance/rejection for evaluation metrics.
        Call after verify is complete.
        """
        if token_idx not in self._token_states:
            return

        state = self._token_states[token_idx]
        if state.frozen and not accepted:
            self._true_positive += 1  # Correctly predicted rejection
        elif state.frozen and accepted:
            self._false_positive += 1  # Wrong: froze an accepted token
        elif not state.frozen and not accepted:
            self._false_negative += 1  # Missed: should have frozen
        else:
            self._true_negative += 1  # Correctly continued

    def get_statistics(self) -> dict:
        total_predictions = (
            self._true_positive + self._false_positive +
            self._true_negative + self._false_negative
        )
        precision = (
            self._true_positive / (self._true_positive + self._false_positive)
            if (self._true_positive + self._false_positive) > 0
            else 0.0
        )
        recall = (
            self._true_positive / (self._true_positive + self._false_negative)
            if (self._true_positive + self._false_negative) > 0
            else 0.0
        )
        avg_freeze_layer = (
            self._freeze_layer_sum / self._total_freezes
            if self._total_freezes > 0
            else self.num_layers
        )

        return {
            "total_checks": self._total_checks,
            "total_freezes": self._total_freezes,
            "avg_freeze_layer": round(avg_freeze_layer, 1),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0, 4),
            "true_positive": self._true_positive,
            "false_positive": self._false_positive,
            "true_negative": self._true_negative,
            "false_negative": self._false_negative,
        }

    def estimate_maf_reduction(self, original_maf: float, K: int) -> dict:
        """
        Estimate MAF reduction from early termination.

        If we freeze a token at layer l*, we save (L - l*) layers of expert loading
        for that token's marginal experts.
        """
        if self._total_freezes == 0:
            return {
                "original_maf": original_maf,
                "reduced_maf": original_maf,
                "reduction_pct": 0.0,
            }

        avg_freeze_layer = self._freeze_layer_sum / self._total_freezes
        freeze_rate = self._total_freezes / max(1, len(self._token_states))

        # Fraction of layers saved per frozen token
        layer_savings = (self.num_layers - avg_freeze_layer) / self.num_layers

        # MAF reduction ≈ freeze_rate × marginal_contribution × layer_savings
        # Marginal contribution of one token ≈ (MAF - (MAF without that token))
        # Approximate: marginal ≈ (MAF - 1) / K for K draft tokens
        marginal_per_token = (original_maf - 1.0) / K if K > 0 else 0
        maf_reduction = freeze_rate * marginal_per_token * layer_savings
        reduced_maf = original_maf - maf_reduction

        return {
            "original_maf": round(original_maf, 4),
            "reduced_maf": round(max(1.0, reduced_maf), 4),
            "reduction_pct": round(maf_reduction / original_maf * 100, 2),
            "freeze_rate": round(freeze_rate, 4),
            "avg_freeze_layer": round(avg_freeze_layer, 1),
            "layer_savings_frac": round(layer_savings, 4),
        }
