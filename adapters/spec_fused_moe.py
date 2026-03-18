"""
SpecFusedMoE — Speculative-Decoding-Aware Fused MoE Operator
=============================================================
Key insight: During SD verify, K+1 tokens are processed simultaneously.
Each token routes to top-k experts. Vanilla fused_moe loads experts
independently per token, causing MAF = K+1 load amplification.

SpecFusedMoE deduplicates expert loads across the verify batch:
  actual_loads = |∪_{i=0}^{K} E_i^(l)|  (instead of (K+1)×k)

This module provides:
1. `SpecFusedMoE` — Python wrapper that deduplicates before dispatching
2. MAF-aware priority ordering by acceptance probability
3. Early-abort support for rejected draft tokens
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpecFusedMoE(torch.nn.Module):
    """
    SD-aware fused MoE operator with cross-token expert deduplication.
    
    During verify phase, K+1 tokens share overlapping experts. This module:
    1. Collects top-k experts across all tokens in the verify batch
    2. Computes the unique expert set (dedup)
    3. Loads each unique expert exactly once
    4. Dispatches tokens to their respective experts
    5. Returns the same output as vanilla fused_moe
    """

    def __init__(
        self,
        num_experts: int = 128,
        top_k: int = 8,
        hidden_size: int = 2048,
        moe_intermediate_size: int = 768,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob

        # Statistics
        self._total_naive_loads = 0
        self._total_dedup_loads = 0
        self._total_calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,       # [batch_size, hidden_size]
        router_logits: torch.Tensor,        # [batch_size, num_experts]
        expert_weights: dict,               # expert_id -> {gate_proj, up_proj, down_proj}
        acceptance_probs: Optional[torch.Tensor] = None,  # [batch_size] probability each token is accepted
        frozen_mask: Optional[torch.Tensor] = None,       # [batch_size] bool: True = skip MoE
    ) -> torch.Tensor:
        """
        Forward pass with cross-token expert deduplication.

        Returns:
            output: [batch_size, hidden_size]
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Step 1: Compute routing (top-k experts per token)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # top_indices: [batch_size, top_k]
        # top_weights: [batch_size, top_k]

        if self.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Step 2: Handle frozen tokens (early termination)
        if frozen_mask is not None:
            active_mask = ~frozen_mask
        else:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        active_indices = active_mask.nonzero(as_tuple=True)[0]
        if len(active_indices) == 0:
            return hidden_states.clone()

        active_hidden = hidden_states[active_indices]
        active_top_indices = top_indices[active_indices]
        active_top_weights = top_weights[active_indices]
        active_batch = active_hidden.shape[0]

        # Step 3: Expert deduplication — compute unique expert set
        all_expert_ids = active_top_indices.reshape(-1).unique()
        naive_loads = active_batch * self.top_k
        dedup_loads = len(all_expert_ids)

        self._total_naive_loads += naive_loads
        self._total_dedup_loads += dedup_loads
        self._total_calls += 1

        # Step 4: Priority ordering (optional, for pipelined loading)
        if acceptance_probs is not None:
            # Sort tokens by acceptance probability (high first)
            active_probs = acceptance_probs[active_indices]
            sorted_order = torch.argsort(active_probs, descending=True)
            # Reorder: process high-probability tokens first
            # This enables early abort if a low-prob token is rejected mid-verify
            active_hidden = active_hidden[sorted_order]
            active_top_indices = active_top_indices[sorted_order]
            active_top_weights = active_top_weights[sorted_order]

        # Step 5: Compute MoE output with deduplication
        output = torch.zeros_like(active_hidden)

        # Create expert-to-token dispatch table
        # For each unique expert, find which (token, slot) pairs need it
        expert_dispatch = {}  # expert_id -> list of (token_idx_in_active, slot_idx, weight)
        for t in range(active_batch):
            for s in range(self.top_k):
                eid = active_top_indices[t, s].item()
                if eid not in expert_dispatch:
                    expert_dispatch[eid] = []
                expert_dispatch[eid].append((t, s, active_top_weights[t, s]))

        # Process each unique expert exactly once
        for expert_id in all_expert_ids.tolist():
            if expert_id not in expert_weights:
                continue

            params = expert_weights[expert_id]
            dispatch_list = expert_dispatch.get(expert_id, [])
            if not dispatch_list:
                continue

            # Gather tokens that use this expert
            token_indices = [d[0] for d in dispatch_list]
            weights = torch.tensor([d[2].item() for d in dispatch_list],
                                   device=device, dtype=active_hidden.dtype)

            # Batch compute: expert(hidden[tokens])
            expert_input = active_hidden[token_indices]  # [num_tokens_using_expert, hidden_size]

            # SiLU-gated FFN: out = down_proj(silu(gate_proj(x)) * up_proj(x))
            gate_out = F.silu(expert_input @ params["gate_proj"].T)
            up_out = expert_input @ params["up_proj"].T
            expert_out = (gate_out * up_out) @ params["down_proj"].T  # [n, hidden_size]

            # Weighted accumulation
            for i, (t_idx, s_idx, w) in enumerate(dispatch_list):
                output[t_idx] += w * expert_out[i]

        # Step 6: Write back results
        if acceptance_probs is not None:
            # Unsort
            inv_order = torch.argsort(sorted_order)
            output = output[inv_order]

        result = hidden_states.clone()
        result[active_indices] = output

        return result

    @property
    def dedup_ratio(self) -> float:
        """Fraction of expert loads saved by deduplication."""
        if self._total_naive_loads == 0:
            return 0.0
        return 1.0 - self._total_dedup_loads / self._total_naive_loads

    @property
    def effective_maf(self) -> float:
        """Average MAF = dedup_loads / top_k per call."""
        if self._total_calls == 0:
            return 0.0
        return self._total_dedup_loads / (self._total_calls * self.top_k)

    def get_statistics(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_naive_loads": self._total_naive_loads,
            "total_dedup_loads": self._total_dedup_loads,
            "dedup_ratio": round(self.dedup_ratio, 4),
            "effective_maf": round(self.effective_maf, 4),
        }

    def reset_statistics(self):
        self._total_naive_loads = 0
        self._total_dedup_loads = 0
        self._total_calls = 0


class DedupAnalyzer:
    """
    Analyze potential expert deduplication savings from routing trace data.
    Does not require actual expert weights — works with trace data only.
    """

    def __init__(self, num_experts: int = 128, top_k: int = 8):
        self.num_experts = num_experts
        self.top_k = top_k

    def analyze_verify_batch(
        self,
        expert_indices: torch.Tensor,  # [K+1, top_k] expert indices per token
    ) -> dict:
        """
        Analyze dedup potential for a single verify batch.
        
        Args:
            expert_indices: [K+1, top_k] tensor of expert indices
            
        Returns:
            Dict with naive_loads, dedup_loads, savings, maf
        """
        K_plus_1 = expert_indices.shape[0]
        naive_loads = K_plus_1 * self.top_k
        unique_experts = expert_indices.reshape(-1).unique()
        dedup_loads = len(unique_experts)

        return {
            "K": K_plus_1 - 1,
            "naive_loads": naive_loads,
            "dedup_loads": dedup_loads,
            "savings_pct": round((1 - dedup_loads / naive_loads) * 100, 2),
            "maf": round(dedup_loads / self.top_k, 4),
            "theoretical_max_savings": round(
                (1 - self.num_experts * (1 - (1 - self.top_k / self.num_experts) ** K_plus_1) / naive_loads) * 100, 2
            ),
        }

    def analyze_from_trace(self, events: list[dict], K: int) -> dict:
        """
        Analyze dedup savings from trace data, simulating verify windows.
        
        Args:
            events: List of trace event dicts (from ExpertTraceCollector)
            K: Number of speculative tokens
            
        Returns:
            Aggregate statistics over all windows
        """
        from collections import defaultdict
        import numpy as np

        grouped = defaultdict(list)
        for e in events:
            if e.get("phase", "decode") == "decode":
                key = (e["request_id"], e["layer_id"])
                grouped[key].append(e)

        all_savings = []
        all_mafs = []

        for (req_id, layer_id), layer_events in grouped.items():
            layer_events.sort(key=lambda x: x["token_idx"])

            for start in range(len(layer_events) - K):
                window = layer_events[start: start + K + 1]
                all_experts_in_window = set()
                for evt in window:
                    all_experts_in_window.update(evt["experts"][:self.top_k])

                naive = (K + 1) * self.top_k
                dedup = len(all_experts_in_window)
                savings = 1 - dedup / naive
                maf = dedup / self.top_k

                all_savings.append(savings)
                all_mafs.append(maf)

        if not all_savings:
            return {"error": "No valid windows found"}

        arr_s = np.array(all_savings)
        arr_m = np.array(all_mafs)

        return {
            "K": K,
            "num_windows": len(all_savings),
            "mean_savings_pct": round(float(np.mean(arr_s)) * 100, 2),
            "mean_maf": round(float(np.mean(arr_m)), 4),
            "std_maf": round(float(np.std(arr_m)), 4),
            "p25_maf": round(float(np.percentile(arr_m, 25)), 4),
            "p75_maf": round(float(np.percentile(arr_m, 75)), 4),
            "max_savings_pct": round(float(np.max(arr_s)) * 100, 2),
            "min_savings_pct": round(float(np.min(arr_s)) * 100, 2),
        }
