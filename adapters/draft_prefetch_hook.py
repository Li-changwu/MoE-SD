"""
Draft-Guided Prefetch Hook for EAGLE-3
========================================
Intercepts draft model's router decisions during the draft phase
of speculative decoding to pre-load predicted experts into ELMM's
GPU cache, overlapping PCIe transfers with draft computation.

Key insight: The draft model (EAGLE-3) runs BEFORE the verify phase.
Its router decisions predict which experts the target MoE model will
activate. By prefetching those experts during draft generation, we
can convert many verify-phase cache misses into cache hits.

Integration:
  - Hooks into vLLM's EAGLE-3 draft worker's propose() method
  - Captures router logits from the target model's MoE layers
  - Forwards predicted expert IDs to ELMM's prefetch_for_draft_routing()
  - ELMM does async H2D on a separate CUDA stream

Activated via ELMM plugin when enable_prefetch=True.
"""

import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DraftPrefetchHook:
    """
    Hooks into the EAGLE-3 draft model to capture routing predictions
    and trigger expert prefetch on the ELMM manager.
    """

    def __init__(self, elmm_manager: Any, top_k: int = 8):
        self._elmm = elmm_manager
        self._top_k = top_k
        self._installed = False
        self._original_propose: Optional[Any] = None
        self._draft_rounds = 0

    def install(self) -> bool:
        """
        Monkey-patch the EAGLE-3 draft worker's propose() to intercept
        draft routing and trigger prefetch.

        Returns True if successfully installed.
        """
        try:
            from vllm.v1.spec_decode.eagle import EagleProposer
        except ImportError:
            print("[DraftPrefetch] Cannot import EagleProposer, skipping",
                  file=sys.stderr, flush=True)
            return False

        self._original_propose = EagleProposer.propose
        hook = self

        def patched_propose(self_proposer, *args, **kwargs):
            result = hook._original_propose(self_proposer, *args, **kwargs)
            # After draft tokens are proposed, trigger prefetch
            hook._on_draft_complete(self_proposer)
            return result

        EagleProposer.propose = patched_propose
        self._installed = True
        print("[DraftPrefetch] Installed on EagleProposer.propose",
              file=sys.stderr, flush=True)
        return True

    def _on_draft_complete(self, proposer: Any):
        """
        Called after each draft proposal. Triggers selective prefetch
        based on temporal locality — only for layers with low cache hit rates.
        """
        self._draft_rounds += 1

        elmm = self._elmm
        if not elmm._installed:
            return

        # Only prefetch every N rounds to avoid excessive PCIe contention
        if self._draft_rounds % 3 != 0:
            return

        # Select only layers with low hit rates (where prefetch is valuable)
        # Limit to at most 5 layers per draft round to cap PCIe usage
        layer_scores = []
        for layer_name, ema_rate in elmm._hit_rate_ema.items():
            if ema_rate < 0.8 and layer_name in elmm._last_expert_set:
                layer_scores.append((ema_rate, layer_name))

        # Sort by hit rate ascending (lowest first = most benefit from prefetch)
        layer_scores.sort()
        layers_to_prefetch = layer_scores[:5]

        for _score, layer_name in layers_to_prefetch:
            last_experts = elmm._last_expert_set.get(layer_name)
            if last_experts:
                # Only prefetch experts not already cached
                cache = elmm._layer_caches.get(layer_name)
                if cache:
                    uncached = [e for e in last_experts if not cache.contains(e)]
                    if uncached:
                        elmm.prefetch_experts(layer_name, uncached[:8])

    def uninstall(self):
        """Restore original propose method."""
        if self._installed and self._original_propose is not None:
            try:
                from vllm.v1.spec_decode.eagle import EagleProposer
                EagleProposer.propose = self._original_propose
                self._installed = False
            except ImportError:
                pass

    def get_stats(self) -> dict:
        return {
            "installed": self._installed,
            "draft_rounds": self._draft_rounds,
        }


def install_draft_prefetch(elmm_manager: Any, top_k: int = 8) -> Optional[DraftPrefetchHook]:
    """
    Install draft-guided prefetch hook if enabled.

    Args:
        elmm_manager: The active ELMMManager instance
        top_k: Number of experts per token

    Returns:
        DraftPrefetchHook if installed, None otherwise
    """
    if not elmm_manager or not elmm_manager.config.enable_prefetch:
        return None

    hook = DraftPrefetchHook(elmm_manager, top_k=top_k)
    if hook.install():
        return hook
    return None
