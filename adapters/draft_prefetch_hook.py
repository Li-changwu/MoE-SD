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
            # v3.1: Overflow controller (C1-C4 + C3 K adjustment)
            hook._run_overflow_controller(self_proposer)
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

        When DIPP is enabled on the ELMM manager, uses value-based priority
        scheduling instead of simple last-step locality.
        """
        self._draft_rounds += 1

        elmm = self._elmm
        if not elmm._installed:
            return

        # Only prefetch every N rounds to avoid excessive PCIe contention
        if self._draft_rounds % 3 != 0:
            return

        # === PredCache path: demand-prioritized prefetch scheduling ===
        if elmm._pred_cache is not None and elmm._briskmoe_dipp is None:
            pred_cache = elmm._pred_cache
            # Build cache state for each layer
            cache_states: dict[int, set[int]] = {}
            for layer_name, cache in elmm._layer_caches.items():
                layer_id = elmm._layer_name_to_id.get(layer_name, 0)
                cache_states[layer_id] = set(cache._slot_map.keys())

            num_layers = len(elmm._layer_caches)
            schedule = pred_cache.compute_prefetch_schedule(
                cache_states, num_layers
            )

            if schedule:
                id_to_name = {v: k for k, v in elmm._layer_name_to_id.items()}
                for layer_id, eid, _priority in schedule:
                    layer_name = id_to_name.get(layer_id)
                    if layer_name:
                        elmm.prefetch_experts(layer_name, [eid])
            return

        # === DIPP path: value-based prefetch scheduling ===
        # When PredCache is also enabled, DIPP's Value is enhanced with
        # EMA demand signal: hot experts get priority over one-off misses.
        if elmm._briskmoe_dipp is not None:
            dipp = elmm._briskmoe_dipp
            # Build predictions: {layer_id: {token_pos=0: [expert_ids]}}
            # Use last step's expert set as surrogate for draft predictions
            predictions: dict[int, dict[int, list[int]]] = {}
            cache_state: dict[int, set[int]] = {}
            for layer_name, last_experts in elmm._last_expert_set.items():
                layer_id = elmm._layer_name_to_id.get(layer_name, 0)
                if last_experts:
                    predictions[layer_id] = {0: list(last_experts)}
                cache = elmm._layer_caches.get(layer_name)
                if cache:
                    cache_state[layer_id] = set(cache._slot_map.keys())

            if predictions:
                schedule = dipp.compute_schedule(predictions, cache_state)
                # Enhance DIPP with PredCache EMA demand if available.
                # Re-score each entry: V' = V × (1 + ema_demand) so hot
                # experts are prefetched before one-off misses.
                pred_cache = elmm._pred_cache
                if pred_cache is not None and schedule:
                    boosted: list[tuple[float, int, int]] = []
                    for layer_id, eid, val in schedule:
                        d = pred_cache.get_demand_boost(layer_id, eid)
                        boosted.append((val * (1.0 + d), layer_id, eid))
                    boosted.sort(reverse=True)
                    schedule = [
                        (lid, eid, v) for v, lid, eid in boosted
                    ]
                # Build layer_id → layer_name reverse mapping
                id_to_name = {v: k for k, v in elmm._layer_name_to_id.items()}
                for layer_id, eid, _val in schedule:
                    layer_name = id_to_name.get(layer_id)
                    if layer_name:
                        elmm.prefetch_experts(layer_name, [eid])
                dipp.reset_round()
            return

        # === Legacy path: locality-based prefetch ===
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

    def _run_overflow_controller(self, proposer: Any):
        """v3.1: Run C1→C2→C3→C4→C5 after draft completes.

        C3 adjusts proposer.num_speculative_tokens for the NEXT step.
        C4/C5 act on the CURRENT step (prefetch + pin).
        """
        elmm = self._elmm
        controller = getattr(elmm, '_overflow_controller', None)
        if controller is None or not controller.enabled:
            return

        # Build draft_routing from _last_expert_set
        # Format: {layer_idx: [expert_ids]}
        draft_routing: dict[int, list[int]] = {}
        for layer_name, experts in elmm._last_expert_set.items():
            if experts:
                layer_id = elmm._layer_name_to_id.get(layer_name, -1)
                if layer_id >= 0:
                    draft_routing[layer_id] = list(experts)

        if not draft_routing:
            return

        # Run C1→C2→C3→C4→C5
        report = controller.on_draft_complete(draft_routing)

        # C3: K adjustment tracking (for metrics only).
        # NOTE: Modifying proposer.num_speculative_tokens at runtime is
        # unsafe — vLLM pre-allocates draft_token_ids_cpu, attention masks,
        # and KV-cache slots at init time with the original K. Changing K
        # mid-stream causes CUDA index-out-of-bounds in IndexKernel.cu.
        # For now, C3 computes the recommended K but does NOT apply it.
        # Future: integrate with vLLM's SpecConfig.update() if available.
        _ = controller.get_recommended_K()  # tracked internally by C3

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
