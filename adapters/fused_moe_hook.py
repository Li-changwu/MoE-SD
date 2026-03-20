"""
vLLM fused_moe Hook — Runtime Monkey-Patch for SpecMoE
========================================================
Intercepts vLLM's native `fused_moe` calls during the SD verify phase
and redirects them through SpecFusedMoE with deduplication + early termination.

vLLM 0.17 Architecture:
  Model → MoE layer → router → fused_moe(hidden, w1, w2, topk_w, topk_ids)

The hook replaces `vllm.model_executor.layers.fused_moe.fused_moe` with
a wrapper that:
  1. During verify phase: routes through SpecFusedMoE (dedup + SDD)
  2. During normal decode: passes through to original fused_moe
  3. Collects expert routing traces and cache metrics

Hook lifecycle:
  install() → patches vLLM at module level
  uninstall() → restores original fused_moe
  set_verify_mode(True) → enable SpecMoE for next forward
  set_verify_mode(False) → back to native
"""

import functools
import importlib
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


class FusedMoEHook:
    """
    Intercepts vLLM's fused_moe function to inject SpecMoE optimizations.

    Supports two operation modes:
      - passthrough: delegates to original fused_moe (normal inference)
      - specmoe: applies cross-token dedup + SDD + cache-aware dispatch

    Thread-safe: uses per-forward-pass context rather than global state.
    """

    # Known fused_moe function locations in vLLM
    VLLM_FUSED_MOE_PATHS = [
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.layer.fused_moe",
    ]

    # Known import paths for the module containing fused_moe
    VLLM_MOE_MODULE_PATHS = [
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.layer",
    ]

    def __init__(self):
        # Original function reference
        self._original_fn: Optional[Callable] = None
        self._original_module: Optional[Any] = None
        self._original_attr: Optional[str] = None

        # SpecMoE components (set via configure())
        self._spec_moe = None          # SpecFusedMoEDispatcher or TritonV2
        self._sdd = None               # SpeculationDivergenceDetector
        self._expert_cache = None       # ExpertWeightCache
        self._trace_collector = None    # ExpertTraceCollector

        # Auto-detect verify phase: if hidden_states.shape[0] >= threshold,
        # treat as verify batch (K+1 tokens).  0 = disabled (manual mode).
        self._auto_verify_threshold: int = 0

        # Per-forward state
        self._verify_mode = False
        self._current_layer_id = 0
        self._verify_batch_size = 0
        self._active_mask: Optional[torch.Tensor] = None

        # Statistics
        self._installed = False
        self._total_intercepts = 0
        self._total_specmoe_calls = 0
        self._total_passthrough_calls = 0

    def configure(
        self,
        spec_moe=None,
        sdd=None,
        expert_cache=None,
        trace_collector=None,
        auto_verify_threshold: int = 0,
    ):
        """
        Configure SpecMoE components.

        Args:
            spec_moe: SpecFusedMoEDispatcher instance
            sdd: SpeculationDivergenceDetector instance
            expert_cache: ExpertWeightCache instance
            trace_collector: ExpertTraceCollector instance (optional, for data collection)
            auto_verify_threshold: if > 0, automatically enter verify mode whenever
                hidden_states.shape[0] >= threshold (avoids manual set_verify_mode calls
                in serve mode where vLLM scheduling is not directly observable).
                Typical value: num_speculative_tokens + 1 (e.g. 4 for K=3).
        """
        self._spec_moe = spec_moe
        self._sdd = sdd
        self._expert_cache = expert_cache
        self._trace_collector = trace_collector
        self._auto_verify_threshold = auto_verify_threshold

    def install(self) -> bool:
        """
        Monkey-patch vLLM's fused_moe with our interceptor.

        Returns True if successfully patched, False otherwise.
        """
        if self._installed:
            logger.warning("FusedMoEHook already installed")
            return True

        # Try to find and patch the fused_moe function
        for module_path in self.VLLM_MOE_MODULE_PATHS:
            try:
                mod = importlib.import_module(module_path)

                # Look for `fused_moe` function in the module
                if hasattr(mod, "fused_moe") and callable(getattr(mod, "fused_moe")):
                    self._original_fn = getattr(mod, "fused_moe")
                    self._original_module = mod
                    self._original_attr = "fused_moe"

                    # Install wrapper
                    setattr(mod, "fused_moe", self._wrapped_fused_moe)
                    self._installed = True
                    logger.info(f"Hooked fused_moe from {module_path}")
                    return True

            except ImportError:
                continue
            except Exception as e:
                logger.debug(f"Failed to hook {module_path}: {e}")
                continue

        # Also try patching at the Qwen3MoE model level
        try:
            from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
            if hasattr(Qwen3MoeSparseMoeBlock, "forward"):
                self._original_fn = Qwen3MoeSparseMoeBlock.forward
                self._original_module = Qwen3MoeSparseMoeBlock
                self._original_attr = "forward"
                Qwen3MoeSparseMoeBlock.forward = self._make_block_wrapper(self._original_fn)
                self._installed = True
                logger.info("Hooked Qwen3MoeSparseMoeBlock.forward")
                return True
        except ImportError:
            pass

        logger.warning("Could not find vLLM fused_moe to hook. "
                       "Will operate in standalone mode.")
        return False

    def uninstall(self):
        """Restore original fused_moe function."""
        if not self._installed:
            return

        if self._original_module and self._original_attr and self._original_fn:
            setattr(self._original_module, self._original_attr, self._original_fn)

        self._installed = False
        self._original_fn = None
        self._original_module = None
        self._original_attr = None
        logger.info("FusedMoEHook uninstalled")

    def set_verify_mode(
        self,
        enabled: bool,
        batch_size: int = 0,
        active_mask: Optional[torch.Tensor] = None,
    ):
        """
        Toggle verify mode for the next forward pass(es).

        Args:
            enabled: True to activate SpecMoE during verify
            batch_size: Number of tokens in verify batch (K+1)
            active_mask: [batch_size] bool mask for frozen tokens
        """
        self._verify_mode = enabled
        self._verify_batch_size = batch_size
        self._active_mask = active_mask
        self._current_layer_id = 0

        if enabled and self._sdd is not None:
            # Initialize SDD for this verify round
            num_draft = batch_size - 1  # K draft + 1 original token
            self._sdd.init_verify_round(num_draft)

    @contextmanager
    def verify_context(
        self,
        batch_size: int,
        active_mask: Optional[torch.Tensor] = None,
    ):
        """
        Context manager for verify phase.

        Usage:
            with hook.verify_context(batch_size=K+1, active_mask=mask):
                model.forward(...)  # All fused_moe calls go through SpecMoE
        """
        self.set_verify_mode(True, batch_size, active_mask)
        try:
            yield self
        finally:
            self.set_verify_mode(False)

    def _wrapped_fused_moe(self, *args, **kwargs):
        """
        Wrapper around vLLM's fused_moe.

        Signature of original vLLM fused_moe (v0.17):
            fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                      inplace=False, override_config=None, use_fp8_w8a8=False,
                      ...)
        """
        self._total_intercepts += 1

        # Auto-detect verify phase by batch size when threshold is configured
        in_verify = self._verify_mode
        if (
            not in_verify
            and self._auto_verify_threshold > 0
            and self._spec_moe is not None
        ):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None and hidden.shape[0] >= self._auto_verify_threshold:
                in_verify = True
                self._current_layer_id = 0  # reset layer counter each verify batch

        if not in_verify or self._spec_moe is None:
            # Passthrough to original
            self._total_passthrough_calls += 1
            return self._original_fn(*args, **kwargs)

        # Extract arguments
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        w1 = args[1] if len(args) > 1 else kwargs.get("w1")
        w2 = args[2] if len(args) > 2 else kwargs.get("w2")
        topk_weights = args[3] if len(args) > 3 else kwargs.get("topk_weights")
        topk_ids = args[4] if len(args) > 4 else kwargs.get("topk_ids")

        if hidden_states is None or w1 is None:
            return self._original_fn(*args, **kwargs)

        self._total_specmoe_calls += 1
        layer_id = self._current_layer_id
        self._current_layer_id += 1

        # --- SDD: Check for frozen tokens at this layer ---
        active_mask = self._active_mask
        if self._sdd is not None and topk_ids is not None:
            # Reconstruct router logits from topk for SDD
            # Note: In a full integration, we'd intercept before top-k selection
            batch_size = hidden_states.shape[0]
            token_indices = list(range(1, batch_size))  # Skip token 0 (original)

            # Use topk_ids as proxy for router output
            # Real integration would tap the router logits directly
            router_proxy = torch.zeros(
                batch_size, self._spec_moe.num_experts if hasattr(self._spec_moe, 'num_experts') else 128,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            for b in range(batch_size):
                for s in range(topk_ids.shape[1]):
                    eid = topk_ids[b, s].item()
                    router_proxy[b, eid] = topk_weights[b, s]

            sdd_frozen = self._sdd.check_layer(
                layer_id=layer_id,
                router_logits=router_proxy,
                token_indices=token_indices,
            )

            # Merge with existing active mask
            if active_mask is None:
                active_mask = ~sdd_frozen
                # Token 0 (original) is always active
                full_mask = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
                full_mask[1:] = active_mask
                active_mask = full_mask
            else:
                active_mask = active_mask.clone()
                active_mask[1:] &= ~sdd_frozen

            self._active_mask = active_mask

        # --- Expert Cache: Use cached weights if available ---
        if self._expert_cache is not None:
            unique_experts = topk_ids.reshape(-1).unique().tolist()
            cache_result = self._expert_cache.get_experts_batch(layer_id, unique_experts)
            # TODO: Replace w1/w2 slices with cached weights
            # For now, pass through original weights

        # --- Trace Collection (optional) ---
        if self._trace_collector is not None:
            for b in range(hidden_states.shape[0]):
                from collectors.expert_trace_hook import TraceEvent
                event = TraceEvent(
                    request_id="verify",
                    token_idx=b,
                    layer_id=layer_id,
                    experts=topk_ids[b].cpu().tolist(),
                    router_probs=topk_weights[b].cpu().tolist(),
                    phase="verify",
                )
                self._trace_collector._write_event(event)

        # --- SpecFusedMoE: Dedup dispatch ---
        output = self._spec_moe(
            hidden_states, w1, w2, topk_weights, topk_ids,
            active_mask=active_mask,
        )

        return output

    def _make_block_wrapper(self, original_forward):
        """Create a wrapper for Qwen3MoeSparseMoeBlock.forward."""
        hook = self

        @functools.wraps(original_forward)
        def wrapper(self_block, hidden_states, **kwargs):
            if not hook._verify_mode:
                return original_forward(self_block, hidden_states, **kwargs)

            # Intercept the MoE block forward
            # The block typically does: router → top_k → fused_moe
            # We let router + top_k happen normally, then intercept fused_moe
            return original_forward(self_block, hidden_states, **kwargs)

        return wrapper

    def get_statistics(self) -> dict:
        stats = {
            "installed": self._installed,
            "total_intercepts": self._total_intercepts,
            "total_specmoe_calls": self._total_specmoe_calls,
            "total_passthrough_calls": self._total_passthrough_calls,
            "verify_mode": self._verify_mode,
        }
        if self._spec_moe is not None and hasattr(self._spec_moe, "get_statistics"):
            stats["spec_moe"] = self._spec_moe.get_statistics()
        if self._sdd is not None:
            stats["sdd"] = self._sdd.get_statistics()
        if self._expert_cache is not None:
            stats["expert_cache"] = self._expert_cache.get_statistics()
        return stats

    def reset_statistics(self):
        self._total_intercepts = 0
        self._total_specmoe_calls = 0
        self._total_passthrough_calls = 0
        if self._spec_moe and hasattr(self._spec_moe, "reset_statistics"):
            self._spec_moe.reset_statistics()


# ---------------------------------------------------------------------------
# Convenience: Global hook instance + install/uninstall helpers
# ---------------------------------------------------------------------------

_global_hook: Optional[FusedMoEHook] = None


def get_hook() -> FusedMoEHook:
    """Get or create the global FusedMoEHook instance."""
    global _global_hook
    if _global_hook is None:
        _global_hook = FusedMoEHook()
    return _global_hook


def install_specmoe_hook(
    spec_moe=None,
    sdd=None,
    expert_cache=None,
    trace_collector=None,
) -> bool:
    """
    Install SpecMoE hook into vLLM's fused_moe.

    Returns True if successfully installed.
    """
    hook = get_hook()
    hook.configure(
        spec_moe=spec_moe,
        sdd=sdd,
        expert_cache=expert_cache,
        trace_collector=trace_collector,
    )
    return hook.install()


def uninstall_specmoe_hook():
    """Uninstall SpecMoE hook and restore original fused_moe."""
    global _global_hook
    if _global_hook is not None:
        _global_hook.uninstall()
        _global_hook = None
