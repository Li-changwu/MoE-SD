"""
SpecMoE Engine — End-to-End Integration of All Components
============================================================
Orchestrates the full SpecMoE pipeline:
  1. Draft phase: EAGLE-3 generates K draft tokens
     → Collect draft routing predictions for prefetch
  2. Pre-verify: Prefetch predicted experts to GPU cache
  3. Verify phase: Target model verifies K+1 tokens in one forward
     → SpecFusedMoE deduplicates expert loads
     → SDD freezes divergent draft tokens early
     → Expert cache serves hot experts from GPU
  4. Post-verify: Update statistics, report acceptance

This is the top-level entry point for the SpecMoE system.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from adapters.triton_spec_moe import SpecFusedMoEDispatcher, SpecFusedMoETritonV2
from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig, PrefetchScheduler
from adapters.layer_early_terminator import SpeculationDivergenceDetector, SDDConfig
from adapters.fused_moe_hook import FusedMoEHook, get_hook
from collectors.expert_trace_hook import ExpertTraceCollector, compute_maf_from_trace
from collectors.expert_locality_analyzer import ExpertTemporalLocalityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SpecMoEConfig:
    """Full configuration for the SpecMoE engine."""
    # Model architecture
    num_experts: int = 128
    top_k: int = 8
    hidden_size: int = 2048
    moe_intermediate_size: int = 768
    num_moe_layers: int = 48

    # Speculation
    max_spec_tokens: int = 5  # Maximum K
    default_k: int = 3

    # Expert deduplication
    enable_dedup: bool = True
    use_triton: bool = True

    # SDD (early termination)
    enable_sdd: bool = True
    sdd_min_check_layer: int = 8
    sdd_consecutive_threshold: int = 3
    sdd_method: str = "combined"

    # Expert cache
    enable_cache: bool = True
    gpu_cache_budget_gb: float = 8.0
    enable_prefetch: bool = True
    prefetch_depth: int = 16

    # Trace collection (for profiling, off in production)
    enable_trace: bool = False
    trace_output_dir: str = "results/moe_trace"

    # vLLM integration
    hook_vllm: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "SpecMoEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "SpecMoEConfig":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


class SpecMoEEngine:
    """
    Top-level engine that orchestrates all SpecMoE components.

    Lifecycle:
        engine = SpecMoEEngine(config)
        engine.initialize(model)  # Register expert weights, install hooks
        # ... inference happens, hooks intercept automatically ...
        engine.shutdown()  # Cleanup
    """

    def __init__(self, config: Optional[SpecMoEConfig] = None):
        self.config = config or SpecMoEConfig()

        # Components (initialized lazily)
        self._dispatcher: Optional[SpecFusedMoEDispatcher] = None
        self._triton_v2: Optional[SpecFusedMoETritonV2] = None
        self._sdd: Optional[SpeculationDivergenceDetector] = None
        self._cache: Optional[ExpertWeightCache] = None
        self._prefetch: Optional[PrefetchScheduler] = None
        self._hook: Optional[FusedMoEHook] = None
        self._trace_collector: Optional[ExpertTraceCollector] = None
        self._locality_analyzer: Optional[ExpertTemporalLocalityAnalyzer] = None

        # State
        self._initialized = False
        self._current_k = self.config.default_k
        self._verify_rounds = 0
        self._total_tokens_generated = 0
        self._total_accepted = 0
        self._total_proposed = 0

        # Timing
        self._verify_time_ms = 0.0
        self._prefetch_time_ms = 0.0
        self._sdd_time_ms = 0.0

    def initialize(self, model=None):
        """
        Initialize all SpecMoE components.

        Args:
            model: Optional model instance (for registering expert weights
                   and attaching router hooks)
        """
        cfg = self.config

        # 1. Create SpecFusedMoE dispatcher
        if cfg.enable_dedup:
            self._dispatcher = SpecFusedMoEDispatcher(
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                use_triton=cfg.use_triton,
            )
            self._triton_v2 = SpecFusedMoETritonV2(
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
            )
            logger.info("SpecFusedMoE dispatcher initialized")

        # 2. Create SDD
        if cfg.enable_sdd:
            sdd_config = SDDConfig(
                min_check_layer=cfg.sdd_min_check_layer,
                consecutive_threshold=cfg.sdd_consecutive_threshold,
                method=cfg.sdd_method,
            )
            self._sdd = SpeculationDivergenceDetector(
                config=sdd_config,
                num_layers=cfg.num_moe_layers,
            )
            logger.info("SDD initialized")

        # 3. Create Expert Cache
        if cfg.enable_cache:
            cache_config = ExpertCacheConfig(
                gpu_budget_bytes=int(cfg.gpu_cache_budget_gb * 1024**3),
                enable_prefetch=cfg.enable_prefetch,
                prefetch_depth=cfg.prefetch_depth,
                pin_cpu_memory=True,
            )
            self._cache = ExpertWeightCache(config=cache_config)
            self._prefetch = PrefetchScheduler(
                cache=self._cache,
                num_layers=cfg.num_moe_layers,
            )
            logger.info(f"Expert cache initialized ({cfg.gpu_cache_budget_gb:.1f} GB budget)")

        # 4. Create Trace Collector (optional)
        if cfg.enable_trace:
            trace_dir = Path(cfg.trace_output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_collector = ExpertTraceCollector(
                output_path=str(trace_dir / "specmoe_trace.jsonl")
            )
            logger.info(f"Trace collection enabled → {trace_dir}")

        # 5. Create Locality Analyzer
        self._locality_analyzer = ExpertTemporalLocalityAnalyzer(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            num_layers=cfg.num_moe_layers,
        )

        # 6. Register expert weights from model
        if model is not None and cfg.enable_cache:
            self._register_model_experts(model)

        # 7. Install vLLM hook
        if cfg.hook_vllm:
            self._hook = get_hook()
            self._hook.configure(
                spec_moe=self._dispatcher,
                sdd=self._sdd,
                expert_cache=self._cache,
                trace_collector=self._trace_collector,
            )
            success = self._hook.install()
            if not success:
                logger.warning("vLLM hook installation failed; "
                               "will operate in manual dispatch mode")

        self._initialized = True
        logger.info("SpecMoE engine initialized")

    def _register_model_experts(self, model):
        """Extract and register expert weights from a loaded model."""
        registered = 0
        for name, param in model.named_parameters():
            # Qwen3MoE weight names:
            #   model.layers.{l}.mlp.experts.{e}.gate_proj.weight  [N, D]
            #   model.layers.{l}.mlp.experts.{e}.up_proj.weight    [N, D]
            #   model.layers.{l}.mlp.experts.{e}.down_proj.weight  [D, N]
            #
            # We need to pack into: w1[E, 2N, D] and w2[E, D, N] per layer
            pass  # Packing logic depends on model state dict layout

        # Alternative: register from vLLM's packed weight format directly
        for name, module in model.named_modules():
            if hasattr(module, "w1") and hasattr(module, "w2"):
                # Already in packed format [E, 2N, D] and [E, D, N]
                parts = name.split(".")
                layer_id = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_id = int(parts[i + 1])
                        except ValueError:
                            pass

                if layer_id is not None:
                    self._cache.register_experts(
                        layer_id=layer_id,
                        w1=module.w1.data,
                        w2=module.w2.data,
                    )
                    registered += 1

        logger.info(f"Registered expert weights from {registered} MoE layers")

    # -------------------------------------------------------------------
    # Draft Phase API
    # -------------------------------------------------------------------

    def on_draft_step(
        self,
        draft_routing: dict[int, list[list[int]]],
        token_acceptance_probs: Optional[list[float]] = None,
    ):
        """
        Called after each draft token is generated.
        Triggers expert prefetch based on draft routing predictions.

        Args:
            draft_routing: {layer_id: [[experts_token0], [experts_token1], ...]}
            token_acceptance_probs: Per-token acceptance probability estimates
        """
        if self._prefetch is None:
            return

        for layer_id, token_experts in draft_routing.items():
            all_experts = []
            for experts in token_experts:
                all_experts.extend(experts)
            self._prefetch.on_draft_routing(
                layer_id=layer_id,
                expert_ids=list(set(all_experts)),
                token_acceptance_probs=token_acceptance_probs,
            )

    # -------------------------------------------------------------------
    # Verify Phase API
    # -------------------------------------------------------------------

    def begin_verify(self, K: int, active_mask: Optional[torch.Tensor] = None):
        """
        Signal the start of a verify phase.

        Args:
            K: Number of draft tokens to verify
            active_mask: Optional initial active mask [K+1]
        """
        self._current_k = K

        # Sync prefetch
        if self._prefetch is not None:
            t0 = time.monotonic()
            self._prefetch.prepare_verify(K)
            self._prefetch_time_ms += (time.monotonic() - t0) * 1000

        # Activate verify mode on hook
        if self._hook is not None:
            self._hook.set_verify_mode(
                enabled=True,
                batch_size=K + 1,
                active_mask=active_mask,
            )

        # Initialize SDD
        if self._sdd is not None:
            self._sdd.init_verify_round(num_draft_tokens=K)

    def end_verify(
        self,
        accepted_tokens: int,
        proposed_tokens: int,
        used_experts: Optional[dict[int, list[int]]] = None,
    ):
        """
        Signal the end of a verify phase. Report results.

        Args:
            accepted_tokens: Number of draft tokens accepted
            proposed_tokens: Number of draft tokens proposed (K)
            used_experts: {layer_id: [expert_ids]} actually used
        """
        self._verify_rounds += 1
        self._total_accepted += accepted_tokens
        self._total_proposed += proposed_tokens
        self._total_tokens_generated += accepted_tokens + 1  # +1 for bonus token

        # Deactivate verify mode
        if self._hook is not None:
            self._hook.set_verify_mode(enabled=False)

        # Report prefetch accuracy
        if self._prefetch is not None and used_experts:
            self._prefetch.report_verify_result(used_experts)

        # Report SDD accuracy
        if self._sdd is not None:
            for t_idx in range(proposed_tokens):
                is_accepted = t_idx < accepted_tokens
                self._sdd.report_acceptance(t_idx, is_accepted)

        # Record locality
        if self._locality_analyzer is not None and used_experts:
            self._locality_analyzer.record_verify_round(
                round_id=self._verify_rounds,
                expert_indices={
                    lid: [[eid] for eid in eids]
                    for lid, eids in used_experts.items()
                },
            )

    # -------------------------------------------------------------------
    # Manual dispatch API (for standalone use without vLLM hooks)
    # -------------------------------------------------------------------

    def dispatch_moe(
        self,
        hidden_states: torch.Tensor,   # [T, D]
        w1: torch.Tensor,              # [E, 2*N, D]
        w2: torch.Tensor,              # [E, D, N]
        topk_weights: torch.Tensor,    # [T, top_k]
        topk_ids: torch.Tensor,        # [T, top_k]
        layer_id: int = 0,
        is_verify: bool = True,
    ) -> torch.Tensor:
        """
        Directly dispatch a MoE computation with SpecMoE optimizations.
        Use this when not hooked into vLLM.

        Args:
            hidden_states: Input [T, D]
            w1, w2: Expert weights
            topk_weights, topk_ids: Routing decisions
            layer_id: Current MoE layer index
            is_verify: Whether this is an SD verify pass

        Returns:
            output: [T, D]
        """
        if not is_verify or self._dispatcher is None:
            # Fallback to vanilla computation
            from adapters.triton_spec_moe import SpecFusedMoEFunction
            output, _, _ = SpecFusedMoEFunction.forward(
                hidden_states, w1, w2, topk_weights, topk_ids
            )
            return output

        # SDD check
        active_mask = None
        if self._sdd is not None:
            batch_size = hidden_states.shape[0]
            router_proxy = torch.zeros(
                batch_size, self.config.num_experts,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            for b in range(batch_size):
                for s in range(topk_ids.shape[1]):
                    eid = topk_ids[b, s].item()
                    router_proxy[b, eid] = topk_weights[b, s]

            token_indices = list(range(1, batch_size))
            # SDD only checks draft tokens (indices 1..T-1)
            sdd_frozen = self._sdd.check_layer(
                layer_id=layer_id,
                router_logits=router_proxy[1:],  # Only draft tokens
                token_indices=token_indices,
            )
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
            active_mask[1:] = ~sdd_frozen

        # Dedup dispatch
        t0 = time.monotonic()
        output = self._dispatcher(
            hidden_states, w1, w2, topk_weights, topk_ids,
            active_mask=active_mask,
        )
        self._verify_time_ms += (time.monotonic() - t0) * 1000

        return output

    # -------------------------------------------------------------------
    # Statistics & Reporting
    # -------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        return self._total_accepted / max(1, self._total_proposed)

    @property
    def mean_accepted_length(self) -> float:
        return self._total_accepted / max(1, self._verify_rounds)

    def get_statistics(self) -> dict:
        stats = {
            "initialized": self._initialized,
            "config": self.config.to_dict(),
            "verify_rounds": self._verify_rounds,
            "tokens_generated": self._total_tokens_generated,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_accepted_length": round(self.mean_accepted_length, 2),
            "verify_time_ms": round(self._verify_time_ms, 2),
            "prefetch_time_ms": round(self._prefetch_time_ms, 2),
        }

        if self._dispatcher is not None:
            stats["dispatcher"] = self._dispatcher.get_statistics()
        if self._sdd is not None:
            stats["sdd"] = self._sdd.get_statistics()
        if self._cache is not None:
            stats["cache"] = self._cache.get_statistics()
        if self._prefetch is not None:
            stats["prefetch"] = self._prefetch.get_statistics()
        if self._hook is not None:
            stats["hook"] = self._hook.get_statistics()
        if self._locality_analyzer is not None:
            locality = self._locality_analyzer.compute_statistics()
            if locality.num_rounds > 0:
                stats["locality"] = {
                    "inter_round_overlap": round(locality.mean_interround_overlap, 4),
                    "reuse_distance": round(locality.mean_reuse_distance, 2),
                    "cache_hit_estimate": round(locality.cache_hit_rate_estimate, 4),
                }

        return stats

    def save_statistics(self, path: str):
        """Save engine statistics to JSON."""
        stats = self.get_statistics()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Statistics saved to {path}")

    def shutdown(self):
        """Cleanup: uninstall hooks, flush traces, save stats."""
        if self._hook is not None:
            self._hook.uninstall()

        if self._trace_collector is not None:
            self._trace_collector.detach()

        if self._cache is not None:
            self._cache.clear_gpu_cache()

        self._initialized = False
        logger.info("SpecMoE engine shut down")

    def reset_statistics(self):
        self._verify_rounds = 0
        self._total_tokens_generated = 0
        self._total_accepted = 0
        self._total_proposed = 0
        self._verify_time_ms = 0.0
        self._prefetch_time_ms = 0.0
        self._sdd_time_ms = 0.0
        if self._dispatcher:
            self._dispatcher.reset_statistics()
        if self._cache:
            self._cache.reset_statistics()
        if self._prefetch:
            self._prefetch.reset_statistics()


# ---------------------------------------------------------------------------
# Quick-Start Helper
# ---------------------------------------------------------------------------

def create_specmoe_engine(
    num_experts: int = 128,
    top_k: int = 8,
    hidden_size: int = 2048,
    moe_intermediate_size: int = 768,
    num_layers: int = 48,
    gpu_cache_gb: float = 8.0,
    enable_sdd: bool = True,
    enable_trace: bool = False,
    model=None,
) -> SpecMoEEngine:
    """
    Create and initialize a SpecMoE engine with common defaults.

    Usage:
        engine = create_specmoe_engine(model=loaded_model)
        engine.begin_verify(K=3)
        # ... model.forward() runs with hooks active ...
        engine.end_verify(accepted=2, proposed=3)
    """
    config = SpecMoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_moe_layers=num_layers,
        gpu_cache_budget_gb=gpu_cache_gb,
        enable_sdd=enable_sdd,
        enable_trace=enable_trace,
    )
    engine = SpecMoEEngine(config)
    engine.initialize(model)
    return engine
