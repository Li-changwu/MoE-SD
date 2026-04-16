"""
vLLM General Plugin: ELMM (Expert-Level Memory Management)
==========================================================
Registers a monkey-patch on ``Worker.load_model`` so that ELMM activates
automatically after model loading in the GPU worker process.

Activated via the ``vllm.general_plugins`` entry point in pyproject.toml.
Controlled by ``VLLM_PLUGINS=elmm`` environment variable.

Configuration via environment variables:
  ELMM_CACHE_GB     — GPU cache budget in GB        (default: 8)
  ELMM_PREFETCH     — enable async prefetch (0/1)   (default: 1)
  ELMM_LOG_INTERVAL — log stats every N steps (0=off) (default: 0)
"""

import logging
import os

logger = logging.getLogger(__name__)


def register():
    """
    Called by vLLM's plugin loader (``load_general_plugins``) in every
    process — engine core + each GPU worker — **before** model loading.

    We monkey-patch ``Worker.load_model`` so that after the original
    load completes, ELMM installs itself on the loaded model.
    """
    import sys
    print(f"[ELMM] register() called in PID={os.getpid()}", file=sys.stderr, flush=True)

    # Point vLLM's FusedMoE config lookup at our A6000-tuned tile configs
    # (must be set before any model loading triggers get_moe_configs()).
    if "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        _tuned_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "vllm_tuned_a6000",
        )
        if os.path.isdir(_tuned_dir):
            os.environ["VLLM_TUNED_CONFIG_FOLDER"] = _tuned_dir
            print(f"[ELMM] VLLM_TUNED_CONFIG_FOLDER={_tuned_dir}",
                  file=sys.stderr, flush=True)

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as e:
        print(f"[ELMM] Failed to import Worker: {e}", file=sys.stderr, flush=True)
        return

    _original_load_model = Worker.load_model

    def _patched_load_model(self) -> None:
        print(f"[ELMM] patched load_model called in PID={os.getpid()}", file=sys.stderr, flush=True)
        # Run original model loading (loads weights, creates CUDA graphs, etc.)
        _original_load_model(self)
        print(f"[ELMM] original load_model done, activating ELMM...", file=sys.stderr, flush=True)

        # Read config from environment
        cache_gb = float(os.environ.get("ELMM_CACHE_GB", "8"))
        enable_prefetch = os.environ.get("ELMM_PREFETCH", "1") == "1"
        log_interval = int(os.environ.get("ELMM_LOG_INTERVAL", "0"))
        enable_locality = os.environ.get("ELMM_LOCALITY", "1") == "1"
        locality_export_dir = os.environ.get("ELMM_LOCALITY_DIR", "")
        locality_export_interval = int(os.environ.get("ELMM_LOCALITY_INTERVAL", "0"))
        enable_adaptive = os.environ.get("ELMM_ADAPTIVE_BUDGET", "1") == "1"
        rebalance_interval = int(os.environ.get("ELMM_REBALANCE_INTERVAL", "200000"))
        pool_direct = os.environ.get("ELMM_POOL_DIRECT", "1") == "1"
        direct_dispatch = os.environ.get("ELMM_DIRECT_DISPATCH", "0") == "1"
        gpu_cache = os.environ.get("ELMM_GPU_CACHE", "0") == "1"
        phase_profiling = os.environ.get("ELMM_PROFILE", "0") == "1"
        profile_warmup = int(os.environ.get("ELMM_PROFILE_WARMUP", "200"))
        profile_steps = int(os.environ.get("ELMM_PROFILE_STEPS", "100"))
        stale_remap = int(os.environ.get("ELMM_STALE_REMAP", "0"))
        stale_remap_warmup = int(os.environ.get("ELMM_STALE_REMAP_WARMUP", "32"))
        stale_remap_max_interval = int(os.environ.get("ELMM_STALE_REMAP_MAX_INTERVAL", "128"))
        enable_cuda_graph = os.environ.get("ELMM_CUDA_GRAPH", "1") == "1"
        enable_shared_parallel = os.environ.get("ELMM_SHARED_PARALLEL", "1") == "1"
        enable_oracle_prefetch = os.environ.get("ELMM_ORACLE_PREFETCH", "1") == "1"
        enable_hfde = os.environ.get("ELMM_HFDE", "1") == "1"
        enable_stacked_gating = os.environ.get("ELMM_STACKED_GATING", "1") == "1"
        stacked_gating_top_k = int(os.environ.get("ELMM_SG_TOP_K", "4"))
        # RWAWE (P2)
        enable_rwawe = os.environ.get("ELMM_RWAWE", "1") == "1"
        rwawe_alpha = float(os.environ.get("ELMM_RWAWE_ALPHA", "0.6"))
        rwawe_lambda = float(os.environ.get("ELMM_RWAWE_LAMBDA", "0.1"))
        rwawe_beta = float(os.environ.get("ELMM_RWAWE_BETA", "0.05"))
        # Draft-Utility (P3)
        enable_draft_utility = os.environ.get("ELMM_DRAFT_UTILITY", "1") == "1"
        # Entropy-Aware Budget (P4)
        enable_entropy_budget = os.environ.get("ELMM_ENTROPY_BUDGET", "1") == "1"
        entropy_ema_gamma = float(os.environ.get("ELMM_ENTROPY_EMA_GAMMA", "0.05"))
        entropy_kappa = float(os.environ.get("ELMM_ENTROPY_KAPPA", "1.0"))
        # TASER v2: Dual-Rail Hot Expert Routing
        taser_miss_budget_ratio = float(os.environ.get("ELMM_TASER_MISS_BUDGET", "0.15"))
        taser_converge_threshold = float(os.environ.get("ELMM_TASER_CONVERGE", "0.90"))
        taser_converge_max_steps = int(os.environ.get("ELMM_TASER_CONVERGE_MAX", "10"))
        taser_drift_ema_alpha = float(os.environ.get("ELMM_TASER_DRIFT_ALPHA", "0.1"))
        taser_drift_miss_threshold = float(os.environ.get("ELMM_TASER_DRIFT_THRESH", "0.3"))
        taser_drift_warmup = int(os.environ.get("ELMM_TASER_DRIFT_WARMUP", "10"))
        taser_cold_slots = int(os.environ.get("ELMM_TASER_COLD_SLOTS", "3"))
        # BriskMoE integration
        enable_sacr = os.environ.get("BRISKMOE_SACR", "0") == "1"
        enable_elp = os.environ.get("BRISKMOE_ELP", "0") == "1"
        enable_dipp = os.environ.get("BRISKMOE_DIPP", "0") == "1"
        enable_pred_cache = os.environ.get("BRISKMOE_PREDCACHE", "0") == "1"
        enable_aces = os.environ.get("ELMM_ACES", "0") == "1"
        aces_beta = float(os.environ.get("ELMM_ACES_BETA", "0.9"))
        aces_taser = os.environ.get("ELMM_ACES_TASER", "0") == "1"
        pred_cache_lru_weight = float(os.environ.get("BRISKMOE_PREDCACHE_LRU_WEIGHT", "10.0"))
        sacr_alpha = float(os.environ.get("BRISKMOE_SACR_ALPHA", "0.3"))
        sacr_beta = float(os.environ.get("BRISKMOE_SACR_BETA", "0.2"))
        sacr_gamma = float(os.environ.get("BRISKMOE_SACR_GAMMA", "0.5"))
        elp_pin_ratio = float(os.environ.get("BRISKMOE_ELP_PIN_RATIO", "0.7"))
        elp_promotion_threshold = int(os.environ.get("BRISKMOE_ELP_THRESHOLD", "5"))
        elp_demotion_window = int(os.environ.get("BRISKMOE_ELP_DEMOTION", "50"))
        # Unified Scheduling (D1-D6)
        enable_unified = os.environ.get("ELMM_UNIFIED", "0") == "1"
        unified_split_ratio = float(os.environ.get("ELMM_UNIFIED_SPLIT_RATIO", "0.4"))
        unified_tail_slots = int(os.environ.get("ELMM_UNIFIED_TAIL_SLOTS", "8"))
        unified_budget_gb = float(os.environ.get("ELMM_UNIFIED_BUDGET_GB", "0"))
        unified_governor_mode = os.environ.get("ELMM_UNIFIED_GOVERNOR", "rule")
        unified_verify_pattern = os.environ.get("ELMM_UNIFIED_PATTERN", "auto")
        unified_K_min = int(os.environ.get("ELMM_UNIFIED_K_MIN", "1"))
        unified_K_max = int(os.environ.get("ELMM_UNIFIED_K_MAX", "5"))
        unified_K_default = int(os.environ.get("ELMM_UNIFIED_K_DEFAULT", "3"))
        unified_prefetch_aggressiveness = float(os.environ.get("ELMM_UNIFIED_AGGRESSIVENESS", "0.7"))

        from adapters.elmm_plugin import ELMMConfig, activate_elmm

        config = ELMMConfig(
            gpu_cache_budget_bytes=int(cache_gb * 1024**3),
            enable_prefetch=enable_prefetch,
            log_interval=log_interval,
            enable_locality_collection=enable_locality,
            locality_export_dir=locality_export_dir,
            locality_export_interval=locality_export_interval,
            enable_adaptive_budget=enable_adaptive,
            rebalance_interval=rebalance_interval,
            enable_pool_direct=pool_direct,
            enable_direct_dispatch=direct_dispatch,
            enable_gpu_cache=gpu_cache,
            enable_phase_profiling=phase_profiling,
            profile_warmup=profile_warmup,
            profile_steps=profile_steps,
            stale_remap_interval=stale_remap,
            stale_remap_warmup=stale_remap_warmup,
            stale_remap_max_interval=stale_remap_max_interval,
            enable_cuda_graph=enable_cuda_graph,
            enable_shared_parallel=enable_shared_parallel,
            enable_oracle_prefetch=enable_oracle_prefetch,
            enable_hfde=enable_hfde,
            enable_stacked_gating=enable_stacked_gating,
            stacked_gating_top_k=stacked_gating_top_k,
            enable_rwawe=enable_rwawe,
            rwawe_alpha=rwawe_alpha,
            rwawe_lambda=rwawe_lambda,
            rwawe_beta=rwawe_beta,
            enable_draft_utility=enable_draft_utility,
            enable_entropy_budget=enable_entropy_budget,
            entropy_ema_gamma=entropy_ema_gamma,
            entropy_kappa=entropy_kappa,
            taser_miss_budget_ratio=taser_miss_budget_ratio,
            taser_converge_threshold=taser_converge_threshold,
            taser_converge_max_steps=taser_converge_max_steps,
            taser_drift_ema_alpha=taser_drift_ema_alpha,
            taser_drift_miss_threshold=taser_drift_miss_threshold,
            taser_drift_warmup=taser_drift_warmup,
            taser_cold_slots=taser_cold_slots,
            enable_sacr=enable_sacr,
            enable_elp=enable_elp,
            enable_dipp=enable_dipp,
            sacr_alpha=sacr_alpha,
            sacr_beta=sacr_beta,
            sacr_gamma=sacr_gamma,
            elp_pin_ratio=elp_pin_ratio,
            elp_promotion_threshold=elp_promotion_threshold,
            elp_demotion_window=elp_demotion_window,
            enable_pred_cache=enable_pred_cache,
            pred_cache_lru_weight=pred_cache_lru_weight,
            enable_aces=enable_aces,
            aces_beta=aces_beta,
            aces_taser=aces_taser,
            # Unified Scheduling (D1-D6)
            enable_unified_scheduling=enable_unified,
            unified_split_ratio=unified_split_ratio,
            unified_tail_slots=unified_tail_slots,
            unified_budget_bytes=int(unified_budget_gb * 1024**3),
            unified_governor_mode=unified_governor_mode,
            unified_verify_pattern=unified_verify_pattern,
            unified_K_min=unified_K_min,
            unified_K_max=unified_K_max,
            unified_K_default=unified_K_default,
            unified_prefetch_aggressiveness=unified_prefetch_aggressiveness,
        )

        model = self.model_runner.model
        activate_elmm(model, config)
        print(f"[ELMM] activate_elmm() complete", file=sys.stderr, flush=True)

        # Fix memory accounting: ELMM allocates GPU cache + scratchpad
        # OUTSIDE vLLM's model_memory_usage tracking context.
        # Without this, vLLM over-allocates KV cache and OOMs.
        elmm_mgr = __import__("adapters.elmm_plugin", fromlist=["get_elmm_manager"]).get_elmm_manager()
        if elmm_mgr and elmm_mgr._installed:
            elmm_gpu_bytes = 0
            # Cache pools
            for cache in elmm_mgr._layer_caches.values():
                elmm_gpu_bytes += cache._w13_pool.nelement() * cache._w13_pool.element_size()
                elmm_gpu_bytes += cache._w2_pool.nelement() * cache._w2_pool.element_size()
            # Scratchpad (only if allocated on GPU)
            if elmm_mgr._scratchpad_on_gpu:
                if elmm_mgr._scratch_w13 is not None:
                    elmm_gpu_bytes += elmm_mgr._scratch_w13.nelement() * elmm_mgr._scratch_w13.element_size()
                if elmm_mgr._scratch_w2 is not None:
                    elmm_gpu_bytes += elmm_mgr._scratch_w2.nelement() * elmm_mgr._scratch_w2.element_size()
            # D4 ResidencyManager GPU pools (unified scheduling)
            if hasattr(elmm_mgr, '_unified_residency') and elmm_mgr._unified_residency is not None:
                for pool in elmm_mgr._unified_residency._pools.values():
                    for t in (pool.head_w13, pool.head_w2, pool.tail_w13, pool.tail_w2):
                        elmm_gpu_bytes += t.nelement() * t.element_size()
            self.model_runner.model_memory_usage += elmm_gpu_bytes
            print(f"[ELMM] Added {elmm_gpu_bytes / 1024**3:.2f} GiB to model_memory_usage "
                  f"(total: {self.model_runner.model_memory_usage / 1024**3:.2f} GiB)",
                  file=sys.stderr, flush=True)

        # Install draft-guided prefetch hook (if prefetch enabled)
        # SP-MoE baseline mode: use SP-MoE's cross-model predictor + cutoff
        # instead of BriskMoE's DIPP/PredCache/locality-based prefetch.
        enable_spmoe = os.environ.get("SPMOE_ENABLE", "0") == "1"
        if enable_spmoe:
            from adapters.spmoe_baseline import activate_spmoe
            spmoe = activate_spmoe(
                __import__("adapters.elmm_plugin", fromlist=["get_elmm_manager"]).get_elmm_manager()
            )
            if spmoe:
                print(f"[ELMM] SP-MoE baseline mode activated", file=sys.stderr, flush=True)
            else:
                print(f"[ELMM] SP-MoE activation failed, falling back to default",
                      file=sys.stderr, flush=True)
        elif enable_prefetch:
            from adapters.draft_prefetch_hook import install_draft_prefetch
            pfh = install_draft_prefetch(
                __import__("adapters.elmm_plugin", fromlist=["get_elmm_manager"]).get_elmm_manager()
            )
            if pfh:
                print(f"[ELMM] Draft-guided prefetch hook installed", file=sys.stderr, flush=True)

        # v3.1: Install overflow controller (C1-C6) if enabled
        enable_overflow_ctrl = os.environ.get("ELMM_OVERFLOW_CTRL", "0") == "1"
        if enable_overflow_ctrl:
            ctrl_elmm = __import__("adapters.elmm_plugin", fromlist=["get_elmm_manager"]).get_elmm_manager()
            if ctrl_elmm and ctrl_elmm._installed:
                from adapters.overflow_controller import OverflowController
                K_max = int(os.environ.get("ELMM_OVERFLOW_K_MAX", "4"))
                K_min = int(os.environ.get("ELMM_OVERFLOW_K_MIN", "1"))
                stall_thresh = float(os.environ.get("ELMM_OVERFLOW_STALL_THRESH", "1.0"))
                controller = OverflowController(
                    ctrl_elmm,
                    K_max=K_max,
                    K_min=K_min,
                    stall_threshold_ms=stall_thresh,
                )
                controller.configure()
                ctrl_elmm._overflow_controller = controller
                print(f"[ELMM] v3.1 Overflow Controller installed (K=[{K_min},{K_max}], "
                      f"stall_thresh={stall_thresh}ms)", file=sys.stderr, flush=True)

    Worker.load_model = _patched_load_model
    print(f"[ELMM] Worker.load_model patched successfully", file=sys.stderr, flush=True)
    logger.info("ELMM plugin registered: Worker.load_model will be patched")
