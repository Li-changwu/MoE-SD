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
        rebalance_interval = int(os.environ.get("ELMM_REBALANCE_INTERVAL", "5000"))
        pool_direct = os.environ.get("ELMM_POOL_DIRECT", "1") == "1"
        direct_dispatch = os.environ.get("ELMM_DIRECT_DISPATCH", "1") == "1"
        gpu_cache = os.environ.get("ELMM_GPU_CACHE", "1") == "1"
        phase_profiling = os.environ.get("ELMM_PROFILE", "0") == "1"
        stale_remap = int(os.environ.get("ELMM_STALE_REMAP", "0"))
        stale_remap_warmup = int(os.environ.get("ELMM_STALE_REMAP_WARMUP", "32"))
        stale_remap_max_interval = int(os.environ.get("ELMM_STALE_REMAP_MAX_INTERVAL", "128"))
        enable_cuda_graph = os.environ.get("ELMM_CUDA_GRAPH", "1") == "1"
        enable_shared_parallel = os.environ.get("ELMM_SHARED_PARALLEL", "1") == "1"
        enable_oracle_prefetch = os.environ.get("ELMM_ORACLE_PREFETCH", "1") == "1"

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
            stale_remap_interval=stale_remap,
            stale_remap_warmup=stale_remap_warmup,
            stale_remap_max_interval=stale_remap_max_interval,
            enable_cuda_graph=enable_cuda_graph,
            enable_shared_parallel=enable_shared_parallel,
            enable_oracle_prefetch=enable_oracle_prefetch,
        )

        model = self.model_runner.model
        activate_elmm(model, config)
        print(f"[ELMM] activate_elmm() complete", file=sys.stderr, flush=True)

        # Install draft-guided prefetch hook (if prefetch enabled)
        if enable_prefetch:
            from adapters.draft_prefetch_hook import install_draft_prefetch
            pfh = install_draft_prefetch(
                __import__("adapters.elmm_plugin", fromlist=["get_elmm_manager"]).get_elmm_manager()
            )
            if pfh:
                print(f"[ELMM] Draft-guided prefetch hook installed", file=sys.stderr, flush=True)

    Worker.load_model = _patched_load_model
    print(f"[ELMM] Worker.load_model patched successfully", file=sys.stderr, flush=True)
    logger.info("ELMM plugin registered: Worker.load_model will be patched")
