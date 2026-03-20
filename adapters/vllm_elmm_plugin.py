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

        from adapters.elmm_plugin import ELMMConfig, activate_elmm

        config = ELMMConfig(
            gpu_cache_budget_bytes=int(cache_gb * 1024**3),
            enable_prefetch=enable_prefetch,
            log_interval=log_interval,
        )

        model = self.model_runner.model
        activate_elmm(model, config)
        print(f"[ELMM] activate_elmm() complete", file=sys.stderr, flush=True)

    Worker.load_model = _patched_load_model
    print(f"[ELMM] Worker.load_model patched successfully", file=sys.stderr, flush=True)
    logger.info("ELMM plugin registered: Worker.load_model will be patched")
