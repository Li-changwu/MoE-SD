"""vLLM general plugin: AdapMoE official-like (no ELMM).

Usage:
  VLLM_PLUGINS=adapmoe_official ADAPMOE_OFFICIAL_VLLM=1 python ...
"""

from __future__ import annotations

import errno
import multiprocessing as mp
import os
import sys
import threading


def _maybe_patch_lock_for_shm_exhaustion() -> None:
    """Fallback for environments where /dev/shm semaphores are exhausted.

    vLLM uniprocess executor uses multiprocessing.Lock(). On hosts with a tiny
    or occupied /dev/shm, that can fail with Errno 28 before model load.
    """
    try:
        test_lock = mp.Lock()
        test_lock.acquire()
        test_lock.release()
        return
    except OSError as e:
        if e.errno != errno.ENOSPC:
            return

    print(
        "[AdapMoE-vLLM] /dev/shm semaphore exhausted; using threading.Lock fallback",
        file=sys.stderr,
        flush=True,
    )

    def _thread_lock(*_args, **_kwargs):
        return threading.Lock()

    mp.Lock = _thread_lock
    try:
        import multiprocessing.context as mp_ctx

        mp_ctx.BaseContext.Lock = lambda self: threading.Lock()
    except Exception:
        pass
    try:
        import vllm.v1.executor.uniproc_executor as uniproc_executor

        uniproc_executor.Lock = _thread_lock
    except Exception:
        pass


def register():
    print(
        f"[AdapMoE-vLLM] register() called in PID={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
    _maybe_patch_lock_for_shm_exhaustion()

    if os.environ.get("ADAPMOE_OFFICIAL_VLLM", "1") != "1":
        print(
            "[AdapMoE-vLLM] disabled by ADAPMOE_OFFICIAL_VLLM=0",
            file=sys.stderr,
            flush=True,
        )
        return

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as e:  # pragma: no cover
        print(f"[AdapMoE-vLLM] import Worker failed: {e}", file=sys.stderr, flush=True)
        return

    _orig_load_model = Worker.load_model

    def _patched_load_model(self) -> None:
        _orig_load_model(self)
        try:
            from adapters.vllm_adapmoe_official import activate_vllm_adapmoe_official

            model = self.model_runner.model
            ctrl = activate_vllm_adapmoe_official(model)
            if ctrl is not None:
                print("[AdapMoE-vLLM] official-like controller activated", file=sys.stderr, flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[AdapMoE-vLLM] activation failed: {e}", file=sys.stderr, flush=True)

    Worker.load_model = _patched_load_model
    print("[AdapMoE-vLLM] Worker.load_model patched", file=sys.stderr, flush=True)
