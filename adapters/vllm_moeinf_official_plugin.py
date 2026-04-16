"""vLLM general plugin: MoE-Infinity official-like (no ELMM).

Usage:
  VLLM_PLUGINS=moeinf_official MOE_INFINITY_OFFICIAL_VLLM=1 python ...
"""

from __future__ import annotations

import os
import sys


def register():
    print(
        f"[MoEInf-vLLM] register() called in PID={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
    if os.environ.get("MOE_INFINITY_OFFICIAL_VLLM", "1") != "1":
        print(
            "[MoEInf-vLLM] disabled by MOE_INFINITY_OFFICIAL_VLLM=0",
            file=sys.stderr,
            flush=True,
        )
        return

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as e:  # pragma: no cover
        print(f"[MoEInf-vLLM] import Worker failed: {e}", file=sys.stderr, flush=True)
        return

    _orig_load_model = Worker.load_model

    def _patched_load_model(self) -> None:
        _orig_load_model(self)
        try:
            from adapters.vllm_moeinf_official import activate_vllm_moeinf_official

            model = self.model_runner.model
            ctrl = activate_vllm_moeinf_official(model)
            if ctrl is not None:
                print("[MoEInf-vLLM] official-like controller activated", file=sys.stderr, flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[MoEInf-vLLM] activation failed: {e}", file=sys.stderr, flush=True)

    Worker.load_model = _patched_load_model
    print("[MoEInf-vLLM] Worker.load_model patched", file=sys.stderr, flush=True)
