#!/usr/bin/env python3
"""
SpecMoE-enhanced vLLM Server v2 — with runtime hook reconfiguration.

Single model load; switch SpecMoE configurations between benchmark runs
via POST /specmoe/configure.

Usage:
  python scripts/specmoe_server_v2.py \
    /root/models/Qwen3-30B-A3B-Instruct-2507 \
    --port 8192 \
    --speculative-config '{"method":"eagle3","model":"...","num_speculative_tokens":3}' \
    --gpu-memory-utilization 0.90 --cpu-offload-gb 30 \
    --max-model-len 4096 --dtype bfloat16 --enforce-eager --trust-remote-code
"""
import asyncio
import json
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.fused_moe_hook import FusedMoEHook

# ── Global hook ───────────────────────────────────────────────────────────────
_hook = FusedMoEHook()
_hook_lock = asyncio.Lock()  # Prevents reconfiguration during in-flight requests

# EAGLE-3 spec uses K=3, so verify batch = K+1 = 4 tokens
_VERIFY_BATCH_THRESHOLD = 4


def _configure_hook(cfg_id: int) -> dict:
    """Reconfigure hook for a given experiment config."""
    # Full reset — including verify_mode so stale state from previous config
    # cannot leak into the next benchmark run.
    _hook.set_verify_mode(False)
    _hook._spec_moe = None
    _hook._sdd = None
    _hook._expert_cache = None
    _hook._auto_verify_threshold = 0
    _hook._total_intercepts = 0
    _hook._total_specmoe_calls = 0
    _hook._total_passthrough_calls = 0

    if cfg_id <= 2:
        # Config 1/2: passthrough — hook stays installed but does nothing
        return {"status": "ok", "cfg_id": cfg_id, "mode": "passthrough"}

    # Config 3+: dedup
    if cfg_id >= 3:
        from adapters.triton_spec_moe import SpecFusedMoEDispatcher
        _hook.configure(
            spec_moe=SpecFusedMoEDispatcher(),
            auto_verify_threshold=_VERIFY_BATCH_THRESHOLD,
        )

    # Config 4+: SDD
    if cfg_id >= 4:
        from adapters.layer_early_terminator import (
            SDDConfig, SpeculationDivergenceDetector,
        )
        _hook._sdd = SpeculationDivergenceDetector(
            config=SDDConfig(
                min_check_layer=8,
                method="combined",
                consecutive_threshold=3,
            ),
            num_layers=48,
        )

    # Config 5: cache
    if cfg_id >= 5:
        from adapters.expert_cache import ExpertCacheConfig, ExpertWeightCache
        _hook._expert_cache = ExpertWeightCache(
            config=ExpertCacheConfig(
                gpu_budget_bytes=8 * 1024**3,
                eviction_policy="lru",
                enable_prefetch=True,
                pin_cpu_memory=True,
            )
        )

    # Do NOT call set_verify_mode(True) here.
    # The hook auto-detects verify phase by batch size (_auto_verify_threshold).
    features = []
    if cfg_id >= 3: features.append("dedup")
    if cfg_id >= 4: features.append("sdd")
    if cfg_id >= 5: features.append("cache")
    return {"status": "ok", "cfg_id": cfg_id, "mode": "+".join(features)}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from vllm.entrypoints.launcher import serve_http
    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        create_server_socket,
        init_app_state,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # Parse standard vLLM args (mirror vllm serve subcommand)
    base_parser = FlexibleArgumentParser(description="SpecMoE-v2 Server")
    parser = make_arg_parser(base_parser)
    args = parser.parse_args(sys.argv[1:])

    # Install hook BEFORE model loads
    _hook.install()
    print("[SpecMoE-v2] Hook installed (passthrough mode)")

    sock = create_server_socket((args.host or "", args.port))

    def signal_handler(*_):
        raise KeyboardInterrupt("terminated")
    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        # ── Add SpecMoE control endpoints ─────────────────────────────────
        @app.post("/specmoe/configure")
        async def specmoe_configure(request: Request):
            body = await request.json()
            cfg_id = int(body.get("cfg_id", 2))
            # Hold the lock so no in-flight request races with reconfiguration
            async with _hook_lock:
                result = _configure_hook(cfg_id)
            print(f"[SpecMoE-v2] Configured: {result}")
            return JSONResponse(result)

        @app.get("/specmoe/status")
        async def specmoe_status():
            return JSONResponse({
                "installed": _hook._installed,
                "verify_mode": _hook._verify_mode,
                "auto_verify_threshold": _hook._auto_verify_threshold,
                "total_intercepts": _hook._total_intercepts,
                "specmoe_calls": _hook._total_specmoe_calls,
                "passthrough_calls": _hook._total_passthrough_calls,
            })

        # ── Init app state & serve ────────────────────────────────────────
        await init_app_state(engine_client, app.state, args)

        print(f"[SpecMoE-v2] Server starting on port {args.port}")
        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=5,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()
        _hook.uninstall()
        print("[SpecMoE-v2] Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
