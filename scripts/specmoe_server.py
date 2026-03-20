#!/usr/bin/env python3
"""
SpecMoE-enhanced vLLM OpenAI Server
====================================
Installs FusedMoEHook BEFORE starting the vLLM OpenAI-compatible server,
so all inference requests pass through SpecMoE optimizations.

This allows configs 3-5 to serve via the same OpenAI API as configs 1-2,
making `vllm bench serve` results directly comparable across all configs.

Architecture:
  1. Parse CLI args (same as `vllm serve` + SpecMoE extras)
  2. Install FusedMoEHook (monkey-patch fused_moe in same process)
  3. Call vllm.entrypoints.openai.api_server.run_server()
  4. vLLM V1 AsyncLLM runs in-process → hook intercepts all fused_moe calls

Usage:
  python scripts/specmoe_server.py \\
    --model /root/models/Qwen3-30B-A3B-Instruct-2507 \\
    --speculative-config '{"method":"eagle3","model":"/root/models/...speculator.eagle3","num_speculative_tokens":3}' \\
    --port 8192 \\
    --specmoe-dedup --specmoe-sdd --specmoe-cache
"""
import asyncio
import sys
import os

# Ensure adapters are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def install_specmoe_hook(enable_dedup, enable_sdd, enable_cache):
    """Install FusedMoEHook with requested SpecMoE components."""
    from adapters.fused_moe_hook import FusedMoEHook

    hook = FusedMoEHook()

    if enable_dedup:
        from adapters.triton_spec_moe import SpecFusedMoEDispatcher
        dispatcher = SpecFusedMoEDispatcher()
        hook.configure(spec_moe=dispatcher)
        print("[SpecMoE Server] Dedup dispatcher configured")

    if enable_sdd:
        from adapters.layer_early_terminator import (
            SpeculationDivergenceDetector, SDDConfig,
        )
        sdd_config = SDDConfig(
            min_check_layer=8,
            method="combined",
            consecutive_threshold=3,
        )
        sdd = SpeculationDivergenceDetector(config=sdd_config, num_layers=48)
        hook._sdd = sdd
        print("[SpecMoE Server] SDD early termination configured")

    if enable_cache:
        from adapters.expert_cache import ExpertWeightCache, ExpertCacheConfig
        cache_config = ExpertCacheConfig(
            gpu_budget_bytes=8 * 1024**3,
            eviction_policy="lru",
            enable_prefetch=True,
            pin_cpu_memory=True,
        )
        cache = ExpertWeightCache(config=cache_config)
        hook._expert_cache = cache
        print("[SpecMoE Server] Expert cache (8GB LRU) configured")

    ok = hook.install()
    if ok:
        print("[SpecMoE Server] FusedMoEHook installed successfully")
    else:
        print("[SpecMoE Server] WARNING: Hook install failed, running vanilla")

    return hook


def main():
    # ── Extract SpecMoE-specific args before passing rest to vLLM ──
    specmoe_flags = {"--specmoe-dedup", "--specmoe-sdd", "--specmoe-cache"}
    enable_dedup = "--specmoe-dedup" in sys.argv
    enable_sdd = "--specmoe-sdd" in sys.argv
    enable_cache = "--specmoe-cache" in sys.argv

    # Remove SpecMoE flags from argv so vLLM parser doesn't choke
    vllm_argv = [a for a in sys.argv[1:] if a not in specmoe_flags]

    any_specmoe = enable_dedup or enable_sdd or enable_cache

    # ── Install hook BEFORE vLLM loads anything ──
    hook = None
    if any_specmoe:
        print(f"[SpecMoE Server] Installing hook: "
              f"dedup={enable_dedup}, sdd={enable_sdd}, cache={enable_cache}")
        hook = install_specmoe_hook(enable_dedup, enable_sdd, enable_cache)
    else:
        print("[SpecMoE Server] No SpecMoE components enabled, running vanilla vLLM")

    # ── Parse vLLM args and start server ──
    from vllm.entrypoints.openai.api_server import (
        make_arg_parser, run_server,
    )

    parser = make_arg_parser()
    args = parser.parse_args(vllm_argv)

    print(f"[SpecMoE Server] Starting on port {args.port}...")
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
