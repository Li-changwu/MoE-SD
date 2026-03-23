#!/usr/bin/env python3
"""Focused Triton FusedMoE kernel benchmark with proper JIT warm-up.

Tests a curated set of promising tile configurations on the exact
ELMM decode problem (M=4, E=17, top_k=8).

Each config gets heavy warmup to ensure Triton JIT compilation is complete
before measurement begins.
"""
import json
import sys
import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def bench_one_config(name: str, config: dict, hidden, w1, w2,
                     topk_ids, topk_weights, out_w1, out_w2,
                     E: int, top_k: int, warmup: int = 200, iters: int = 500):
    """Benchmark a single config with heavy warmup to eliminate JIT noise."""
    M = hidden.size(0)
    K = w1.size(2)
    N_w1 = w1.size(1)
    act_dim = w2.size(2)
    bsm = config["BLOCK_SIZE_M"]

    sorted_ids, expert_ids, num_tok_padded = moe_align_block_size(
        topk_ids, bsm, E, None
    )
    flat_act = torch.randn(M * top_k, act_dim, device="cuda", dtype=hidden.dtype)
    compute_type = tl.bfloat16

    def call_w1():
        invoke_fused_moe_triton_kernel(
            hidden, w1, out_w1,
            None, None, None,
            sorted_ids, expert_ids, num_tok_padded,
            False, top_k, config, compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )

    def call_w2():
        invoke_fused_moe_triton_kernel(
            flat_act, w2, out_w2,
            None, None, topk_weights,
            sorted_ids, expert_ids, num_tok_padded,
            True, 1, config, compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )

    # Heavy warmup — ensures JIT is done and GPU is at steady thermal state
    for _ in range(warmup):
        call_w1()
        call_w2()
    torch.cuda.synchronize()

    # Benchmark combined (W1 + W2) as one unit — this is the per-layer cost
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call_w1()
        call_w2()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # μs

    times.sort()
    # Use p10, p50, p90 for stability analysis
    p10 = times[len(times) // 10]
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    mean = sum(times) / len(times)

    return {"name": name, "p10": p10, "p50": p50, "p90": p90, "mean": mean,
            "config": config}


def main():
    M, top_k, E = 4, 8, 17
    N_w1, K, act_dim = 1536, 2048, 768

    print(f"=== Focused Triton Config Benchmark ===", file=sys.stderr)
    print(f"M={M}, top_k={top_k}, E={E}, warmup=200, iters=500", file=sys.stderr)
    print(f"W1=[{E},{N_w1},{K}] W2=[{E},{K},{act_dim}] bf16\n", file=sys.stderr)

    hidden = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, N_w1, K, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(E, K, act_dim, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.stack([
        torch.randperm(E, device="cuda")[:top_k] for _ in range(M)
    ]).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(M, top_k, device="cuda", dtype=torch.float32), dim=-1
    )
    out_w1 = torch.empty(M, top_k, N_w1, device="cuda", dtype=torch.bfloat16)
    out_w2 = torch.empty(M, top_k, K, device="cuda", dtype=torch.bfloat16)

    # Curated configs to test
    configs = {
        # Current baseline
        "baseline_M16_N64_K128_S2_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        },
        # Larger K (fewer loop iters: 32→8)
        "K256_S2_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        },
        # More stages (better pipelining for memory-bound)
        "K128_S3_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
        },
        # More stages + wider N
        "N128_K64_S3_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
        },
        # More warps (more memory requests)
        "K128_S2_W8": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
        },
        # Wider N + 8 warps
        "N128_K64_S2_W8": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
        },
        # Big K + 8 warps
        "K256_S2_W8": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
        },
        # Max pipelining within 48KB SMEM
        "K64_S4_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4,
        },
        # Max pipelining with wider N (needs 100KB SMEM)
        "N128_K64_S5_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5,
        },
        # 100KB SMEM: K128 + 4 stages
        "K128_S4_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4,
        },
        # Wider with grouping
        "N128_K128_S2_W8_G8": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2,
        },
        # BSM=32 (more tokens per block)
        "M32_N64_K128_S2_W4": {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        },
        # Very wide N (256) with small K
        "N256_K64_S2_W8": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
        },
        # Aggressive: N128_K128_S2_W4 (72KB SMEM)
        "N128_K128_S2_W4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        },
    }

    results = []
    for name, cfg in configs.items():
        print(f"  Testing {name}...", end="", file=sys.stderr, flush=True)
        try:
            r = bench_one_config(name, cfg, hidden, w1, w2,
                                 topk_ids, topk_weights, out_w1, out_w2,
                                 E, top_k)
            results.append(r)
            print(f" p50={r['p50']:.1f}μs mean={r['mean']:.1f}μs", file=sys.stderr)
        except Exception as e:
            print(f" FAILED: {e}", file=sys.stderr)

    results.sort(key=lambda r: r["p50"])
    baseline_p50 = next(r["p50"] for r in results if "baseline" in r["name"])

    print(f"\n{'='*75}", file=sys.stderr)
    print(f"RESULTS (sorted by p50, baseline={baseline_p50:.1f}μs):", file=sys.stderr)
    print(f"{'Rank':>4} {'Name':>35} {'p10':>8} {'p50':>8} {'p90':>8} {'mean':>8} {'vs_base':>8}",
          file=sys.stderr)
    for i, r in enumerate(results):
        speedup = baseline_p50 / r["p50"]
        marker = " ★" if speedup > 1.03 else (" ." if speedup > 0.97 else " ✗")
        print(f"{i+1:4d} {r['name']:>35} {r['p10']:8.1f} {r['p50']:8.1f} "
              f"{r['p90']:8.1f} {r['mean']:8.1f} {speedup:7.3f}×{marker}",
              file=sys.stderr)

    # HBM utilization for best config
    total_bytes = E * N_w1 * K * 2 + E * K * act_dim * 2
    best = results[0]
    hbm_util = total_bytes / (best["p50"] * 1e-6 * 768e9) * 100
    print(f"\nBest: {best['name']} → p50={best['p50']:.1f}μs "
          f"(HBM util={hbm_util:.1f}%)", file=sys.stderr)

    # JSON output
    print(json.dumps({
        "baseline_p50_us": baseline_p50,
        "best": {"name": best["name"], "p50_us": best["p50"],
                 "config": best["config"]},
        "all_results": [{k: v for k, v in r.items()} for r in results],
    }, indent=2))


if __name__ == "__main__":
    main()
