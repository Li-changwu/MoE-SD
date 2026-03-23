#!/usr/bin/env python3
"""Triton FusedMoE kernel config sweep for ELMM on RTX A6000.

Benchmarks different tile configurations for the exact problem dimensions
used in ELMM Direct Dispatch:
  W1: [17, 1536, 2048] bf16  (gate+up projection)
  W2: [17, 2048, 768]  bf16  (down projection)
  M=4, top_k=8 (EAGLE-3 K=3 decode)

Usage:
  python scripts/triton_config_sweep.py [--warmup 50] [--iters 200]
"""
import argparse
import itertools
import json
import sys
import time

import torch
import triton.language as tl

# -- vLLM imports --
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def make_test_data(M: int, top_k: int, E: int, N_w1: int, K: int, act_dim: int,
                   device: str = "cuda", dtype=torch.bfloat16):
    """Create synthetic tensors matching ELMM decode path."""
    hidden = torch.randn(M, K, device=device, dtype=dtype)
    w1 = torch.randn(E, N_w1, K, device=device, dtype=dtype)
    w2 = torch.randn(E, K, act_dim, device=device, dtype=dtype)

    # Simulate routing: each token picks top_k experts from E pool slots
    topk_ids = torch.stack([
        torch.randperm(E, device=device)[:top_k] for _ in range(M)
    ]).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(M, top_k, device=device, dtype=torch.float32), dim=-1
    )

    # Pre-allocate 3D outputs (kernel accesses C.stride(2))
    out_w1 = torch.empty(M, top_k, N_w1, device=device, dtype=dtype)
    out_w2 = torch.empty(M, top_k, K, device=device, dtype=dtype)

    return hidden, w1, w2, topk_ids, topk_weights, out_w1, out_w2


def bench_config(config: dict, hidden, w1, w2, topk_ids, topk_weights,
                 out_w1, out_w2, E: int, top_k: int,
                 warmup: int, iters: int) -> tuple[float, float]:
    """Benchmark a single tile config. Returns (w1_us, w2_us) median times."""

    M = hidden.size(0)
    N_w1 = w1.size(1)
    K = w1.size(2)
    act_dim = w2.size(2)

    bsm = config["BLOCK_SIZE_M"]

    # Pre-compute alignment (same for all iterations)
    sorted_ids, expert_ids, num_tok_padded = moe_align_block_size(
        topk_ids, bsm, E, None
    )

    compute_type = tl.bfloat16

    flat_act_w2 = torch.randn(M * top_k, act_dim, device="cuda", dtype=hidden.dtype)

    def run_w1():
        invoke_fused_moe_triton_kernel(
            hidden, w1, out_w1,
            None, None, None,
            sorted_ids, expert_ids, num_tok_padded,
            False, top_k, config, compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )

    def run_w2():
        invoke_fused_moe_triton_kernel(
            flat_act_w2, w2, out_w2,
            None, None, topk_weights,
            sorted_ids, expert_ids, num_tok_padded,
            True, 1, config, compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
        )

    # Warmup
    for _ in range(warmup):
        run_w1()
        run_w2()
    torch.cuda.synchronize()

    # Benchmark W1
    w1_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_w1()
        end.record()
        torch.cuda.synchronize()
        w1_times.append(start.elapsed_time(end) * 1000)  # μs

    # Benchmark W2
    w2_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_w2()
        end.record()
        torch.cuda.synchronize()
        w2_times.append(start.elapsed_time(end) * 1000)  # μs

    w1_med = sorted(w1_times)[len(w1_times) // 2]
    w2_med = sorted(w2_times)[len(w2_times) // 2]
    return w1_med, w2_med


def generate_configs():
    """Generate candidate tile configurations."""
    candidates = []

    block_ms = [16, 32]
    block_ns = [64, 128, 256]
    block_ks = [64, 128, 256]
    group_ms = [1, 8, 16]
    warp_counts = [4, 8]
    stage_counts = [2, 3, 4, 5]

    for bsm, bsn, bsk, gm, warps, stages in itertools.product(
        block_ms, block_ns, block_ks, group_ms, warp_counts, stage_counts
    ):
        # Shared memory check (GA102: 100 KB max)
        a_smem = bsm * bsk * 2  # bf16
        b_smem = bsn * bsk * 2
        total_smem = (a_smem + b_smem) * stages
        if total_smem > 99 * 1024:
            continue

        # K must divide hidden_size (2048)
        if 2048 % bsk != 0:
            continue

        # N_w1=1536 and act_dim=768 must be >= bsn (otherwise wasted blocks)
        # Actually kernel handles this via masking, but very wasteful if bsn > N
        if bsn > 1536:
            continue

        candidates.append({
            "BLOCK_SIZE_M": bsm,
            "BLOCK_SIZE_N": bsn,
            "BLOCK_SIZE_K": bsk,
            "GROUP_SIZE_M": gm,
            "num_warps": warps,
            "num_stages": stages,
        })

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=20,
                        help="Show top N configs")
    args = parser.parse_args()

    # Problem dimensions
    M, top_k, E = 4, 8, 17
    N_w1, K, act_dim = 1536, 2048, 768

    print(f"=== Triton FusedMoE Config Sweep ===", file=sys.stderr)
    print(f"Problem: M={M}, top_k={top_k}, E={E}, "
          f"W1=[{E},{N_w1},{K}], W2=[{E},{K},{act_dim}] bf16",
          file=sys.stderr)
    print(f"Warmup={args.warmup}, Iters={args.iters}", file=sys.stderr)

    hidden, w1, w2, topk_ids, topk_weights, out_w1, out_w2 = make_test_data(
        M, top_k, E, N_w1, K, act_dim
    )

    configs = generate_configs()
    print(f"Testing {len(configs)} configurations...", file=sys.stderr)

    results = []
    current_config = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    }

    # Benchmark current config first as baseline
    try:
        w1_us, w2_us = bench_config(
            current_config, hidden, w1, w2, topk_ids, topk_weights,
            out_w1, out_w2, E, top_k, args.warmup, args.iters
        )
        baseline_total = w1_us + w2_us
        print(f"\nBASELINE: W1={w1_us:.1f}μs W2={w2_us:.1f}μs "
              f"Total={baseline_total:.1f}μs", file=sys.stderr)
    except Exception as e:
        print(f"BASELINE FAILED: {e}", file=sys.stderr)
        baseline_total = float("inf")

    for i, cfg in enumerate(configs):
        tag = (f"M{cfg['BLOCK_SIZE_M']}_N{cfg['BLOCK_SIZE_N']}_"
               f"K{cfg['BLOCK_SIZE_K']}_G{cfg['GROUP_SIZE_M']}_"
               f"W{cfg['num_warps']}_S{cfg['num_stages']}")
        try:
            w1_us, w2_us = bench_config(
                cfg, hidden, w1, w2, topk_ids, topk_weights,
                out_w1, out_w2, E, top_k, args.warmup, args.iters
            )
            total = w1_us + w2_us
            speedup = baseline_total / total if total > 0 else 0
            results.append({
                "config": cfg,
                "tag": tag,
                "w1_us": round(w1_us, 1),
                "w2_us": round(w2_us, 1),
                "total_us": round(total, 1),
                "speedup": round(speedup, 3),
            })
            marker = " ★" if speedup > 1.05 else ""
            if (i + 1) % 10 == 0 or speedup > 1.05:
                print(f"  [{i+1}/{len(configs)}] {tag}: "
                      f"{total:.1f}μs ({speedup:.3f}×){marker}",
                      file=sys.stderr)
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] {tag}: FAILED ({e})",
                  file=sys.stderr)

    # Sort by total time
    results.sort(key=lambda r: r["total_us"])

    print(f"\n{'='*80}", file=sys.stderr)
    print(f"TOP {args.top_n} CONFIGURATIONS (baseline={baseline_total:.1f}μs):",
          file=sys.stderr)
    print(f"{'Rank':>4} {'Tag':>35} {'W1(μs)':>8} {'W2(μs)':>8} "
          f"{'Total':>8} {'Speedup':>8}", file=sys.stderr)
    for i, r in enumerate(results[:args.top_n]):
        marker = " ★" if r["speedup"] > 1.05 else ""
        print(f"{i+1:4d} {r['tag']:>35} {r['w1_us']:8.1f} {r['w2_us']:8.1f} "
              f"{r['total_us']:8.1f} {r['speedup']:7.3f}×{marker}",
              file=sys.stderr)

    # Output JSON for programmatic use
    output = {
        "baseline": {
            "config": current_config,
            "total_us": round(baseline_total, 1),
        },
        "best": results[0] if results else None,
        "top_configs": results[:args.top_n],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
