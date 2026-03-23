#!/usr/bin/env python3
"""
Micro-benchmark: INT8-W8A16 vs BF16 MoE kernel throughput.

Directly measures the Triton FusedMoE kernel with synthetic expert weights
matching Qwen3-30B-A3B dimensions:
  w13: [num_experts, 1536, 2048]  (gate+up fused)
  w2:  [num_experts, 2048, 768]

Reports:
  - Kernel latency (ms)
  - Effective bandwidth (GB/s)
  - Speedup ratio INT8/BF16
"""

import json
import time
import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    moe_align_block_size,
)

# ---------- Qwen3-30B-A3B geometry ----------
HIDDEN_DIM = 2048
INTERMEDIATE = 768
N_W1 = INTERMEDIATE * 2   # 1536  (gate + up fused)
NUM_EXPERTS = 8            # pool slots (simulating top_k batch)
TOP_K = 8
BATCH_TOKENS = 1           # single-token decode (MoE step)

WARMUP = 50
ITERS = 200


def quantize_int8(W_bf16: torch.Tensor):
    """Symmetric per-channel absmax quantization."""
    amax = W_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 127.0).squeeze(-1)          # [E, N]
    W_int8 = (W_bf16 / amax * 127.0).round().clamp(-128, 127).to(torch.int8)
    return W_int8, scale.to(torch.float32)


def get_tile_config():
    """Default tile config for A6000."""
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    }


def run_kernel(A, B, B_scale, topk_weights, topk_ids,
               num_experts, top_k, config, int8_mode, N_out,
               mul_routed_weight=False):
    """Run a single fused_moe kernel invocation."""
    M = A.shape[0]
    C = torch.zeros(M, top_k, N_out, dtype=A.dtype, device=A.device)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts, None
    )
    ct = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    invoke_fused_moe_triton_kernel(
        A, B, C,
        None, B_scale,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight, top_k, config,
        compute_type=ct,
        use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=int8_mode, use_int4_w4a16=False,
        per_channel_quant=False,
    )
    return C


def bench_one(label, A, w1, w1_scale, w2, w2_scale,
              topk_weights, topk_ids, num_experts, top_k, config,
              int8_mode):
    """Full MoE step: W1 → silu_and_mul → W2, timed."""
    M = A.shape[0]
    N_w1 = w1.shape[1]
    act_dim = N_w1 // 2
    K_hidden = w1.shape[2]

    # Warmup
    for _ in range(WARMUP):
        c1 = run_kernel(A, w1, w1_scale, None, topk_ids,
                        num_experts, top_k, config, int8_mode, N_w1,
                        mul_routed_weight=False)
        flat = c1.reshape(-1, N_w1)
        gate = flat[:, :act_dim]
        up = flat[:, act_dim:]
        inter = torch.nn.functional.silu(gate) * up
        c2 = run_kernel(inter, w2, w2_scale, topk_weights, topk_ids,
                        num_experts, 1, config, int8_mode, K_hidden,
                        mul_routed_weight=True)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c1 = run_kernel(A, w1, w1_scale, None, topk_ids,
                        num_experts, top_k, config, int8_mode, N_w1,
                        mul_routed_weight=False)
        flat = c1.reshape(-1, N_w1)
        gate = flat[:, :act_dim]
        up = flat[:, act_dim:]
        inter = torch.nn.functional.silu(gate) * up
        c2 = run_kernel(inter, w2, w2_scale, topk_weights, topk_ids,
                        num_experts, 1, config, int8_mode, K_hidden,
                        mul_routed_weight=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    avg_ms = sum(times_ms) / len(times_ms)
    p50 = sorted(times_ms)[len(times_ms) // 2]
    p99 = sorted(times_ms)[int(len(times_ms) * 0.99)]

    # Bytes read: W1 + W2 weights (dominant term; activations negligible)
    if int8_mode:
        bytes_w1 = num_experts * N_w1 * K_hidden * 1  # INT8
        bytes_w2 = num_experts * K_hidden * act_dim * 1
        bytes_scale = num_experts * (N_w1 + K_hidden) * 4  # FP32 scale
        total_bytes = bytes_w1 + bytes_w2 + bytes_scale
    else:
        bytes_w1 = num_experts * N_w1 * K_hidden * 2  # BF16
        bytes_w2 = num_experts * K_hidden * act_dim * 2
        total_bytes = bytes_w1 + bytes_w2

    bw_gb_s = (total_bytes / 1e9) / (avg_ms / 1000)

    print(f"\n  [{label}]")
    print(f"    Latency:   avg={avg_ms:.3f} ms  p50={p50:.3f} ms  p99={p99:.3f} ms")
    print(f"    Weights:   {total_bytes / 1e6:.2f} MB")
    print(f"    Bandwidth: {bw_gb_s:.1f} GB/s")

    return {
        "label": label,
        "avg_ms": round(avg_ms, 4),
        "p50_ms": round(p50, 4),
        "p99_ms": round(p99, 4),
        "weight_bytes": total_bytes,
        "bw_gb_s": round(bw_gb_s, 1),
    }


def main():
    torch.manual_seed(42)
    device = "cuda"
    config = get_tile_config()
    E = NUM_EXPERTS

    print(f"=== INT8 W8A16 vs BF16 MoE Kernel Micro-Benchmark ===")
    print(f"  Experts: {E} (top_k={TOP_K})")
    print(f"  Geometry: w13=[{E}, {N_W1}, {HIDDEN_DIM}], w2=[{E}, {HIDDEN_DIM}, {INTERMEDIATE}]")
    print(f"  Batch: {BATCH_TOKENS} token(s), Warmup: {WARMUP}, Iters: {ITERS}")

    # Input activations
    A = torch.randn(BATCH_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    topk_weights = torch.ones(BATCH_TOKENS, TOP_K, dtype=torch.float32, device=device) / TOP_K
    topk_ids = torch.arange(TOP_K, device=device).unsqueeze(0).expand(BATCH_TOKENS, -1).to(torch.int32)

    # BF16 weights
    w1_bf16 = torch.randn(E, N_W1, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E, HIDDEN_DIM, INTERMEDIATE, dtype=torch.bfloat16, device=device)

    # INT8 weights + scales
    w1_int8, w1_scale = quantize_int8(w1_bf16)
    w2_int8, w2_scale = quantize_int8(w2_bf16)

    # Verify shapes
    print(f"  w1_bf16: {w1_bf16.shape}, w1_int8: {w1_int8.shape}, w1_scale: {w1_scale.shape}")
    print(f"  w2_bf16: {w2_bf16.shape}, w2_int8: {w2_int8.shape}, w2_scale: {w2_scale.shape}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  GPU free: {free_gb:.2f} GB")

    # ---- BF16 ----
    bf16_result = bench_one(
        "BF16", A, w1_bf16, None, w2_bf16, None,
        topk_weights, topk_ids, E, TOP_K, config, False,
    )

    # ---- INT8 W8A16 ----
    int8_result = bench_one(
        "INT8-W8A16", A, w1_int8, w1_scale, w2_int8, w2_scale,
        topk_weights, topk_ids, E, TOP_K, config, True,
    )

    # ---- Summary ----
    speedup = bf16_result["avg_ms"] / int8_result["avg_ms"] if int8_result["avg_ms"] > 0 else 0
    bw_ratio = int8_result["bw_gb_s"] / bf16_result["bw_gb_s"] if bf16_result["bw_gb_s"] > 0 else 0
    size_ratio = bf16_result["weight_bytes"] / int8_result["weight_bytes"]

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  BF16  latency: {bf16_result['avg_ms']:.3f} ms   BW: {bf16_result['bw_gb_s']:.1f} GB/s")
    print(f"  INT8  latency: {int8_result['avg_ms']:.3f} ms   BW: {int8_result['bw_gb_s']:.1f} GB/s")
    print(f"  Speedup:       {speedup:.3f}× ({(speedup - 1) * 100:+.1f}%)")
    print(f"  Weight size:   BF16={bf16_result['weight_bytes']/1e6:.2f} MB → INT8={int8_result['weight_bytes']/1e6:.2f} MB ({size_ratio:.2f}× reduction)")
    print(f"  BW efficiency: {bw_ratio:.2f}×")

    # Save results
    results = {
        "bf16": bf16_result,
        "int8_w8a16": int8_result,
        "speedup": round(speedup, 4),
        "weight_size_ratio": round(size_ratio, 2),
        "bw_efficiency_ratio": round(bw_ratio, 2),
        "config": {
            "num_experts": E,
            "top_k": TOP_K,
            "hidden_dim": HIDDEN_DIM,
            "intermediate": INTERMEDIATE,
            "batch_tokens": BATCH_TOKENS,
            "warmup": WARMUP,
            "iters": ITERS,
        },
    }

    out_path = "/root/MoE-SD/results/int8_kernel_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
