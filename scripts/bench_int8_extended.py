#!/usr/bin/env python3
"""
Extended INT8 W8A16 benchmark:
1. MoE kernel at different batch sizes (1, 4, 8, 16, 32)
2. CPU→GPU pool copy bandwidth: BF16 vs INT8
3. Net effect analysis: cache hit rate × kernel cost + miss rate × copy cost
"""

import json
import time
import torch
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    moe_align_block_size,
)

HIDDEN_DIM = 2048
INTERMEDIATE = 768
N_W1 = INTERMEDIATE * 2
NUM_EXPERTS = 8
TOP_K = 8
WARMUP = 30
ITERS = 100


def quantize_int8(W_bf16):
    amax = W_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 127.0).squeeze(-1)
    W_int8 = (W_bf16 / amax * 127.0).round().clamp(-128, 127).to(torch.int8)
    return W_int8, scale.to(torch.float32)


def get_tile_config():
    return {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
    }


def bench_kernel(batch_size, int8_mode, device="cuda"):
    """Benchmark full MoE step (W1 → act → W2) for given batch size."""
    config = get_tile_config()
    E = NUM_EXPERTS

    A = torch.randn(batch_size, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    topk_weights = torch.ones(batch_size, TOP_K, dtype=torch.float32, device=device) / TOP_K
    topk_ids = torch.arange(TOP_K, device=device).unsqueeze(0).expand(batch_size, -1).to(torch.int32)

    w1_bf16 = torch.randn(E, N_W1, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E, HIDDEN_DIM, INTERMEDIATE, dtype=torch.bfloat16, device=device)

    if int8_mode:
        w1, w1_scale = quantize_int8(w1_bf16)
        w2, w2_scale = quantize_int8(w2_bf16)
        del w1_bf16, w2_bf16
    else:
        w1, w1_scale = w1_bf16, None
        w2, w2_scale = w2_bf16, None

    ct = tl.bfloat16
    N_w1 = w1.shape[1]
    K_hidden = w1.shape[2]
    act_dim = N_w1 // 2

    def step():
        C1 = torch.zeros(batch_size, TOP_K, N_w1, dtype=torch.bfloat16, device=device)
        s1, e1, n1 = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E, None)
        invoke_fused_moe_triton_kernel(
            A, w1, C1, None, w1_scale, None, s1, e1, n1,
            False, TOP_K, config, compute_type=ct,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=int8_mode, use_int4_w4a16=False,
            per_channel_quant=False,
        )
        flat = C1.reshape(-1, N_w1)
        inter = torch.nn.functional.silu(flat[:, :act_dim]) * flat[:, act_dim:]
        C2 = torch.zeros(batch_size, TOP_K, K_hidden, dtype=torch.bfloat16, device=device)
        s2, e2, n2 = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E, None)
        invoke_fused_moe_triton_kernel(
            inter, w2, C2, None, w2_scale, topk_weights, s2, e2, n2,
            True, 1, config, compute_type=ct,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=int8_mode, use_int4_w4a16=False,
            per_channel_quant=False,
        )
        return C2.sum(dim=1)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    p50_ms = sorted(times)[len(times) // 2] * 1000
    return avg_ms, p50_ms


def bench_cpu_to_gpu_copy():
    """Benchmark CPU→GPU copy for a single expert (w13 + w2)."""
    # Per-expert sizes matching Qwen3-30B-A3B
    w13_shape = (N_W1, HIDDEN_DIM)    # (1536, 2048)
    w2_shape = (HIDDEN_DIM, INTERMEDIATE)  # (2048, 768)

    # BF16 source on CPU (pinned)
    w13_cpu = torch.randn(*w13_shape, dtype=torch.bfloat16).pin_memory()
    w2_cpu = torch.randn(*w2_shape, dtype=torch.bfloat16).pin_memory()

    # INT8 source on CPU (pinned)
    w13_int8_cpu = torch.randint(-128, 127, w13_shape, dtype=torch.int8).pin_memory()
    w2_int8_cpu = torch.randint(-128, 127, w2_shape, dtype=torch.int8).pin_memory()
    w13_scale_cpu = torch.randn(w13_shape[0], dtype=torch.float32).pin_memory()
    w2_scale_cpu = torch.randn(w2_shape[0], dtype=torch.float32).pin_memory()

    # GPU targets
    w13_gpu_bf16 = torch.empty_like(w13_cpu, device="cuda")
    w2_gpu_bf16 = torch.empty_like(w2_cpu, device="cuda")
    w13_gpu_int8 = torch.empty_like(w13_int8_cpu, device="cuda")
    w2_gpu_int8 = torch.empty_like(w2_int8_cpu, device="cuda")
    w13_scale_gpu = torch.empty_like(w13_scale_cpu, device="cuda")
    w2_scale_gpu = torch.empty_like(w2_scale_cpu, device="cuda")

    results = {}

    # BF16 copy
    for _ in range(20):
        w13_gpu_bf16.copy_(w13_cpu, non_blocking=True)
        w2_gpu_bf16.copy_(w2_cpu, non_blocking=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w13_gpu_bf16.copy_(w13_cpu, non_blocking=True)
        w2_gpu_bf16.copy_(w2_cpu, non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    bf16_ms = sum(times) / len(times) * 1000
    bf16_bytes = (w13_cpu.nelement() + w2_cpu.nelement()) * 2
    bf16_bw = bf16_bytes / 1e9 / (bf16_ms / 1000)
    results["bf16"] = {"ms": round(bf16_ms, 4), "bytes": bf16_bytes, "bw_gb_s": round(bf16_bw, 1)}

    # INT8 copy
    for _ in range(20):
        w13_gpu_int8.copy_(w13_int8_cpu, non_blocking=True)
        w2_gpu_int8.copy_(w2_int8_cpu, non_blocking=True)
        w13_scale_gpu.copy_(w13_scale_cpu, non_blocking=True)
        w2_scale_gpu.copy_(w2_scale_cpu, non_blocking=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w13_gpu_int8.copy_(w13_int8_cpu, non_blocking=True)
        w2_gpu_int8.copy_(w2_int8_cpu, non_blocking=True)
        w13_scale_gpu.copy_(w13_scale_cpu, non_blocking=True)
        w2_scale_gpu.copy_(w2_scale_cpu, non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    int8_ms = sum(times) / len(times) * 1000
    int8_bytes = (w13_int8_cpu.nelement() + w2_int8_cpu.nelement()) * 1 + (w13_scale_cpu.nelement() + w2_scale_cpu.nelement()) * 4
    int8_bw = int8_bytes / 1e9 / (int8_ms / 1000)
    results["int8"] = {"ms": round(int8_ms, 4), "bytes": int8_bytes, "bw_gb_s": round(int8_bw, 1)}
    results["speedup"] = round(bf16_ms / int8_ms, 3)

    return results


def main():
    torch.manual_seed(42)
    print("=" * 60)
    print("  INT8 W8A16 Extended Benchmark")
    print("=" * 60)

    # Part 1: Kernel at different batch sizes
    print("\n--- Part 1: MoE Kernel Latency (W1+act+W2) ---")
    print(f"  {'Batch':>5}  {'BF16 (ms)':>10}  {'INT8 (ms)':>10}  {'Speedup':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")

    kernel_results = []
    for bs in [1, 4, 8, 16, 32]:
        bf16_avg, _ = bench_kernel(bs, False)
        int8_avg, _ = bench_kernel(bs, True)
        spd = bf16_avg / int8_avg if int8_avg > 0 else 0
        print(f"  {bs:>5}  {bf16_avg:>10.3f}  {int8_avg:>10.3f}  {spd:>7.3f}×")
        kernel_results.append({
            "batch_size": bs, "bf16_ms": round(bf16_avg, 4),
            "int8_ms": round(int8_avg, 4), "speedup": round(spd, 4),
        })

    # Part 2: CPU→GPU copy
    print("\n--- Part 2: CPU→GPU Expert Copy (per expert) ---")
    copy_results = bench_cpu_to_gpu_copy()
    print(f"  BF16 copy: {copy_results['bf16']['ms']:.3f} ms  ({copy_results['bf16']['bytes']/1e6:.2f} MB, {copy_results['bf16']['bw_gb_s']:.1f} GB/s)")
    print(f"  INT8 copy: {copy_results['int8']['ms']:.3f} ms  ({copy_results['int8']['bytes']/1e6:.2f} MB, {copy_results['int8']['bw_gb_s']:.1f} GB/s)")
    print(f"  Copy speedup: {copy_results['speedup']:.3f}×")

    # Part 3: Net analysis
    print("\n--- Part 3: Net Effect Analysis ---")
    # With current ELMM: ~17 slots BF16, cache hit ~80%. INT8 → 34 slots, hit ~90%+
    bf16_kern_ms = kernel_results[0]["bf16_ms"]  # batch=1
    int8_kern_ms = kernel_results[0]["int8_ms"]
    bf16_copy_ms = copy_results["bf16"]["ms"]
    int8_copy_ms = copy_results["int8"]["ms"]

    for hit_rate_bf16, hit_rate_int8, label in [
        (0.80, 0.90, "Conservative (80→90%)"),
        (0.70, 0.85, "Moderate (70→85%)"),
        (0.60, 0.80, "Aggressive (60→80%)"),
    ]:
        # Cost per expert = hit_rate * kernel + (1-hit_rate) * (copy + kernel)
        bf16_cost = hit_rate_bf16 * bf16_kern_ms + (1 - hit_rate_bf16) * (bf16_copy_ms + bf16_kern_ms)
        int8_cost = hit_rate_int8 * int8_kern_ms + (1 - hit_rate_int8) * (int8_copy_ms + int8_kern_ms)
        net_speedup = bf16_cost / int8_cost if int8_cost > 0 else 0
        print(f"  {label}: BF16={bf16_cost:.3f} ms, INT8={int8_cost:.3f} ms → {net_speedup:.3f}× ({(net_speedup-1)*100:+.1f}%)")

    # Save all results
    all_results = {
        "kernel_scaling": kernel_results,
        "cpu_to_gpu_copy": copy_results,
        "analysis": {
            "bf16_kernel_1tok_ms": bf16_kern_ms,
            "int8_kernel_1tok_ms": int8_kern_ms,
            "bf16_copy_ms": bf16_copy_ms,
            "int8_copy_ms": int8_copy_ms,
        },
    }
    out_path = "/root/MoE-SD/results/int8_extended_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
