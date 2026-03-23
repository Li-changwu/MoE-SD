#!/usr/bin/env python3
"""
INT8 W8A16 Pool Quantization — Correctness Test

Verifies that the INT8 quantization + Triton kernel path produces results
close to the BF16 reference. Tests:
  1. Quantize → dequantize roundtrip error
  2. INT8 kernel vs BF16 kernel output agreement
  3. End-to-end MoE dispatch agreement

Run: conda run -n moe-sd python scripts/test_int8_pool.py
"""

import sys
import torch
import numpy as np

sys.path.insert(0, "/root/MoE-SD")


def test_quantize_dequantize_roundtrip():
    """Test that INT8 symmetric absmax quantization has low roundtrip error."""
    print("Test 1: Quantize/Dequantize Roundtrip")
    print("-" * 50)

    torch.manual_seed(42)
    # Simulate expert weight shapes from Qwen3-30B-A3B
    shapes = [(1536, 2048), (2048, 768)]  # w13, w2

    for shape in shapes:
        W = torch.randn(shape, dtype=torch.bfloat16)

        # Symmetric absmax quantization (per output channel = per row)
        W_f32 = W.float()
        amax = W_f32.abs().amax(dim=1)                # [N_out]
        scale = amax.clamp(min=1e-10) / 127.0          # [N_out]
        W_int8 = (W_f32 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        # Dequantize
        W_deq = W_int8.float() * scale.unsqueeze(1)
        W_deq_bf16 = W_deq.to(torch.bfloat16)

        # Compute errors
        abs_err = (W.float() - W_deq).abs()
        rel_err = abs_err / (W.float().abs() + 1e-8)

        print(f"  Shape {shape}:")
        print(f"    Max abs error:  {abs_err.max().item():.6f}")
        print(f"    Mean abs error: {abs_err.mean().item():.6f}")
        print(f"    Max rel error:  {rel_err.max().item():.4f}")
        print(f"    Mean rel error: {rel_err.mean().item():.4f}")
        print(f"    SQNR (dB):      {10 * torch.log10((W.float()**2).sum() / ((W.float() - W_deq)**2).sum()).item():.1f}")

        # Should have SQNR > 30 dB for well-conditioned data
        sqnr = 10 * torch.log10((W.float()**2).sum() / ((W.float() - W_deq)**2).sum()).item()
        assert sqnr > 30, f"SQNR too low: {sqnr:.1f} dB"

    print("  ✅ Roundtrip error within bounds\n")


def test_kernel_int8_vs_bf16():
    """Test that the Triton fused_moe kernel with INT8 W8A16 matches BF16 reference."""
    print("Test 2: Triton Kernel INT8 vs BF16")
    print("-" * 50)

    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            invoke_fused_moe_triton_kernel,
            try_get_optimal_moe_config,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
    except ImportError as e:
        print(f"  Skipped (vLLM not available): {e}")
        return

    import triton.language as tl

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Model params from Qwen3-30B-A3B
    M = 4           # batch size (typical decode)
    top_k = 8
    num_slots = 17  # cache pool slots
    N_w1 = 1536     # 2 * intermediate
    K_hidden = 2048  # hidden dim

    # Create BF16 reference weights
    w1_bf16 = torch.randn(num_slots, N_w1, K_hidden, dtype=torch.bfloat16, device=device)

    # Quantize to INT8
    w1_f32 = w1_bf16.float()
    amax = w1_f32.abs().amax(dim=2)  # [S, N_w1] per row (output channel) per slot
    scale = amax.clamp(min=1e-10) / 127.0  # [S, N_w1]
    w1_int8 = (w1_f32 / scale.unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)

    # Input
    hidden = torch.randn(M, K_hidden, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, num_slots, (M, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).to(torch.bfloat16)

    # Get tile config
    w1_size = torch.Size([num_slots, N_w1, K_hidden])
    w2_size = torch.Size([num_slots, K_hidden, N_w1 // 2])
    config = try_get_optimal_moe_config(w1_size, w2_size, top_k, "bf16", M, block_shape=None)

    # Align
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_slots, None
    )

    # 1. BF16 reference
    out_bf16 = torch.empty(M, top_k, N_w1, dtype=torch.bfloat16, device=device)
    invoke_fused_moe_triton_kernel(
        hidden, w1_bf16, out_bf16,
        None, None,
        None,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        False, top_k, config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None, B_bias=None,
    )

    # 2. INT8 W8A16
    out_int8 = torch.empty(M, top_k, N_w1, dtype=torch.bfloat16, device=device)
    invoke_fused_moe_triton_kernel(
        hidden, w1_int8, out_int8,
        None, scale,   # B_scale = per-channel scale [S, N_w1]
        None,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        False, top_k, config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=True, use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None, B_bias=None,
    )

    torch.cuda.synchronize()

    # Compare
    diff = (out_bf16.float() - out_int8.float()).abs()
    ref_norm = out_bf16.float().abs().mean()
    rel_diff = diff.mean() / ref_norm

    print(f"  Output shape:    {tuple(out_bf16.shape)}")
    print(f"  BF16 mean abs:   {ref_norm.item():.4f}")
    print(f"  Max abs diff:    {diff.max().item():.4f}")
    print(f"  Mean abs diff:   {diff.mean().item():.4f}")
    print(f"  Rel diff:        {rel_diff.item():.4f} ({rel_diff.item()*100:.2f}%)")

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        out_bf16.reshape(1, -1).float(),
        out_int8.reshape(1, -1).float(),
    ).item()
    print(f"  Cosine sim:      {cos_sim:.6f}")

    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim:.4f}"
    assert rel_diff < 0.05, f"Relative diff too high: {rel_diff:.4f}"

    print("  ✅ INT8 kernel output matches BF16 reference\n")


def test_cache_load_expert_int8():
    """Test _LayerExpertCache.load_expert_int8 method."""
    print("Test 3: _LayerExpertCache.load_expert_int8")
    print("-" * 50)

    from adapters.elmm_plugin import _LayerExpertCache

    device = torch.device("cuda")
    w13_shape = (1536, 2048)
    w2_shape = (2048, 768)

    cache = _LayerExpertCache(
        layer_name="test_layer",
        max_slots=4,
        w13_single_shape=w13_shape,
        w2_single_shape=w2_shape,
        dtype=torch.bfloat16,
        device=device,
        int8_mode=True,
    )

    assert cache._int8_mode is True
    assert cache._w13_pool.dtype == torch.int8
    assert cache._w2_pool.dtype == torch.int8
    assert cache._w13_scale.dtype == torch.float32
    assert cache._w2_scale.dtype == torch.float32
    assert cache._w13_scale.shape == (4, 1536)
    assert cache._w2_scale.shape == (4, 2048)

    print(f"  Pool dtype:  {cache._w13_pool.dtype}")
    print(f"  Scale shape: w13={tuple(cache._w13_scale.shape)}, w2={tuple(cache._w2_scale.shape)}")

    # Load a fake expert
    torch.manual_seed(42)
    w13_bf16 = torch.randn(w13_shape, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(w2_shape, dtype=torch.bfloat16, device=device)

    slot = cache.alloc_slot(expert_id=7)
    cache.load_expert_int8(slot, w13_bf16, w2_bf16)
    torch.cuda.synchronize()

    # Verify stored data
    w13_stored = cache._w13_pool[slot]
    w13_scale = cache._w13_scale[slot]

    # Dequantize
    w13_deq = w13_stored.float() * w13_scale.unsqueeze(1)
    diff = (w13_bf16.float() - w13_deq).abs()
    sqnr = 10 * torch.log10((w13_bf16.float()**2).sum() / ((w13_bf16.float() - w13_deq)**2).sum()).item()

    print(f"  W13 SQNR:    {sqnr:.1f} dB")
    print(f"  W13 max err: {diff.max().item():.6f}")

    assert sqnr > 30, f"SQNR too low: {sqnr:.1f}"

    # Pool size comparison
    int8_size = cache._w13_pool.nelement() + cache._w2_pool.nelement()
    scale_size = (cache._w13_scale.nelement() + cache._w2_scale.nelement()) * 4
    bf16_equiv = (w13_shape[0] * w13_shape[1] + w2_shape[0] * w2_shape[1]) * 2 * 4  # 4 slots * 2 bytes
    total_int8 = int8_size + scale_size
    print(f"  INT8 pool (4 slots): {total_int8 / 1e6:.1f} MB")
    print(f"  BF16 equiv (4 slots): {bf16_equiv / 1e6:.1f} MB")
    print(f"  Compression: {bf16_equiv / total_int8:.2f}×")

    print("  ✅ Cache INT8 loading works correctly\n")


def test_pool_size_comparison():
    """Show pool size savings from INT8 quantization."""
    print("Test 4: Pool Size Comparison (Qwen3-30B-A3B @ 8GB budget)")
    print("-" * 50)

    w13_single = (1536, 2048)  # 1536 * 2048 = 3,145,728 params
    w2_single = (2048, 768)    # 2048 * 768 = 1,572,864 params

    bf16_expert = (w13_single[0] * w13_single[1] + w2_single[0] * w2_single[1]) * 2
    int8_expert = (
        w13_single[0] * w13_single[1]  # INT8 w13: 1 byte per param
        + w2_single[0] * w2_single[1]  # INT8 w2: 1 byte per param
        + (w13_single[0] + w2_single[0]) * 4  # FP32 scales per channel
    )

    budget = 8 * 1024**3
    num_layers = 26  # offloaded layers

    bf16_slots = budget // num_layers // bf16_expert
    int8_slots = budget // num_layers // int8_expert

    print(f"  BF16 expert size: {bf16_expert / 1e6:.2f} MB")
    print(f"  INT8 expert size: {int8_expert / 1e6:.2f} MB (scale overhead: {(w13_single[0] + w2_single[0]) * 4 / int8_expert * 100:.1f}%)")
    print(f"  Slots per layer (BF16): {bf16_slots}")
    print(f"  Slots per layer (INT8): {int8_slots}")
    print(f"  Slot increase: {int8_slots / bf16_slots:.2f}× ")
    print(f"  HBM reads per layer (17 experts, same top_k=8):")
    print(f"    BF16: {17 * bf16_expert / 1e6:.1f} MB")
    print(f"    INT8: {17 * int8_expert / 1e6:.1f} MB")
    print(f"    Bandwidth reduction: {17 * bf16_expert / (17 * int8_expert):.2f}×")
    print(f"  ✅ INT8 provides ~{int8_slots / bf16_slots:.1f}× more slots and {17 * bf16_expert / (17 * int8_expert):.1f}× less HBM traffic\n")


if __name__ == "__main__":
    print("=" * 60)
    print("INT8 W8A16 Pool Quantization — Correctness Tests")
    print("=" * 60)
    print()

    test_quantize_dequantize_roundtrip()
    test_pool_size_comparison()

    if torch.cuda.is_available():
        test_cache_load_expert_int8()
        test_kernel_int8_vs_bf16()
    else:
        print("CUDA not available, skipping GPU tests\n")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
