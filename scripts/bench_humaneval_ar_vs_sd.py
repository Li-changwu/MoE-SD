#!/usr/bin/env python3
"""
BriskMoE Benchmark: AR vs SD with real HumanEval prompts.

Uses vLLM LLM API to run 10 HumanEval prompts through:
  1. AR baseline (Qwen3-30B-A3B, no speculative decoding)
  2. SD with Eagle3 (K=3)

Measures per-prompt latency, TPOT, throughput, and acceptance rate.
"""

import json
import os
import sys
import time
import gc

# ── Config ──
MODEL = "/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR = "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATA_FILE = "/root/MoE-SD/data/humaneval_bench.jsonl"
RESULT_DIR = "/root/MoE-SD/results/ar_vs_sd_humaneval"
NUM_PROMPTS = 10
MAX_OUTPUT_TOKENS = 256  # code completion needs more tokens
WARMUP_COUNT = 2
GPU_MEM_UTIL = 0.90
CPU_OFFLOAD_GB = 30
MAX_MODEL_LEN = 4096
NUM_SPECULATIVE_TOKENS = 3


def load_prompts(path, n):
    """Load first n prompts from jsonl file."""
    prompts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts


def run_ar_benchmark(prompts, warmup_count):
    """Run AR (autoregressive) benchmark."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("  Loading AR model (no speculative decoding)...")
    print("=" * 70)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM_UTIL,
        cpu_offload_gb=CPU_OFFLOAD_GB,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    # Warmup
    print(f"\n  Warmup: {warmup_count} iterations...")
    for i in range(warmup_count):
        _ = llm.generate([prompts[0]], sampling_params)
        print(f"    warmup {i + 1}/{warmup_count} done")

    # Benchmark: run each prompt individually to measure per-prompt latency
    results = []
    print(f"\n  Benchmark: {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()

        output = outputs[0]
        output_text = output.outputs[0].text
        num_output_tokens = len(output.outputs[0].token_ids)
        num_input_tokens = len(output.prompt_token_ids)
        latency = t1 - t0

        result = {
            "prompt_idx": i,
            "input_tokens": num_input_tokens,
            "output_tokens": num_output_tokens,
            "latency_s": round(latency, 3),
            "tpot_ms": round((latency / num_output_tokens) * 1000, 1) if num_output_tokens > 0 else 0,
            "throughput_tps": round(num_output_tokens / latency, 2) if latency > 0 else 0,
            "output_preview": output_text[:120],
        }
        results.append(result)
        print(f"    [{i + 1}/{len(prompts)}] {num_input_tokens} in → {num_output_tokens} out, "
              f"{latency:.2f}s, {result['throughput_tps']:.2f} tok/s")

    # Cleanup
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def run_sd_benchmark(prompts, warmup_count):
    """Run SD (speculative decoding with Eagle3) benchmark."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("  Loading SD model (Eagle3 K=3)...")
    print("=" * 70)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM_UTIL,
        cpu_offload_gb=CPU_OFFLOAD_GB,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        speculative_config={
            "method": "eagle3",
            "model": SPECULATOR,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    # Warmup
    print(f"\n  Warmup: {warmup_count} iterations...")
    for i in range(warmup_count):
        _ = llm.generate([prompts[0]], sampling_params)
        print(f"    warmup {i + 1}/{warmup_count} done")

    # Benchmark
    results = []
    print(f"\n  Benchmark: {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()

        output = outputs[0]
        output_text = output.outputs[0].text
        num_output_tokens = len(output.outputs[0].token_ids)
        num_input_tokens = len(output.prompt_token_ids)
        latency = t1 - t0

        result = {
            "prompt_idx": i,
            "input_tokens": num_input_tokens,
            "output_tokens": num_output_tokens,
            "latency_s": round(latency, 3),
            "tpot_ms": round((latency / num_output_tokens) * 1000, 1) if num_output_tokens > 0 else 0,
            "throughput_tps": round(num_output_tokens / latency, 2) if latency > 0 else 0,
            "output_preview": output_text[:120],
        }
        results.append(result)
        print(f"    [{i + 1}/{len(prompts)}] {num_input_tokens} in → {num_output_tokens} out, "
              f"{latency:.2f}s, {result['throughput_tps']:.2f} tok/s")

    # Cleanup
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def analyze_and_save(ar_results, sd_results, prompts):
    """Analyze results and save summary."""
    os.makedirs(RESULT_DIR, exist_ok=True)

    def aggregate(results):
        total_out = sum(r["output_tokens"] for r in results)
        total_lat = sum(r["latency_s"] for r in results)
        avg_lat = total_lat / len(results)
        avg_tpot = sum(r["tpot_ms"] for r in results) / len(results)
        avg_tps = sum(r["throughput_tps"] for r in results) / len(results)
        avg_in = sum(r["input_tokens"] for r in results) / len(results)
        avg_out = sum(r["output_tokens"] for r in results) / len(results)
        return {
            "num_prompts": len(results),
            "avg_input_tokens": round(avg_in, 1),
            "avg_output_tokens": round(avg_out, 1),
            "total_output_tokens": total_out,
            "total_latency_s": round(total_lat, 2),
            "avg_latency_s": round(avg_lat, 3),
            "avg_tpot_ms": round(avg_tpot, 1),
            "avg_throughput_tps": round(avg_tps, 2),
        }

    ar_agg = aggregate(ar_results)
    sd_agg = aggregate(sd_results)
    speedup = ar_agg["avg_latency_s"] / sd_agg["avg_latency_s"] if sd_agg["avg_latency_s"] > 0 else 0

    # Print summary
    print("\n" + "=" * 78)
    print("  BriskMoE HumanEval Benchmark: AR vs SD (Qwen3-30B-A3B + EAGLE-3)")
    print("=" * 78)
    print()
    print(f"  数据集:    HumanEval ({NUM_PROMPTS} prompts, real code completion)")
    print(f"  最大输出:  {MAX_OUTPUT_TOKENS} tokens (greedy decoding)")
    print(f"  硬件:      NVIDIA RTX A6000 48GB, PCIe Gen4 x16")
    print(f"  CPU Offload: {CPU_OFFLOAD_GB} GB")
    print()

    print("-" * 78)
    print(f"  {'指标':<35} {'AR Baseline':>18} {'SD (Eagle3 K=3)':>18}")
    print("-" * 78)
    print(f"  {'Avg Input Tokens':<35} {ar_agg['avg_input_tokens']:>18.1f} {sd_agg['avg_input_tokens']:>18.1f}")
    print(f"  {'Avg Output Tokens':<35} {ar_agg['avg_output_tokens']:>18.1f} {sd_agg['avg_output_tokens']:>18.1f}")
    print(f"  {'Avg Latency (s)':<35} {ar_agg['avg_latency_s']:>17.3f}s {sd_agg['avg_latency_s']:>17.3f}s")
    print(f"  {'Avg TPOT (ms/token)':<35} {ar_agg['avg_tpot_ms']:>18.1f} {sd_agg['avg_tpot_ms']:>18.1f}")
    print(f"  {'Avg Throughput (tok/s)':<35} {ar_agg['avg_throughput_tps']:>18.2f} {sd_agg['avg_throughput_tps']:>18.2f}")
    print("-" * 78)
    print(f"  {'Latency Speedup (AR/SD)':<35} {speedup:>17.2f}x")
    print(f"  {'Throughput Ratio (SD/AR)':<35} {sd_agg['avg_throughput_tps'] / ar_agg['avg_throughput_tps']:>17.2f}x" if ar_agg["avg_throughput_tps"] > 0 else "")
    print("-" * 78)

    # Per-prompt comparison
    print("\n  Per-prompt comparison:")
    print(f"  {'#':<4} {'InTok':>6} {'AR OutTok':>10} {'SD OutTok':>10} {'AR Lat(s)':>10} {'SD Lat(s)':>10} {'AR tps':>8} {'SD tps':>8} {'Speedup':>8}")
    print("  " + "-" * 76)
    for i in range(len(ar_results)):
        ar = ar_results[i]
        sd = sd_results[i]
        sp = ar["latency_s"] / sd["latency_s"] if sd["latency_s"] > 0 else 0
        print(f"  {i:<4} {ar['input_tokens']:>6} {ar['output_tokens']:>10} {sd['output_tokens']:>10} "
              f"{ar['latency_s']:>10.2f} {sd['latency_s']:>10.2f} "
              f"{ar['throughput_tps']:>8.2f} {sd['throughput_tps']:>8.2f} {sp:>7.2f}x")

    # Save structured results
    summary = {
        "experiment": "ar_vs_sd_humaneval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "humaneval",
        "num_prompts": NUM_PROMPTS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "hardware": {
            "gpu": "NVIDIA RTX A6000 48GB",
            "pcie": "Gen4 x16",
            "cpu_offload_gb": CPU_OFFLOAD_GB,
        },
        "model": {
            "target": "Qwen3-30B-A3B-Instruct-2507",
            "speculator": f"Eagle3 (K={NUM_SPECULATIVE_TOKENS})",
        },
        "ar_aggregate": ar_agg,
        "sd_aggregate": sd_agg,
        "comparison": {
            "latency_speedup": round(speedup, 3),
            "throughput_ratio_sd_over_ar": round(sd_agg["avg_throughput_tps"] / ar_agg["avg_throughput_tps"], 3) if ar_agg["avg_throughput_tps"] > 0 else 0,
            "sd_is_faster": speedup > 1.0,
        },
        "ar_per_prompt": ar_results,
        "sd_per_prompt": sd_results,
    }

    summary_path = os.path.join(RESULT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {summary_path}")

    # Also save individual results
    with open(os.path.join(RESULT_DIR, "ar_results.json"), "w") as f:
        json.dump(ar_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULT_DIR, "sd_results.json"), "w") as f:
        json.dump(sd_results, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load prompts
    prompts = load_prompts(DATA_FILE, NUM_PROMPTS)
    print(f"Loaded {len(prompts)} HumanEval prompts")
    for i, p in enumerate(prompts):
        # Show first line of each prompt
        first_line = p.strip().split("\n")[0]
        print(f"  [{i}] {first_line[:80]}")

    # Phase 1: AR benchmark
    ar_results = run_ar_benchmark(prompts, WARMUP_COUNT)

    # Save intermediate AR results
    with open(os.path.join(RESULT_DIR, "ar_results.json"), "w") as f:
        json.dump(ar_results, f, indent=2, ensure_ascii=False)
    print("\n  AR results saved. Now loading SD model...")

    # Give GPU time to clean up
    time.sleep(5)

    # Phase 2: SD benchmark
    sd_results = run_sd_benchmark(prompts, WARMUP_COUNT)

    # Phase 3: Analyze
    analyze_and_save(ar_results, sd_results, prompts)


if __name__ == "__main__":
    main()
