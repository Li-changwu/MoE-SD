#!/usr/bin/env python3
"""BriskMoE HumanEval Benchmark Runner.

Feeds real HumanEval prompts through vLLM's LLM.generate() API to benchmark
latency with realistic expert routing patterns.

Unlike `vllm bench latency` (random token IDs → uniform expert routing),
real code prompts exhibit strong expert locality that BriskMoE can exploit.
"""

import argparse
import json
import os
import time

import numpy as np


def load_prompts(dataset_path: str, num_prompts: int) -> list[str]:
    """Load prompts from JSONL file. Each line: {"prompt": "...", ...}."""
    prompts = []
    with open(dataset_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            prompts.append(obj["prompt"])
            if len(prompts) >= num_prompts:
                break
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM latency with real HumanEval prompts"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to HumanEval JSONL file")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=50,
                        help="Number of prompts to benchmark (after warmup)")
    parser.add_argument("--warmup-prompts", type=int, default=5,
                        help="Number of warmup prompts (not timed)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--cpu-offload-gb", type=float, default=30)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--speculative-config", type=str, default=None,
                        help="JSON string for speculative decoding config")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    # Lazy import
    from vllm import LLM, SamplingParams

    total_needed = args.warmup_prompts + args.num_prompts
    raw_prompts = load_prompts(args.dataset, total_needed)
    if len(raw_prompts) < total_needed:
        # Cycle prompts if dataset is smaller than requested
        import itertools
        raw_prompts = list(itertools.islice(
            itertools.cycle(raw_prompts), total_needed
        ))

    warmup_prompts = raw_prompts[: args.warmup_prompts]
    bench_prompts = raw_prompts[args.warmup_prompts : args.warmup_prompts + args.num_prompts]

    # Build LLM kwargs
    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    if args.speculative_config:
        spec_cfg = json.loads(args.speculative_config)
        llm_kwargs["speculative_config"] = spec_cfg

    print(f"Loading model: {args.model}")
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    # Warmup
    print(f"Warming up with {len(warmup_prompts)} prompts...")
    for i, prompt in enumerate(warmup_prompts):
        llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        print(f"  warmup {i + 1}/{len(warmup_prompts)}")

    # Benchmark - one prompt at a time (batch=1, sequential, CCF-A consensus)
    print(f"\nBenchmarking {len(bench_prompts)} prompts (batch=1, sequential)...")
    latencies = []
    output_lens = []
    for i, prompt in enumerate(bench_prompts):
        start = time.perf_counter()
        outputs = llm.generate(
            [prompt], sampling_params=sampling_params, use_tqdm=False
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        n_out = len(outputs[0].outputs[0].token_ids)
        output_lens.append(n_out)
        tps = n_out / elapsed if elapsed > 0 else 0
        print(f"  [{i + 1}/{len(bench_prompts)}] {elapsed:.3f}s, "
              f"{n_out} tokens, {tps:.2f} tok/s")

    latencies = np.array(latencies)
    output_lens_arr = np.array(output_lens)
    tpot = latencies / np.maximum(output_lens_arr, 1) * 1000  # ms
    tps = output_lens_arr / latencies

    # Results
    percentages = [10, 25, 50, 75, 90, 99]
    lat_pcts = np.percentile(latencies, percentages)
    tpot_pcts = np.percentile(tpot, percentages)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Prompts benchmarked:  {len(bench_prompts)}")
    print(f"  Output len (mean):    {np.mean(output_lens_arr):.1f}")
    print(f"  Latency  mean:        {np.mean(latencies):.3f}s")
    print(f"  Latency  median:      {np.median(latencies):.3f}s")
    print(f"  Latency  p99:         {lat_pcts[-1]:.3f}s")
    print(f"  TPOT     mean:        {np.mean(tpot):.2f} ms")
    print(f"  TPOT     median:      {np.median(tpot):.2f} ms")
    print(f"  TPS      mean:        {np.mean(tps):.2f} tok/s")
    print(f"  TPS      median:      {np.median(tps):.2f} tok/s")

    for pct, lat_v, tpot_v in zip(percentages, lat_pcts, tpot_pcts):
        print(f"  p{pct:<3d}  latency={lat_v:.3f}s  TPOT={tpot_v:.2f}ms")

    results = {
        "num_prompts": len(bench_prompts),
        "output_len_config": args.output_len,
        "output_len_mean": float(np.mean(output_lens_arr)),
        "avg_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "latencies": latencies.tolist(),
        "output_lens": output_lens_arr.tolist(),
        "tpot_mean_ms": float(np.mean(tpot)),
        "tpot_median_ms": float(np.median(tpot)),
        "tps_mean": float(np.mean(tps)),
        "tps_median": float(np.median(tps)),
        "percentiles": {
            str(p): {"latency": float(l), "tpot_ms": float(t)}
            for p, l, t in zip(percentages, lat_pcts, tpot_pcts)
        },
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "output_len": args.output_len,
            "num_prompts": args.num_prompts,
            "warmup_prompts": args.warmup_prompts,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cpu_offload_gb": args.cpu_offload_gb,
            "speculative_config": args.speculative_config,
        },
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
