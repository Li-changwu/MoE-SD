#!/usr/bin/env python3
"""
ShareGPT Serving Benchmark Client
===================================
Sends real prompts from a ShareGPT dataset to a vLLM OpenAI-compatible server
and measures throughput / latency metrics.

Unlike `vllm bench serve --dataset-name random`, this uses natural language
prompts so speculative decoding draft models can predict effectively.

Output format is compatible with vllm bench serve results for easy comparison.

Usage:
  python scripts/bench_serving_sharegpt.py \
    --base-url http://127.0.0.1:8192 \
    --dataset /root/MoE-SD/data/combined_sharegpt.json \
    --num-prompts 50 --max-tokens 128 --request-rate 1.0 \
    --result-dir ./results
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


def load_sharegpt_prompts(dataset_path: str, num_prompts: int, seed: int = 42):
    """Load prompts from ShareGPT format dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(data)

    prompts = []
    for item in data:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        human_msg = convs[0].get("value", "").strip()
        if len(human_msg) < 10:
            continue
        prompts.append(human_msg)
        if len(prompts) >= num_prompts:
            break

    return prompts


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
):
    """Send a single completion request and measure timing."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    t_start = time.perf_counter()
    ttft = None
    generated_text = ""
    output_tokens = 0
    status_code = 0

    try:
        async with session.post(url, json=payload) as resp:
            status_code = resp.status
            ttft = time.perf_counter() - t_start
            body = await resp.json()

            if resp.status == 200:
                choices = body.get("choices", [])
                if choices:
                    generated_text = choices[0].get("text", "")
                usage = body.get("usage", {})
                output_tokens = usage.get("completion_tokens", 0)
                if output_tokens == 0 and generated_text:
                    # Approximate if usage not reported
                    output_tokens = max(1, len(generated_text.split()))
            else:
                print(f"  [WARN] req {request_id}: status={resp.status}")

    except Exception as e:
        print(f"  [ERR] req {request_id}: {e}")
        return None

    t_end = time.perf_counter()
    e2e_latency = t_end - t_start

    return {
        "request_id": request_id,
        "status": status_code,
        "ttft_s": ttft,
        "e2e_latency_s": e2e_latency,
        "output_tokens": output_tokens,
        "prompt_len_chars": len(prompt),
    }


async def run_benchmark(args):
    """Run the full serving benchmark."""
    prompts = load_sharegpt_prompts(args.dataset, args.num_prompts, args.seed)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")
    print(f"Request rate: {args.request_rate} QPS")
    print(f"Max tokens: {args.max_tokens}")

    # Generate Poisson arrival times
    rng = random.Random(args.seed)
    if args.request_rate == float("inf"):
        intervals = [0.0] * len(prompts)
    else:
        intervals = [rng.expovariate(args.request_rate) for _ in prompts]

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        results = []
        tasks = []

        async def schedule_request(idx, prompt, delay):
            if delay > 0:
                await asyncio.sleep(delay)
            r = await send_request(
                session, args.base_url, args.model,
                prompt, args.max_tokens, idx,
            )
            if r:
                results.append(r)

        t_bench_start = time.perf_counter()

        for i, (prompt, interval) in enumerate(zip(prompts, intervals)):
            cumulative_delay = sum(intervals[: i + 1])
            tasks.append(asyncio.create_task(
                schedule_request(i, prompt, cumulative_delay)
            ))

        await asyncio.gather(*tasks)
        t_bench_end = time.perf_counter()

    bench_duration = t_bench_end - t_bench_start
    return results, bench_duration, len(prompts)


def compute_metrics(results, bench_duration, num_prompts):
    """Compute aggregate metrics from individual request results."""
    successful = [r for r in results if r["status"] == 200]
    if not successful:
        return {"status": "no_successful_requests"}

    ttfts = [r["ttft_s"] * 1000 for r in successful]  # ms
    e2e_lats = [r["e2e_latency_s"] * 1000 for r in successful]  # ms
    output_tokens = [r["output_tokens"] for r in successful]

    total_output_tokens = sum(output_tokens)

    # TPOT: (e2e - ttft) / (output_tokens - 1) for each request
    tpots = []
    itls = []
    for r in successful:
        n = r["output_tokens"]
        if n > 1:
            tpot = (r["e2e_latency_s"] - r["ttft_s"]) / (n - 1) * 1000  # ms
            tpots.append(tpot)
            itls.append(tpot)  # non-streaming: ITL ≈ TPOT

    def pct(vals, p):
        s = sorted(vals)
        idx = int(len(s) * p / 100)
        idx = min(idx, len(s) - 1)
        return s[idx]

    metrics = {
        "successful_requests": len(successful),
        "total_requests": num_prompts,
        "duration_s": bench_duration,
        "total_generated_tokens": total_output_tokens,
        "output_throughput_tok_s": total_output_tokens / bench_duration if bench_duration > 0 else 0,
        "request_throughput_req_s": len(successful) / bench_duration if bench_duration > 0 else 0,
        # TTFT
        "mean_ttft_ms": statistics.mean(ttfts),
        "median_ttft_ms": statistics.median(ttfts),
        "p95_ttft_ms": pct(ttfts, 95),
        "p99_ttft_ms": pct(ttfts, 99),
        # E2EL
        "mean_e2el_ms": statistics.mean(e2e_lats),
        "median_e2el_ms": statistics.median(e2e_lats),
        "p95_e2el_ms": pct(e2e_lats, 95),
        "p99_e2el_ms": pct(e2e_lats, 99),
    }

    if tpots:
        metrics.update({
            "mean_tpot_ms": statistics.mean(tpots),
            "median_tpot_ms": statistics.median(tpots),
            "p95_tpot_ms": pct(tpots, 95),
            "p99_tpot_ms": pct(tpots, 99),
            "mean_itl_ms": statistics.mean(itls),
        })

    return metrics


def print_results(metrics):
    """Print results in vllm bench serve compatible format."""
    print("\n============ Serving Benchmark Result ============")
    print(f"Successful requests:                     {metrics.get('successful_requests', 0)}")
    print(f"Benchmark duration (s):                  {metrics.get('duration_s', 0):.2f}")
    print(f"Total generated tokens:                  {metrics.get('total_generated_tokens', 0)}")
    print(f"Request throughput (req/s):              {metrics.get('request_throughput_req_s', 0):.2f}")
    print(f"Output token throughput (tok/s):         {metrics.get('output_throughput_tok_s', 0):.2f}")
    print("---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                          {metrics.get('mean_ttft_ms', 0):.2f}")
    print(f"Median TTFT (ms):                        {metrics.get('median_ttft_ms', 0):.2f}")
    print(f"P99 TTFT (ms):                           {metrics.get('p99_ttft_ms', 0):.2f}")
    print("-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                          {metrics.get('mean_tpot_ms', 0):.2f}")
    print(f"Median TPOT (ms):                        {metrics.get('median_tpot_ms', 0):.2f}")
    print(f"P99 TPOT (ms):                           {metrics.get('p99_tpot_ms', 0):.2f}")
    print("---------------Inter-token Latency----------------")
    print(f"Mean ITL (ms):                           {metrics.get('mean_itl_ms', 0):.2f}")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="ShareGPT Serving Benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", default=None)
    args = parser.parse_args()

    results, duration, num = asyncio.run(run_benchmark(args))
    metrics = compute_metrics(results, duration, num)

    print_results(metrics)

    if args.result_dir:
        out_dir = Path(args.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "serve_metrics.json").write_text(json.dumps(metrics, indent=2))
        (out_dir / "serve_requests.json").write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
