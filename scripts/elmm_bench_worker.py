#!/usr/bin/env python3
"""
Benchmark worker for ELMM ablation experiments.
Sends sequential chat completion requests and measures per-request latency/TPS.
"""

import argparse
import json
import time

import requests


PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute Fibonacci numbers efficiently.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "How does a neural network learn from training data?",
    "Compare and contrast democracy and authoritarianism.",
    "Explain how a CPU cache hierarchy works and why it matters.",
    "What is the significance of the Turing test in AI?",
    "Describe the major causes and effects of climate change.",
    "How do hash tables work and what is their time complexity?",
    "Explain the CAP theorem in distributed systems.",
    "What are the key principles of object-oriented programming?",
    "Describe how HTTPS ensures secure communication.",
    "What is quantum entanglement and why is it important?",
    "How does garbage collection work in modern programming languages?",
    "Explain the concept of recursion with a practical example.",
    "What are the advantages of microservices over monolithic architecture?",
    "How does CRISPR gene editing technology work?",
    "Describe the working principle of a transformer neural network.",
    "What are the ethical considerations of artificial intelligence?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    url = f"{args.base_url}/v1/chat/completions"

    # Warmup
    for i in range(args.warmup):
        try:
            requests.post(url, json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 30,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=120)
        except Exception:
            pass
    print(f"  Warmup done ({args.warmup} requests)")

    # Benchmark
    results = []
    total_tokens = 0
    total_time = 0.0

    for i in range(args.num_prompts):
        prompt = PROMPTS[i % len(PROMPTS)]
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.perf_counter() - t0
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tps = completion_tokens / elapsed if elapsed > 0 else 0
            results.append({
                "prompt_idx": i,
                "completion_tokens": completion_tokens,
                "elapsed_s": round(elapsed, 3),
                "tps": round(tps, 3),
            })
            total_tokens += completion_tokens
            total_time += elapsed
            print(f"    [{i+1}/{args.num_prompts}] {completion_tokens} tok, "
                  f"{elapsed:.2f}s, {tps:.1f} tok/s")
        except Exception as e:
            print(f"    [{i+1}/{args.num_prompts}] ERROR: {e}")
            results.append({"prompt_idx": i, "error": str(e)})

    # Write per-request results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Write summary
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    summary = {
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 3),
        "avg_tps": round(avg_tps, 3),
        "num_requests": len([r for r in results if "error" not in r]),
        "errors": len([r for r in results if "error" in r]),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: {summary['avg_tps']} tok/s, "
          f"{summary['total_tokens']} tokens in {summary['total_time_s']}s")


if __name__ == "__main__":
    main()
