#!/usr/bin/env python3
"""
ELMM v1 vs v2 Performance A/B Benchmark
==========================================
Compares ELMM baseline (new features disabled) against ELMM v2
(adaptive budget + prefetch + locality collection).

Runs each configuration:
  1. Launch vLLM server with ELMM plugin
  2. Wait for server ready
  3. Warmup + send benchmark requests
  4. Kill server, collect stats
  5. Compare results
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

# ============================================================================
# Configuration
# ============================================================================
MODEL = "/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR = "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
RESULTS_DIR = "/root/MoE-SD/results/v2_perf_comparison"
PORT = 8000
CONDA_PYTHON = "/opt/miniconda3/envs/moe-sd/bin/python"

SPEC_CONFIG = json.dumps({
    "model": SPECULATOR,
    "method": "eagle3",
    "num_speculative_tokens": 3,
    "draft_tensor_parallel_size": 1,
})

VLLM_ARGS = [
    "--model", MODEL,
    "--speculative-config", SPEC_CONFIG,
    "--tensor-parallel-size", "1",
    "--cpu-offload-gb", "30",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--enforce-eager",
    "--port", str(PORT),
]

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute Fibonacci numbers efficiently.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "How does a neural network learn from training data?",
    "Compare and contrast democracy and authoritarianism.",
    "Explain how a CPU cache hierarchy works and why it matters.",
]

NUM_REQUESTS = 5
MAX_TOKENS = 128
WARMUP_REQUESTS = 2

CONFIGS = {
    "v1_baseline": {
        "label": "Warmup=32 only (fixed interval=16)",
        "env": {
            "VLLM_PLUGINS": "elmm",
            "ELMM_CACHE_GB": "4",
            "ELMM_LOG_INTERVAL": "0",
            "ELMM_PREFETCH": "0",
            "ELMM_LOCALITY": "0",
            "ELMM_ADAPTIVE_BUDGET": "0",
            "ELMM_POOL_DIRECT": "1",
            "ELMM_DIRECT_DISPATCH": "1",
            "ELMM_GPU_CACHE": "0",
            "ELMM_STALE_REMAP": "16",
            "ELMM_STALE_REMAP_WARMUP": "32",
            "ELMM_STALE_REMAP_MAX_INTERVAL": "16",
        },
    },
    "v2_features": {
        "label": "TASER full (warmup=32, adaptive 16→128)",
        "env": {
            "VLLM_PLUGINS": "elmm",
            "ELMM_CACHE_GB": "4",
            "ELMM_LOG_INTERVAL": "0",
            "ELMM_PREFETCH": "0",
            "ELMM_LOCALITY": "0",
            "ELMM_ADAPTIVE_BUDGET": "0",
            "ELMM_POOL_DIRECT": "1",
            "ELMM_DIRECT_DISPATCH": "1",
            "ELMM_GPU_CACHE": "0",
            "ELMM_STALE_REMAP": "16",
            "ELMM_STALE_REMAP_WARMUP": "32",
            "ELMM_STALE_REMAP_MAX_INTERVAL": "128",
        },
    },
}


# ============================================================================
# Utilities
# ============================================================================

def kill_gpu_processes():
    """Kill any processes using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"  Killed GPU process {pid}")
            except (ValueError, ProcessLookupError):
                pass
        if pids:
            time.sleep(5)
    except Exception as e:
        print(f"  Warning: {e}")


def launch_server(config_key: str) -> subprocess.Popen:
    """Launch vLLM server with given ELMM config."""
    cfg = CONFIGS[config_key]
    env = os.environ.copy()
    env.update(cfg["env"])
    # Ensure PYTHONPATH includes MoE-SD
    env["PYTHONPATH"] = "/root/MoE-SD:" + env.get("PYTHONPATH", "")

    log_path = os.path.join(RESULTS_DIR, f"server_{config_key}.log")
    log_file = open(log_path, "w")

    cmd = [CONDA_PYTHON, "-m", "vllm.entrypoints.openai.api_server"] + VLLM_ARGS
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd="/root/MoE-SD",
    )
    print(f"  Server PID: {proc.pid}, log: {log_path}")
    return proc


def wait_for_server(timeout=600):
    """Wait for server to be ready."""
    url = f"http://127.0.0.1:{PORT}/health"
    start = time.time()
    last_print = start
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"  Server ready! ({elapsed:.0f}s)")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass

        now = time.time()
        if now - start > timeout:
            print(f"  ERROR: Server not ready after {timeout}s")
            return False
        if now - last_print > 30:
            print(f"  Still waiting... ({now - start:.0f}s)")
            last_print = now
        time.sleep(3)


def kill_server(proc: subprocess.Popen):
    """Stop server process."""
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    # Make sure GPU is freed
    time.sleep(3)
    kill_gpu_processes()


def run_benchmark(config_key: str) -> dict:
    """Send benchmark requests and measure throughput."""
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    model_name = MODEL

    # Warmup
    print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
    for i in range(WARMUP_REQUESTS):
        try:
            requests.post(url, json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 30,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=180)
        except Exception as e:
            print(f"    Warmup {i}: {e}")

    # Benchmark requests
    results = []
    total_tokens = 0
    total_time = 0.0

    for i in range(NUM_REQUESTS):
        prompt = PROMPTS[i % len(PROMPTS)]
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
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
            print(f"    [{i+1}/{NUM_REQUESTS}] {completion_tokens} tok, "
                  f"{elapsed:.2f}s, {tps:.2f} tok/s")
        except Exception as e:
            print(f"    [{i+1}/{NUM_REQUESTS}] ERROR: {e}")
            results.append({"prompt_idx": i, "error": str(e)})

    # Save per-request data
    output_path = os.path.join(RESULTS_DIR, f"{config_key}_requests.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute summary
    successful = [r for r in results if "error" not in r]
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    per_req_tps = [r["tps"] for r in successful]

    summary = {
        "config": config_key,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 3),
        "avg_tps": round(avg_tps, 3),
        "per_request_tps": per_req_tps,
        "min_tps": round(min(per_req_tps), 3) if per_req_tps else 0,
        "max_tps": round(max(per_req_tps), 3) if per_req_tps else 0,
        "num_requests": len(successful),
        "errors": len(results) - len(successful),
    }

    summary_path = os.path.join(RESULTS_DIR, f"{config_key}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  => avg {avg_tps:.2f} tok/s ({total_tokens} tokens in {total_time:.1f}s)")
    return summary


def run_config(config_key: str) -> dict:
    """Full lifecycle: launch server, benchmark, kill."""
    cfg = CONFIGS[config_key]
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")

    kill_gpu_processes()
    proc = launch_server(config_key)

    try:
        if not wait_for_server(timeout=600):
            # Print last 30 lines of server log
            log_path = os.path.join(RESULTS_DIR, f"server_{config_key}.log")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    lines = f.readlines()
                print("  Last 30 lines of server log:")
                for line in lines[-30:]:
                    print(f"    {line.rstrip()}")
            return {"config": config_key, "error": "server_timeout"}

        return run_benchmark(config_key)
    finally:
        kill_server(proc)


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "v2_locality"), exist_ok=True)

    print("=" * 60)
    print("  ELMM v1 vs v2 Performance A/B Benchmark")
    print("=" * 60)
    print(f"  Model:     {MODEL}")
    print(f"  Requests:  {NUM_REQUESTS} × {MAX_TOKENS} max tokens")
    print(f"  Warmup:    {WARMUP_REQUESTS} requests")
    print(f"  Results:   {RESULTS_DIR}/")

    summaries = {}

    # Run each config
    for config_key in CONFIGS:
        try:
            summaries[config_key] = run_config(config_key)
        except Exception as e:
            print(f"  FATAL: {e}")
            import traceback
            traceback.print_exc()
            summaries[config_key] = {"config": config_key, "error": str(e)}

    # ========================================================================
    # Comparison
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"  Performance Comparison")
    print(f"{'='*60}\n")

    header = f"{'Config':<42} {'avg tok/s':>10} {'min':>6} {'max':>6} {'reqs':>5}"
    print(header)
    print("-" * len(header))

    for key, cfg in CONFIGS.items():
        s = summaries.get(key, {})
        if "error" in s:
            print(f"{cfg['label']:<42} {'ERROR':>10}")
        else:
            print(f"{cfg['label']:<42} {s.get('avg_tps', 0):>10.2f} "
                  f"{s.get('min_tps', 0):>6.2f} {s.get('max_tps', 0):>6.2f} "
                  f"{s.get('num_requests', 0):>5}")

    v1 = summaries.get("v1_baseline", {})
    v2 = summaries.get("v2_features", {})

    if "error" not in v1 and "error" not in v2:
        v1_tps = v1["avg_tps"]
        v2_tps = v2["avg_tps"]
        if v1_tps > 0:
            ratio = v2_tps / v1_tps
            delta_pct = (v2_tps - v1_tps) / v1_tps * 100
            print(f"\n  v2/v1 = {ratio:.3f}x ({delta_pct:+.1f}%)")
            if ratio > 1.02:
                print(f"  => v2 is FASTER by {delta_pct:.1f}%")
            elif ratio < 0.98:
                print(f"  => v2 has {abs(delta_pct):.1f}% overhead")
            else:
                print(f"  => Performance equivalent (within noise)")

    # Save combined comparison
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": {k: summaries.get(k, {}) for k in CONFIGS},
    }
    if "error" not in v1 and "error" not in v2 and v1.get("avg_tps", 0) > 0:
        comparison["speedup_ratio"] = round(v2["avg_tps"] / v1["avg_tps"], 4)

    comp_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Full comparison saved to: {comp_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
