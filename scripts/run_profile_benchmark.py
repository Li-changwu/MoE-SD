#!/usr/bin/env python3
"""
Quick profiling run: launches vLLM with ELMM phase profiling enabled,
sends a few requests, and captures timing breakdown from stderr.
"""
import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL = "/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR = "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
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

RESULTS_DIR = "/root/MoE-SD/results/profile_run"


def kill_gpu_processes():
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    kill_gpu_processes()

    env = os.environ.copy()
    env.update({
        "VLLM_PLUGINS": "elmm",
        "ELMM_CACHE_GB": "4",
        "ELMM_LOG_INTERVAL": "0",
        "ELMM_PREFETCH": "0",
        "ELMM_LOCALITY": "0",
        "ELMM_ADAPTIVE_BUDGET": "0",
        "ELMM_POOL_DIRECT": "1",
        "ELMM_DIRECT_DISPATCH": "1",
        "ELMM_PROFILE": "1",
        "PYTHONPATH": "/root/MoE-SD:" + env.get("PYTHONPATH", ""),
    })

    log_path = os.path.join(RESULTS_DIR, "server_profile.log")
    log_file = open(log_path, "w")
    cmd = [CONDA_PYTHON, "-m", "vllm.entrypoints.openai.api_server"] + VLLM_ARGS
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, cwd="/root/MoE-SD")
    print(f"Server PID: {proc.pid}, log: {log_path}")

    # Wait for server
    url = f"http://127.0.0.1:{PORT}/health"
    start = time.time()
    while time.time() - start < 300:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"Server ready ({time.time()-start:.0f}s)")
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        print("Server timeout!")
        proc.terminate()
        return

    # Send 4 requests (profiling window: warmup=200, steps=100 intercepts)
    # With 26 layers → 200/26 ≈ 8 verify rounds warmup, 100/26 ≈ 4 rounds profiled
    # Each request at 128 tokens ≈ 30-40 verify rounds → total ~120+ rounds ≈ 3000+ intercepts
    NUM_REQS = 4
    model_name = MODEL
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to compute Fibonacci numbers efficiently.",
        "What are the main differences between TCP and UDP protocols?",
        "Describe the process of photosynthesis step by step.",
    ]

    for i in range(NUM_REQS):
        print(f"  Request {i+1}/{NUM_REQS}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompts[i]}],
                    "max_tokens": 128,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=300,
            )
            data = resp.json()
            toks = data.get("usage", {}).get("completion_tokens", 0)
            elapsed = time.perf_counter() - t0
            print(f"{toks} tok, {elapsed:.2f}s, {toks/elapsed:.2f} tok/s")
        except Exception as e:
            print(f"ERROR: {e}")

    # Shut down server
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)

    time.sleep(2)

    # Extract profiling report from log
    print(f"\n{'='*60}")
    print("  Phase Profiling Report (from server log)")
    print(f"{'='*60}")
    with open(log_path) as f:
        in_profile = False
        for line in f:
            if "Phase Profiling Report" in line:
                in_profile = True
            if in_profile:
                print(line.rstrip())
            if in_profile and "End Profiling" in line:
                in_profile = False
                break
    if not in_profile:
        print("  (No profiling report found — may need more requests)")
        # Search for ELMM lines
        with open(log_path) as f:
            for line in f:
                if "[ELMM]" in line:
                    print(f"  {line.rstrip()}")

    print(f"\nFull log: {log_path}")
    kill_gpu_processes()


if __name__ == "__main__":
    main()
