#!/usr/bin/env python3
"""
Pilot Experiment: MoE + SD Resource Competition under Memory Pressure
=====================================================================
Hypothesis: As GPU memory pressure increases, SD speedup degrades and eventually reverses.

Design:
  - Fixed: K=3 (EAGLE3 recommended), real text prompts, cpu_offload=30GB
  - Variable: gpu_memory_utilization P ∈ {0.90, 0.80, 0.70, 0.60, 0.50}
  - For each P: run baseline (no SD) and SD (K=3), measure latency + collect metrics
  - Metrics: avg latency, tok/s, acceptance rate, mean acceptance length, KV cache size

Output: results/pilot_pressure_sweep_<timestamp>/
"""

import subprocess, time, json, os, sys, signal, re, csv
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "/home/sage3/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR = "/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
PORT = 8000
URL = f"http://localhost:{PORT}"
CPU_OFFLOAD = 30
MAX_MODEL_LEN = 2048
SWAP_SPACE = 4
K = 3  # EAGLE3 recommended

PRESSURES = [0.90, 0.80, 0.70, 0.60, 0.50]

# 5 diverse real-text prompts for stable measurement
PROMPTS = [
    "Explain the concept of speculative decoding in large language models in 3 sentences.",
    "Write a Python function that calculates the Fibonacci sequence up to n terms.",
    "What are the main differences between TCP and UDP protocols?",
    "Summarize the key ideas of transformer architecture in neural networks.",
    "Describe three advantages and three disadvantages of renewable energy sources.",
]
MAX_TOKENS = 128
NUM_REPEATS = 2  # run prompt set twice for stability

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results/pilot_pressure_sweep_{TIMESTAMP}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def start_server(gpu_mem, use_sd, log_file):
    """Start vllm serve and return (process, log_path)."""
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--cpu-offload-gb", str(CPU_OFFLOAD),
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enforce-eager",
        "--swap-space", str(SWAP_SPACE),
    ]
    if use_sd:
        spec_config = json.dumps({
            "method": "eagle3",
            "model": SPECULATOR,
            "num_speculative_tokens": K,
        })
        cmd += ["--speculative-config", spec_config]

    log_path = LOG_DIR / log_file
    f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc, log_path


def wait_for_server(timeout=180):
    """Wait until server health endpoint responds."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{URL}/health")
            urllib.request.urlopen(req, timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


def stop_server(proc):
    """Gracefully stop server."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # Wait for GPU memory release
    time.sleep(8)


def send_request(prompt):
    """Send a chat completion request and return (elapsed_s, completion_tokens)."""
    import urllib.request
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=300)
    elapsed = time.time() - start
    data = json.loads(resp.read())
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return elapsed, tokens


def parse_log_metrics(log_path):
    """Extract KV cache size, acceptance rate, mean acceptance length from vllm log."""
    text = log_path.read_text(errors="ignore")

    # KV cache
    kv_match = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    kv_tokens = int(kv_match.group(1).replace(",", "")) if kv_match else None

    kv_gib_match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
    kv_gib = float(kv_gib_match.group(1)) if kv_gib_match else None

    # SpecDecoding metrics (take averages of all reported intervals)
    acc_rates = re.findall(r"Avg Draft acceptance rate:\s*([\d.]+)%", text)
    mean_acc_lens = re.findall(r"Mean acceptance length:\s*([\d.]+)", text)

    avg_acc_rate = None
    avg_mean_acc_len = None
    if acc_rates:
        vals = [float(x) for x in acc_rates]
        avg_acc_rate = sum(vals) / len(vals)
    if mean_acc_lens:
        vals = [float(x) for x in mean_acc_lens]
        avg_mean_acc_len = sum(vals) / len(vals)

    return {
        "kv_cache_tokens": kv_tokens,
        "kv_cache_gib": kv_gib,
        "avg_acceptance_rate": avg_acc_rate,
        "avg_mean_acceptance_length": avg_mean_acc_len,
    }


def run_experiment(gpu_mem, use_sd):
    """Run one experiment: start server, send prompts, collect metrics, stop server."""
    mode = "sd" if use_sd else "baseline"
    label = f"P{gpu_mem:.2f}_{mode}"
    log(f"{'='*60}")
    log(f"  {label}  (gpu_mem={gpu_mem}, SD={'ON K='+str(K) if use_sd else 'OFF'})")
    log(f"{'='*60}")

    log_file = f"pilot_{label}.log"
    proc, log_path = start_server(gpu_mem, use_sd, log_file)

    log("Waiting for server...")
    if not wait_for_server(timeout=180):
        log(f"ERROR: Server failed to start for {label}")
        stop_server(proc)
        return None

    log("Server ready. Sending warmup request...")
    try:
        send_request("Hello, how are you?")
    except Exception as e:
        log(f"Warmup failed: {e}")

    log(f"Running {len(PROMPTS)} prompts x {NUM_REPEATS} repeats...")
    results = []
    total_tokens = 0
    for rep in range(NUM_REPEATS):
        for i, prompt in enumerate(PROMPTS):
            try:
                elapsed, tokens = send_request(prompt)
                tps = tokens / elapsed if elapsed > 0 else 0
                results.append({"elapsed": elapsed, "tokens": tokens, "tps": tps})
                total_tokens += tokens
                log(f"  [{rep+1}.{i+1}] {tokens:3d} tok in {elapsed:.1f}s ({tps:.2f} tok/s)")
            except Exception as e:
                log(f"  [{rep+1}.{i+1}] FAILED: {e}")
                results.append({"elapsed": 999, "tokens": 0, "tps": 0})

    # Wait a moment for metrics to flush
    time.sleep(5)
    stop_server(proc)

    if not results:
        return None

    # Compute aggregates
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_lat = sum(r["elapsed"] for r in results) / len(results)

    # Parse log
    metrics = parse_log_metrics(log_path)

    return {
        "label": label,
        "gpu_mem": gpu_mem,
        "mode": mode,
        "use_sd": use_sd,
        "avg_latency_s": round(avg_lat, 2),
        "avg_tps": round(avg_tps, 2),
        "total_tokens": total_tokens,
        "num_requests": len(results),
        **metrics,
    }


def main():
    log(f"Pilot Pressure Sweep — {TIMESTAMP}")
    log(f"Pressures: {PRESSURES}")
    log(f"K={K}, Prompts={len(PROMPTS)}, Repeats={NUM_REPEATS}")
    log(f"Results: {RESULTS_DIR}")

    all_results = []
    total = len(PRESSURES) * 2
    idx = 0

    for p in PRESSURES:
        for use_sd in [False, True]:
            idx += 1
            log(f"\n>>> Experiment {idx}/{total}")
            result = run_experiment(p, use_sd)
            if result:
                all_results.append(result)
                log(f"  => {result['avg_tps']} tok/s, lat={result['avg_latency_s']}s"
                    + (f", acc_rate={result.get('avg_acceptance_rate','?')}%" if use_sd else ""))
            else:
                log(f"  => SKIPPED (server failed)")

    # Save raw results
    results_json = RESULTS_DIR / "results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV
    csv_path = RESULTS_DIR / "results.csv"
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)

    # Compute speedup table
    log(f"\n{'='*70}")
    log(f"  RESULTS SUMMARY")
    log(f"{'='*70}")
    log(f"{'P':>6} {'Base tok/s':>12} {'SD tok/s':>12} {'Speedup':>10} {'AccRate':>10} {'KV(GiB)':>10}")
    log(f"{'-'*70}")

    speedups = []
    for p in PRESSURES:
        base = next((r for r in all_results if r["gpu_mem"] == p and not r["use_sd"]), None)
        sd = next((r for r in all_results if r["gpu_mem"] == p and r["use_sd"]), None)
        if base and sd:
            speedup = sd["avg_tps"] / base["avg_tps"] if base["avg_tps"] > 0 else 0
            speedups.append({
                "P": p,
                "base_tps": base["avg_tps"],
                "sd_tps": sd["avg_tps"],
                "speedup": round(speedup, 3),
                "base_lat": base["avg_latency_s"],
                "sd_lat": sd["avg_latency_s"],
                "acc_rate": sd.get("avg_acceptance_rate"),
                "mean_acc_len": sd.get("avg_mean_acceptance_length"),
                "kv_gib_base": base.get("kv_cache_gib"),
                "kv_gib_sd": sd.get("kv_cache_gib"),
            })
            ar = f"{sd.get('avg_acceptance_rate', '?'):.1f}%" if sd.get('avg_acceptance_rate') else "?"
            kv = f"{sd.get('kv_cache_gib', '?'):.2f}" if sd.get('kv_cache_gib') else "?"
            log(f"{p:>6.2f} {base['avg_tps']:>12.2f} {sd['avg_tps']:>12.2f} {speedup:>10.3f}x {ar:>10} {kv:>10}")

    # Save speedup table
    speedup_path = RESULTS_DIR / "speedup.json"
    with open(speedup_path, "w") as f:
        json.dump(speedups, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log("Done!")


if __name__ == "__main__":
    main()
