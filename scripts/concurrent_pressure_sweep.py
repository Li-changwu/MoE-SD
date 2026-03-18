#!/usr/bin/env python3
"""
Concurrent Pressure Sweep: Prove MoE + SD KV Cache Competition
===============================================================
Design:
  - Fix: P=0.90, cpu_offload=30, K=3 (EAGLE3)
  - Variable: num_gpu_blocks ∈ [8000, 2000, 1000, 600, 400]
    → Simulates different memory pressure levels by limiting KV cache capacity
  - Load: 50 real-text prompts from alpaca+dolly dataset, sent CONCURRENTLY
  - For each block level: baseline (no SD) vs SD (K=3)
  - Metrics: total throughput, per-request latency, acceptance rate, KV cache usage
"""

import subprocess, time, json, os, sys, signal, re, csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────
MODEL = "/home/sage3/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR = "/home/sage3/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
PORT = 8000
URL = f"http://localhost:{PORT}"
CPU_OFFLOAD = 30
MAX_MODEL_LEN = 2048
K = 3

# Block levels: high (no pressure) → low (extreme pressure)
BLOCK_LEVELS = [8000, 1500, 1000, 800, 600]

# Load prompts from dataset
DATASET_PATH = "data/combined_sharegpt.json"
NUM_PROMPTS = 50
MAX_TOKENS = 256      # longer output → more KV cache per request
CONCURRENCY = 50      # send all prompts at once

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results/concurrent_sweep_{TIMESTAMP}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_prompts():
    """Load prompts from the ShareGPT-format dataset."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    prompts = [d["conversations"][0]["value"] for d in data[:NUM_PROMPTS]]
    log(f"Loaded {len(prompts)} prompts from {DATASET_PATH}")
    return prompts


def start_server(num_blocks, use_sd, log_file):
    """Start vllm serve with specified block limit."""
    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(PORT),
        "--cpu-offload-gb", str(CPU_OFFLOAD),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enforce-eager",
        "--swap-space", "4",
        "--num-gpu-blocks-override", str(num_blocks),
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


def wait_for_server(timeout=300):
    """Wait until server health endpoint responds."""
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
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(8)


def send_single_request(prompt, idx):
    """Send one chat completion request. Returns dict with timing info."""
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
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        elapsed = time.time() - start
        data = json.loads(resp.read())
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {
            "idx": idx,
            "elapsed": elapsed,
            "tokens": tokens,
            "tps": tokens / elapsed if elapsed > 0 else 0,
            "status": "ok",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "idx": idx,
            "elapsed": elapsed,
            "tokens": 0,
            "tps": 0,
            "status": f"error: {e}",
        }


def send_concurrent_requests(prompts):
    """Send all prompts concurrently and return results."""
    results = []
    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(send_single_request, p, i): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            results.append(future.result())

    wall_elapsed = time.time() - wall_start
    results.sort(key=lambda x: x["idx"])
    return results, wall_elapsed


def parse_log_metrics(log_path):
    """Extract metrics from vllm server log."""
    text = log_path.read_text(errors="ignore")

    # KV cache
    kv_match = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    kv_tokens = int(kv_match.group(1).replace(",", "")) if kv_match else None

    kv_gib_match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
    kv_gib = float(kv_gib_match.group(1)) if kv_gib_match else None

    # SpecDecoding metrics
    acc_rates = re.findall(r"Avg Draft acceptance rate:\s*([\d.]+)%", text)
    mean_acc_lens = re.findall(r"Mean acceptance length:\s*([\d.]+)", text)

    # KV cache usage peaks
    kv_usages = re.findall(r"GPU KV cache usage:\s*([\d.]+)%", text)

    avg_acc_rate = None
    avg_mean_acc_len = None
    peak_kv_usage = None

    if acc_rates:
        vals = [float(x) for x in acc_rates]
        avg_acc_rate = sum(vals) / len(vals)
    if mean_acc_lens:
        vals = [float(x) for x in mean_acc_lens]
        avg_mean_acc_len = sum(vals) / len(vals)
    if kv_usages:
        peak_kv_usage = max(float(x) for x in kv_usages)

    return {
        "kv_cache_tokens": kv_tokens,
        "kv_cache_gib": kv_gib,
        "avg_acceptance_rate": avg_acc_rate,
        "avg_mean_acceptance_length": avg_mean_acc_len,
        "peak_kv_usage_pct": peak_kv_usage,
    }


def run_experiment(num_blocks, use_sd, prompts):
    """Run one experiment: start server, send concurrent prompts, collect metrics."""
    mode = "sd" if use_sd else "baseline"
    label = f"B{num_blocks}_{mode}"
    log(f"\n{'='*60}")
    log(f"  {label}  (blocks={num_blocks}, SD={'ON K='+str(K) if use_sd else 'OFF'})")
    log(f"{'='*60}")

    log_file = f"csweep_{label}.log"
    proc, log_path = start_server(num_blocks, use_sd, log_file)

    log("Waiting for server...")
    if not wait_for_server(timeout=300):
        log(f"ERROR: Server failed to start for {label}")
        stop_server(proc)
        return None

    log("Server ready. Sending warmup request...")
    try:
        send_single_request("Hello!", 0)
    except Exception:
        pass
    time.sleep(2)

    log(f"Sending {len(prompts)} requests concurrently...")
    results, wall_time = send_concurrent_requests(prompts)

    # Brief pause for metrics to flush
    time.sleep(5)
    stop_server(proc)

    ok_results = [r for r in results if r["status"] == "ok"]
    fail_count = len(results) - len(ok_results)

    if not ok_results:
        log(f"  All requests failed!")
        return None

    total_tokens = sum(r["tokens"] for r in ok_results)
    avg_tps_per_req = sum(r["tps"] for r in ok_results) / len(ok_results)
    avg_latency = sum(r["elapsed"] for r in ok_results) / len(ok_results)
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    metrics = parse_log_metrics(log_path)

    log(f"  Wall time: {wall_time:.1f}s, Throughput: {throughput:.1f} tok/s")
    log(f"  Avg per-request: {avg_tps_per_req:.2f} tok/s, {avg_latency:.1f}s latency")
    log(f"  OK: {len(ok_results)}/{len(results)}, Total tokens: {total_tokens}")
    if metrics.get("peak_kv_usage_pct"):
        log(f"  Peak KV usage: {metrics['peak_kv_usage_pct']:.1f}%")
    if metrics.get("avg_acceptance_rate"):
        log(f"  Avg acceptance rate: {metrics['avg_acceptance_rate']:.1f}%")

    return {
        "label": label,
        "num_blocks": num_blocks,
        "mode": mode,
        "use_sd": use_sd,
        "wall_time_s": round(wall_time, 2),
        "throughput_tps": round(throughput, 2),
        "avg_per_req_tps": round(avg_tps_per_req, 2),
        "avg_latency_s": round(avg_latency, 2),
        "total_tokens": total_tokens,
        "ok_requests": len(ok_results),
        "failed_requests": fail_count,
        **metrics,
    }


def main():
    log(f"Concurrent Pressure Sweep — {TIMESTAMP}")
    log(f"Block levels: {BLOCK_LEVELS}")
    log(f"K={K}, Prompts={NUM_PROMPTS}, Concurrency={CONCURRENCY}")
    log(f"Results: {RESULTS_DIR}")

    prompts = load_prompts()
    all_results = []
    total = len(BLOCK_LEVELS) * 2
    idx = 0

    for blocks in BLOCK_LEVELS:
        for use_sd in [False, True]:
            idx += 1
            log(f"\n>>> Experiment {idx}/{total}")
            result = run_experiment(blocks, use_sd, prompts)
            if result:
                all_results.append(result)
            else:
                log(f"  => SKIPPED (server failed)")

    # Save raw results
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV
    if all_results:
        keys = all_results[0].keys()
        with open(RESULTS_DIR / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)

    # Summary table
    log(f"\n{'='*80}")
    log(f"  RESULTS SUMMARY")
    log(f"{'='*80}")
    log(f"{'Blocks':>8} {'Base tps':>10} {'SD tps':>10} {'Speedup':>10} {'AccRate':>10} {'PeakKV%':>10}")
    log(f"{'-'*80}")

    speedups = []
    for blocks in BLOCK_LEVELS:
        base = next((r for r in all_results if r["num_blocks"] == blocks and not r["use_sd"]), None)
        sd = next((r for r in all_results if r["num_blocks"] == blocks and r["use_sd"]), None)
        if base and sd:
            speedup = sd["throughput_tps"] / base["throughput_tps"] if base["throughput_tps"] > 0 else 0
            speedups.append({
                "blocks": blocks,
                "base_throughput": base["throughput_tps"],
                "sd_throughput": sd["throughput_tps"],
                "speedup": round(speedup, 3),
                "base_avg_lat": base["avg_latency_s"],
                "sd_avg_lat": sd["avg_latency_s"],
                "acc_rate": sd.get("avg_acceptance_rate"),
                "mean_acc_len": sd.get("avg_mean_acceptance_length"),
                "peak_kv_base": base.get("peak_kv_usage_pct"),
                "peak_kv_sd": sd.get("peak_kv_usage_pct"),
            })
            ar = f"{sd.get('avg_acceptance_rate', 0):.1f}%" if sd.get('avg_acceptance_rate') else "?"
            kv = f"{sd.get('peak_kv_usage_pct', 0):.1f}%" if sd.get('peak_kv_usage_pct') else "?"
            log(f"{blocks:>8} {base['throughput_tps']:>10.1f} {sd['throughput_tps']:>10.1f} {speedup:>10.3f}x {ar:>10} {kv:>10}")

    with open(RESULTS_DIR / "speedup.json", "w") as f:
        json.dump(speedups, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log("Done!")


if __name__ == "__main__":
    main()
