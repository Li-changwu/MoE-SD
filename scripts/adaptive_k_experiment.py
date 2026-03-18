#!/usr/bin/env python3
"""
Adaptive K Comparison Experiment
=================================
Compares three modes at each block level:
  1. baseline   — no speculative decoding
  2. sd_fixed   — SD with fixed K=3 (EAGLE3)
  3. sd_adaptive — SD with dynamic K controlled by StaticGovernor sidecar

Uses the same experimental setup as concurrent_pressure_sweep.py:
  - 50 concurrent real-text prompts from alpaca+dolly dataset
  - Block levels: [8000, 1500, 1000, 800, 600, 300, 200]
  - cpu-offload=30, gpu-memory-utilization=0.90, max-model-len=2048
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

BLOCK_LEVELS = [8000, 600, 300, 200]

DATASET_PATH = "data/combined_sharegpt.json"
NUM_PROMPTS = 50
MAX_TOKENS = 256
CONCURRENCY = 50

# Sidecar config
SIDECAR_SCRIPT = str(Path(__file__).resolve().parent.parent / "adapters" / "metrics_sidecar.py")
SIDECAR_INTERVAL = 0.3  # Poll every 300ms

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"results/adaptive_comparison_{TIMESTAMP}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

K_FILE = "/dev/shm/moe_sd_k"
TRACE_FILE = "/dev/shm/moe_sd_trace.jsonl"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_prompts():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    prompts = [d["conversations"][0]["value"] for d in data[:NUM_PROMPTS]]
    log(f"Loaded {len(prompts)} prompts from {DATASET_PATH}")
    return prompts


def start_server(num_blocks, use_sd, log_file):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
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


def start_sidecar(log_file):
    """Start the metrics sidecar controller."""
    log_path = LOG_DIR / log_file
    f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, SIDECAR_SCRIPT,
         "--port", str(PORT),
         "--interval", str(SIDECAR_INTERVAL),
         "--default-k", str(K)],
        stdout=f, stderr=subprocess.STDOUT,
    )
    return proc, log_path


def stop_sidecar(proc):
    """Stop the sidecar controller."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Cleanup shared memory
    for p in [K_FILE, TRACE_FILE]:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def wait_for_server(timeout=300):
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
    text = log_path.read_text(errors="ignore")
    kv_match = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    kv_tokens = int(kv_match.group(1).replace(",", "")) if kv_match else None
    kv_gib_match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
    kv_gib = float(kv_gib_match.group(1)) if kv_gib_match else None

    acc_rates = re.findall(r"Avg Draft acceptance rate:\s*([\d.]+)%", text)
    mean_acc_lens = re.findall(r"Mean acceptance length:\s*([\d.]+)", text)
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


def parse_sidecar_trace():
    """Parse the sidecar trace to get K decision statistics."""
    if not os.path.exists(TRACE_FILE):
        return {}
    entries = []
    with open(TRACE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        return {}

    ks = [e["k"] for e in entries]
    k_changes = sum(1 for i in range(1, len(ks)) if ks[i] != ks[i-1])
    k_counts = {}
    for k in ks:
        k_counts[k] = k_counts.get(k, 0) + 1

    return {
        "sidecar_steps": len(entries),
        "sidecar_k_changes": k_changes,
        "sidecar_k_distribution": k_counts,
        "sidecar_avg_k": round(sum(ks) / len(ks), 2) if ks else K,
        "sidecar_min_k": min(ks) if ks else K,
        "sidecar_max_k": max(ks) if ks else K,
    }


def run_experiment(num_blocks, mode, prompts):
    """Run one experiment.
    mode: 'baseline', 'sd_fixed', 'sd_adaptive'
    """
    use_sd = mode in ("sd_fixed", "sd_adaptive")
    use_sidecar = (mode == "sd_adaptive")
    label = f"B{num_blocks}_{mode}"

    log(f"\n{'='*60}")
    log(f"  {label}  (blocks={num_blocks}, mode={mode})")
    log(f"{'='*60}")

    log_file = f"adaptive_{label}.log"
    sidecar_log = f"adaptive_{label}_sidecar.log"
    proc, log_path = start_server(num_blocks, use_sd, log_file)

    log("Waiting for server...")
    if not wait_for_server(timeout=300):
        log(f"ERROR: Server failed to start for {label}")
        stop_server(proc)
        return None

    # Start sidecar for adaptive mode
    sidecar_proc = None
    if use_sidecar:
        log("Starting adaptive K sidecar controller...")
        sidecar_proc, sidecar_log_path = start_sidecar(sidecar_log)
        time.sleep(2)  # Let sidecar connect to metrics

    log("Sending warmup request...")
    try:
        send_single_request("Hello!", 0)
    except Exception:
        pass
    time.sleep(2)

    log(f"Sending {len(prompts)} requests concurrently...")
    results, wall_time = send_concurrent_requests(prompts)

    time.sleep(5)

    # Stop sidecar first, then server
    sidecar_stats = {}
    if sidecar_proc:
        sidecar_stats = parse_sidecar_trace()
        stop_sidecar(sidecar_proc)
        # Save sidecar trace
        if os.path.exists(TRACE_FILE):
            import shutil
            shutil.copy2(TRACE_FILE, RESULTS_DIR / f"{label}_sidecar_trace.jsonl")

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
    if sidecar_stats:
        log(f"  Sidecar: avg_k={sidecar_stats.get('sidecar_avg_k')}, "
            f"changes={sidecar_stats.get('sidecar_k_changes')}, "
            f"distribution={sidecar_stats.get('sidecar_k_distribution')}")

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
        **sidecar_stats,
    }


def main():
    log(f"Adaptive K Comparison Experiment — {TIMESTAMP}")
    log(f"Block levels: {BLOCK_LEVELS}")
    log(f"Modes: baseline, sd_fixed (K={K}), sd_adaptive (sidecar)")
    log(f"Results: {RESULTS_DIR}")

    prompts = load_prompts()
    all_results = []
    modes = ["baseline", "sd_fixed", "sd_adaptive"]
    total = len(BLOCK_LEVELS) * len(modes)
    idx = 0

    for blocks in BLOCK_LEVELS:
        for mode in modes:
            idx += 1
            log(f"\n>>> Experiment {idx}/{total}")
            result = run_experiment(blocks, mode, prompts)
            if result:
                all_results.append(result)
            else:
                log(f"  => SKIPPED (server failed)")
            # Save interim results
            with open(RESULTS_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Save CSV
    if all_results:
        keys = list(all_results[0].keys())
        # Collect all keys across results (adaptive has extra sidecar fields)
        for r in all_results:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(RESULTS_DIR / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(all_results)

    # Summary table
    log(f"\n{'='*90}")
    log(f"  RESULTS SUMMARY")
    log(f"{'='*90}")
    log(f"{'Blocks':>8} {'Base':>8} {'Fixed':>8} {'Adapt':>8} {'Fix/B':>8} {'Ada/B':>8} {'Ada/Fix':>8} {'AvgK':>6} {'KV%':>6}")
    log(f"{'-'*90}")

    comparisons = []
    for blocks in BLOCK_LEVELS:
        base = next((r for r in all_results if r["num_blocks"] == blocks and r["mode"] == "baseline"), None)
        fixed = next((r for r in all_results if r["num_blocks"] == blocks and r["mode"] == "sd_fixed"), None)
        adapt = next((r for r in all_results if r["num_blocks"] == blocks and r["mode"] == "sd_adaptive"), None)

        b_tps = base["throughput_tps"] if base else 0
        f_tps = fixed["throughput_tps"] if fixed else 0
        a_tps = adapt["throughput_tps"] if adapt else 0

        fix_over_base = f_tps / b_tps if b_tps > 0 else 0
        ada_over_base = a_tps / b_tps if b_tps > 0 else 0
        ada_over_fix = a_tps / f_tps if f_tps > 0 else 0

        avg_k = adapt.get("sidecar_avg_k", K) if adapt else K
        kv_pct = adapt.get("peak_kv_usage_pct", 0) if adapt else (fixed.get("peak_kv_usage_pct", 0) if fixed else 0)

        log(f"{blocks:>8} {b_tps:>8.1f} {f_tps:>8.1f} {a_tps:>8.1f} "
            f"{fix_over_base:>8.3f} {ada_over_base:>8.3f} {ada_over_fix:>8.3f} "
            f"{avg_k:>6.1f} {kv_pct:>6.1f}")

        comparisons.append({
            "blocks": blocks,
            "base_tps": b_tps,
            "fixed_tps": f_tps,
            "adaptive_tps": a_tps,
            "fixed_over_base": round(fix_over_base, 4),
            "adaptive_over_base": round(ada_over_base, 4),
            "adaptive_over_fixed": round(ada_over_fix, 4),
            "avg_k": avg_k,
            "peak_kv_pct": kv_pct,
        })

    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log("Done!")


if __name__ == "__main__":
    main()
