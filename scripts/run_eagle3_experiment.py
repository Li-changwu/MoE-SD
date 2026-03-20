#!/usr/bin/env python3
"""
EAGLE-3 Ablation Experiment — vLLM Native + SpecMoE Optimizations
===================================================================

Uses vLLM 0.16.0 built-in EAGLE-3 speculative decoding (via speculators format)
to run the 5-config ablation experiment.

Protocol:
  - Sequential single-request (no concurrency, no Poisson arrivals)
  - Each request completes before the next one is sent
  - ShareGPT real prompts for meaningful draft prediction
  - /v1/chat/completions with chat template (thinking disabled)
  - temperature=0 for deterministic acceptance

Configurations:
  C1  No-SD Baseline         Pure autoregressive decoding
  C2  EAGLE-3 Vanilla        Standard SD (K=3), no optimization
  C3  + SpecFusedMoE         C2 + cross-token expert deduplication
  C4  + SDD                  C3 + layer-wise early termination
  C5  Full SpecMoE           C4 + expert cache + prefetch

Architecture:
  Process A — C1: vllm serve (no speculative config)
  Process B — C2-C5: specmoe_server_v2.py (single model load, runtime config switch)

Metrics:
  Output TPS (tok/s)         total_output_tokens / total_time
  Avg TTFT (ms)              first token time - request send time
  Avg E2E Latency (s)        last token time - request send time
  Acceptance Length τ̄         from vLLM /metrics (num_accepted / num_drafts)
  Speedup vs C1              TPS_config / TPS_c1

Usage:
  python scripts/run_eagle3_experiment.py                          # Full C1-C5
  python scripts/run_eagle3_experiment.py --configs 1,2            # C1 vs C2 only
  python scripts/run_eagle3_experiment.py --max-tokens 256         # Longer generation
  python scripts/run_eagle3_experiment.py --num-prompts 20         # Quick test
  python scripts/run_eagle3_experiment.py --dry-run                # Show plan only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR_PATH = "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATASET_PATH = "/root/MoE-SD/data/combined_sharegpt.json"
RESULT_DIR = Path("/root/MoE-SD/results/eagle3_ablation")
CONDA_PYTHON = "/opt/miniconda3/envs/moe-sd/bin/python"
VLLM_BIN = "/opt/miniconda3/envs/moe-sd/bin/vllm"
SERVE_PORT = 8192

ENGINE_PARAMS = {
    "gpu_memory_utilization": 0.90,
    "cpu_offload_gb": 30,
    "max_model_len": 4096,
    "dtype": "bfloat16",
    "enforce_eager": True,
    "trust_remote_code": True,
}

EAGLE3_SPEC_CONFIG = {
    "method": "eagle3",
    "model": SPECULATOR_PATH,
    "num_speculative_tokens": 3,
}

CONFIG_LABELS = {
    1: "C1_no_sd_baseline",
    2: "C2_eagle3_vanilla",
    3: "C3_specmoe_dedup",
    4: "C4_specmoe_sdd",
    5: "C5_specmoe_full",
}

CONFIG_DESCRIPTIONS = {
    1: "No-SD Baseline (pure AR)",
    2: "EAGLE-3 Vanilla (K=3)",
    3: "+ SpecFusedMoE (expert dedup)",
    4: "+ SDD (early termination)",
    5: "Full SpecMoE (+ cache + prefetch)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def wait_for_server(port, timeout=600):
    """Poll /health until ready."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        attempt += 1
        if attempt % 30 == 0:
            log(f"Still waiting for server... ({attempt * 2}s)")
        time.sleep(2)
    return False


def post_json(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def fetch_prometheus_metrics(port):
    """Fetch and parse Prometheus metrics from vLLM /metrics endpoint."""
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode()

        metrics = {}
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            # Parse: metric_name{labels} value  or  metric_name value
            match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?.*?\}?\s+([\d.eE+-]+|NaN|Inf|-Inf)$', line)
            if match:
                name, val = match.group(1), match.group(2)
                try:
                    metrics[name] = float(val)
                except ValueError:
                    pass
        return metrics
    except Exception:
        return {}


def get_spec_decode_metrics(port):
    """Extract speculative decoding metrics from Prometheus endpoint."""
    metrics = fetch_prometheus_metrics(port)
    result = {}

    drafts = metrics.get("vllm:spec_decode_num_drafts_total", 0)
    draft_tokens = metrics.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens_total", 0)

    result["num_drafts"] = drafts
    result["num_draft_tokens"] = draft_tokens
    result["num_accepted_tokens"] = accepted

    if draft_tokens > 0:
        result["acceptance_rate"] = accepted / draft_tokens
    if drafts > 0:
        result["mean_acceptance_length"] = accepted / drafts

    return result


def load_sharegpt_prompts(dataset_path, num_prompts, seed=42):
    """Load prompts from ShareGPT format dataset."""
    import random
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


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential Benchmark Client (inline)
# ═══════════════════════════════════════════════════════════════════════════════

# System prompt to disable Qwen3 thinking mode (speculator trained with thinking off)
NO_THINK_SYSTEM = "You are a helpful assistant. /no_think"


def run_sequential_benchmark(base_url, model, prompts, max_tokens, warmup=1):
    """
    Send prompts sequentially (one at a time, wait for completion).
    Uses streaming /v1/chat/completions with thinking disabled.

    Returns list of per-request results.
    """
    import http.client
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    results = []

    for i, prompt in enumerate(prompts):
        is_warmup = i < warmup

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": NO_THINK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()

        t_start = time.perf_counter()
        ttft = None
        output_tokens = 0
        generated_text = ""
        status_code = 0

        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600)
            conn.request("POST", "/v1/chat/completions", body=payload,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            status_code = resp.status

            if status_code == 200:
                # Read SSE stream for TTFT
                buffer = b""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buffer += chunk

                    # Record TTFT at first data chunk
                    if ttft is None:
                        ttft = time.perf_counter() - t_start

                # Parse the streamed response to count tokens
                text = buffer.decode(errors="replace")
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            event = json.loads(line[6:])
                            choices = event.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                delta_text = delta.get("content", "")
                                if delta_text:
                                    generated_text += delta_text
                            # Check usage in final event
                            usage = event.get("usage")
                            if usage and usage.get("completion_tokens"):
                                output_tokens = usage["completion_tokens"]
                        except json.JSONDecodeError:
                            pass

                # Fallback token count from generated text
                if output_tokens == 0 and generated_text:
                    output_tokens = max(1, len(generated_text) // 4)
            else:
                body = resp.read()
                log(f"Request {i}: HTTP {status_code}", "WARN")

            conn.close()
        except Exception as e:
            log(f"Request {i}: {e}", "ERR")
            continue

        t_end = time.perf_counter()
        e2e_latency = t_end - t_start

        if ttft is None:
            ttft = e2e_latency

        result = {
            "request_id": i,
            "is_warmup": is_warmup,
            "status": status_code,
            "ttft_s": ttft,
            "e2e_latency_s": e2e_latency,
            "output_tokens": output_tokens,
            "prompt_len_chars": len(prompt),
        }
        results.append(result)

        if not is_warmup:
            tps = output_tokens / e2e_latency if e2e_latency > 0 else 0
            log(f"  req {i:3d}: {output_tokens:4d} tok, "
                f"TTFT={ttft*1000:.0f}ms, E2E={e2e_latency:.1f}s, "
                f"TPS={tps:.2f}")

    return results


def run_sequential_benchmark_nonstreaming(base_url, model, prompts, max_tokens, warmup=1):
    """
    Non-streaming fallback: send prompts one at a time via /v1/chat/completions.
    More reliable token counting via usage field.
    Thinking mode disabled for speculator compatibility.
    """
    results = []

    for i, prompt in enumerate(prompts):
        is_warmup = i < warmup

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": NO_THINK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()

        t_start = time.perf_counter()

        try:
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                ttft = time.perf_counter() - t_start
                body = json.loads(resp.read())

            output_tokens = body.get("usage", {}).get("completion_tokens", 0)
            if output_tokens == 0:
                choices = body.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    text = msg.get("content", "")
                    output_tokens = max(1, len(text) // 4)

            status_code = 200
        except Exception as e:
            log(f"Request {i}: {e}", "ERR")
            ttft = time.perf_counter() - t_start
            output_tokens = 0
            status_code = 0

        t_end = time.perf_counter()
        e2e_latency = t_end - t_start

        result = {
            "request_id": i,
            "is_warmup": is_warmup,
            "status": status_code,
            "ttft_s": ttft,
            "e2e_latency_s": e2e_latency,
            "output_tokens": output_tokens,
            "prompt_len_chars": len(prompt),
        }
        results.append(result)

        if not is_warmup:
            tps = output_tokens / e2e_latency if e2e_latency > 0 else 0
            log(f"  req {i:3d}: {output_tokens:4d} tok, "
                f"TTFT={ttft*1000:.0f}ms, E2E={e2e_latency:.1f}s, "
                f"TPS={tps:.2f}")

    return results


def compute_metrics(results, warmup=1):
    """Compute aggregate metrics from sequential benchmark results."""
    # Exclude warmup and failed requests
    valid = [r for r in results if not r["is_warmup"] and r["status"] == 200 and r["output_tokens"] > 0]

    if not valid:
        return {"status": "no_successful_requests"}

    ttfts_ms = [r["ttft_s"] * 1000 for r in valid]
    e2e_s = [r["e2e_latency_s"] for r in valid]
    output_tokens = [r["output_tokens"] for r in valid]

    total_output_tokens = sum(output_tokens)
    total_time = sum(e2e_s)

    def percentile(vals, p):
        s = sorted(vals)
        idx = min(int(len(s) * p / 100), len(s) - 1)
        return s[idx]

    metrics = {
        "status": "ok",
        "num_requests": len(valid),
        "total_output_tokens": total_output_tokens,
        "total_time_s": total_time,
        # Primary metric: output TPS
        "output_tps": total_output_tokens / total_time if total_time > 0 else 0,
        # TTFT
        "mean_ttft_ms": sum(ttfts_ms) / len(ttfts_ms),
        "p50_ttft_ms": percentile(ttfts_ms, 50),
        "p95_ttft_ms": percentile(ttfts_ms, 95),
        "p99_ttft_ms": percentile(ttfts_ms, 99),
        # E2E latency
        "mean_e2e_s": sum(e2e_s) / len(e2e_s),
        "p50_e2e_s": percentile(e2e_s, 50),
        "p95_e2e_s": percentile(e2e_s, 95),
        "p99_e2e_s": percentile(e2e_s, 99),
        # Per-request output tokens
        "mean_output_tokens": sum(output_tokens) / len(output_tokens),
    }

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Server Management
# ═══════════════════════════════════════════════════════════════════════════════

def start_vanilla_server(log_dir):
    """Start vLLM serve without speculative decoding (C1 baseline)."""
    cmd = [
        VLLM_BIN, "serve", MODEL_PATH,
        "--port", str(SERVE_PORT),
        "--gpu-memory-utilization", str(ENGINE_PARAMS["gpu_memory_utilization"]),
        "--cpu-offload-gb", str(ENGINE_PARAMS["cpu_offload_gb"]),
        "--max-model-len", str(ENGINE_PARAMS["max_model_len"]),
        "--dtype", ENGINE_PARAMS["dtype"],
        "--trust-remote-code",
    ]
    if ENGINE_PARAMS.get("enforce_eager"):
        cmd.append("--enforce-eager")

    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = open(log_dir / "server.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    log("Starting vLLM server (no SD)...")
    proc = subprocess.Popen(
        cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
    )
    return proc, server_log


def start_eagle3_server(log_dir):
    """Start specmoe_server_v2 with EAGLE-3 (C2-C5)."""
    server_script = str(Path(__file__).parent / "specmoe_server_v2.py")
    cmd = [
        CONDA_PYTHON, server_script,
        "--model", MODEL_PATH,
        "--port", str(SERVE_PORT),
        "--gpu-memory-utilization", str(ENGINE_PARAMS["gpu_memory_utilization"]),
        "--cpu-offload-gb", str(ENGINE_PARAMS["cpu_offload_gb"]),
        "--max-model-len", str(ENGINE_PARAMS["max_model_len"]),
        "--dtype", ENGINE_PARAMS["dtype"],
        "--trust-remote-code",
        "--speculative-config", json.dumps(EAGLE3_SPEC_CONFIG),
    ]
    if ENGINE_PARAMS.get("enforce_eager"):
        cmd.append("--enforce-eager")

    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = open(log_dir / "server.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    log("Starting SpecMoE-v2 server (EAGLE-3 + hooks)...")
    proc = subprocess.Popen(
        cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
        cwd="/root/MoE-SD",
    )
    return proc, server_log


def stop_server(proc, log_file):
    """Gracefully stop a server process."""
    log("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    log_file.close()
    time.sleep(5)  # Allow GPU memory to be freed


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(cfg_id, prompts, max_tokens, warmup, out_dir, repetitions=1):
    """Run benchmark for a single config, return aggregated metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = CONFIG_LABELS[cfg_id]

    log(f"{'─'*60}")
    log(f"Benchmarking {label}: {CONFIG_DESCRIPTIONS[cfg_id]}")
    log(f"  Prompts: {len(prompts)}, max_tokens: {max_tokens}, warmup: {warmup}")
    log(f"{'─'*60}")

    # Capture spec decode metrics before/after
    sd_before = get_spec_decode_metrics(SERVE_PORT) if cfg_id >= 2 else {}

    all_run_metrics = []
    for rep in range(repetitions):
        if repetitions > 1:
            log(f"  Run {rep+1}/{repetitions}")

        results = run_sequential_benchmark_nonstreaming(
            base_url=f"http://127.0.0.1:{SERVE_PORT}",
            model=MODEL_PATH,
            prompts=prompts,
            max_tokens=max_tokens,
            warmup=warmup,
        )

        metrics = compute_metrics(results, warmup=warmup)
        all_run_metrics.append(metrics)

        # Save per-run data
        run_dir = out_dir / f"run_{rep}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "requests.json").write_text(json.dumps(results, indent=2))
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Capture spec decode metrics after
    sd_after = get_spec_decode_metrics(SERVE_PORT) if cfg_id >= 2 else {}

    # Aggregate across repetitions
    valid_runs = [m for m in all_run_metrics if m.get("status") == "ok"]
    if not valid_runs:
        return {"config_id": cfg_id, "label": label, "status": "failed"}

    def mean_field(field):
        vals = [m[field] for m in valid_runs if field in m]
        return sum(vals) / len(vals) if vals else 0

    aggregated = {
        "config_id": cfg_id,
        "label": label,
        "description": CONFIG_DESCRIPTIONS[cfg_id],
        "status": "ok",
        "repetitions": len(valid_runs),
        # Primary metrics
        "output_tps": mean_field("output_tps"),
        "mean_ttft_ms": mean_field("mean_ttft_ms"),
        "mean_e2e_s": mean_field("mean_e2e_s"),
        "total_output_tokens": mean_field("total_output_tokens"),
        "total_time_s": mean_field("total_time_s"),
        # Percentiles (from first valid run)
        "p50_ttft_ms": valid_runs[0].get("p50_ttft_ms", 0),
        "p95_ttft_ms": valid_runs[0].get("p95_ttft_ms", 0),
        "p50_e2e_s": valid_runs[0].get("p50_e2e_s", 0),
        "p95_e2e_s": valid_runs[0].get("p95_e2e_s", 0),
    }

    # Add spec decode acceptance metrics (delta)
    if sd_after and sd_before:
        delta_drafts = sd_after.get("num_drafts", 0) - sd_before.get("num_drafts", 0)
        delta_accepted = sd_after.get("num_accepted_tokens", 0) - sd_before.get("num_accepted_tokens", 0)
        delta_draft_tokens = sd_after.get("num_draft_tokens", 0) - sd_before.get("num_draft_tokens", 0)

        if delta_drafts > 0:
            aggregated["acceptance_length"] = delta_accepted / delta_drafts
        if delta_draft_tokens > 0:
            aggregated["acceptance_rate"] = delta_accepted / delta_draft_tokens
        aggregated["spec_decode_metrics"] = {
            "drafts": delta_drafts,
            "draft_tokens": delta_draft_tokens,
            "accepted_tokens": delta_accepted,
        }

    # Save aggregated
    (out_dir / "aggregated.json").write_text(json.dumps(aggregated, indent=2))

    return aggregated


def run_process_a(prompts, max_tokens, warmup, repetitions):
    """Process A: Config 1 (no-SD baseline)."""
    out_dir = RESULT_DIR / CONFIG_LABELS[1]
    log_dir = RESULT_DIR / "server_logs" / "c1"

    proc, log_file = start_vanilla_server(log_dir)
    try:
        if not wait_for_server(SERVE_PORT, timeout=600):
            log("Server timeout!", "ERR")
            return {"config_id": 1, "label": CONFIG_LABELS[1], "status": "server_timeout"}

        log("Server ready.")
        return run_config(1, prompts, max_tokens, warmup, out_dir, repetitions)
    finally:
        stop_server(proc, log_file)


def run_process_b(configs, prompts, max_tokens, warmup, repetitions):
    """Process B: Configs 2-5 (EAGLE-3 + SpecMoE variants, single model load)."""
    log_dir = RESULT_DIR / "server_logs" / "c2_c5"

    proc, log_file = start_eagle3_server(log_dir)
    results = []
    try:
        if not wait_for_server(SERVE_PORT, timeout=600):
            log("Server timeout!", "ERR")
            return [{"config_id": c, "label": CONFIG_LABELS[c], "status": "server_timeout"} for c in configs]

        log("EAGLE-3 server ready.")

        for cfg_id in configs:
            out_dir = RESULT_DIR / CONFIG_LABELS[cfg_id]

            # Configure SpecMoE hooks via REST API
            log(f"Configuring hook for C{cfg_id}...")
            try:
                resp = post_json(
                    f"http://127.0.0.1:{SERVE_PORT}/specmoe/configure",
                    {"cfg_id": cfg_id},
                )
                log(f"  Hook response: {resp}")
            except Exception as e:
                log(f"  Hook config failed: {e}", "WARN")

            result = run_config(cfg_id, prompts, max_tokens, warmup, out_dir, repetitions)
            results.append(result)

    finally:
        stop_server(proc, log_file)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Summary & Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results, max_tokens):
    """Print formatted summary table."""
    print()
    print("=" * 100)
    print(f"  EAGLE-3 ABLATION RESULTS — Sequential Single-Request, max_tokens={max_tokens}")
    print("=" * 100)

    header = (f"{'Config':<28} {'Status':<7} {'TPS':>8} {'Speedup':>9} "
              f"{'TTFT(ms)':>10} {'E2E(s)':>8} {'τ̄':>6} {'α':>6}")
    print(header)
    print("─" * 100)

    baseline_tps = None
    for r in all_results:
        label = r.get("label", "?")
        status = r.get("status", "?")

        if status != "ok":
            print(f"{label:<28} {status:<7}")
            continue

        tps = r.get("output_tps", 0)
        ttft = r.get("mean_ttft_ms", 0)
        e2e = r.get("mean_e2e_s", 0)
        tau = r.get("acceptance_length", 0)
        alpha = r.get("acceptance_rate", 0)

        if baseline_tps is None and r.get("config_id") == 1:
            baseline_tps = tps

        speedup = ""
        if baseline_tps and baseline_tps > 0:
            sp = tps / baseline_tps
            speedup = f"{sp:.3f}x"
            if sp < 1.0:
                pct = (sp - 1) * 100
                speedup = f"{sp:.3f}x ({pct:+.1f}%)"

        tau_str = f"{tau:.2f}" if tau > 0 else "—"
        alpha_str = f"{alpha:.1%}" if alpha > 0 else "—"

        print(f"{label:<28} {status:<7} {tps:>7.2f} {speedup:>9} "
              f"{ttft:>10.0f} {e2e:>8.1f} {tau_str:>6} {alpha_str:>6}")

    print("─" * 100)
    print()
    print("Legend:")
    print("  TPS      = Output tokens per second (total_output_tokens / total_time)")
    print("  Speedup  = TPS / C1_TPS")
    print("  TTFT     = Time to first token (ms)")
    print("  E2E      = End-to-end latency per request (s)")
    print("  τ̄        = Mean acceptance length (accepted_tokens / num_drafts)")
    print("  α        = Draft acceptance rate (accepted_tokens / draft_tokens)")
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EAGLE-3 Ablation Experiment (vLLM Native)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--configs", type=str, default="1,2,3,4,5",
                        help="Configs to run: 1=no-SD, 2=EAGLE3, 3=+dedup, 4=+SDD, 5=full")
    parser.add_argument("--num-prompts", type=int, default=50,
                        help="Number of ShareGPT prompts (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens per request (default: 128)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup prompts to skip (default: 1)")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Repetitions per config (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = sorted(int(x.strip()) for x in args.configs.split(","))
    has_c1 = 1 in configs
    eagle_configs = [c for c in configs if c >= 2]

    # Load prompts
    prompts = load_sharegpt_prompts(DATASET_PATH, args.num_prompts, args.seed)
    log(f"Loaded {len(prompts)} prompts from dataset")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Print experiment plan
    print()
    print("=" * 72)
    print("  EAGLE-3 Ablation Experiment")
    print("  vLLM 0.16.0 + Official EAGLE-3 Speculative Decoding")
    print("=" * 72)
    print(f"  Model:        {MODEL_PATH.split('/')[-1]}")
    print(f"  Speculator:   {SPECULATOR_PATH.split('/')[-1]}")
    print(f"  Dataset:      ShareGPT ({len(prompts)} prompts)")
    print(f"  Protocol:     Sequential single-request (no concurrency)")
    print(f"  max_tokens:   {args.max_tokens}")
    print(f"  temperature:  0 (deterministic)")
    print(f"  API:          /v1/chat/completions (thinking disabled)")
    print(f"  warmup:       {args.warmup} prompts")
    print(f"  repetitions:  {args.repetitions}")
    print(f"  GPU:          RTX A6000 (48GB) + {ENGINE_PARAMS['cpu_offload_gb']}GB CPU offload")
    print(f"  Configs:      {configs}")
    print()
    for c in configs:
        print(f"    C{c}: {CONFIG_DESCRIPTIONS[c]}")
    print()
    if has_c1:
        print(f"  Process A:  C1 (separate vLLM serve, no SD)")
    if eagle_configs:
        print(f"  Process B:  {['C'+str(c) for c in eagle_configs]} (single EAGLE-3 model load)")
    print("=" * 72)

    if args.dry_run:
        print("\n  [DRY RUN] — exiting")
        return

    # Save experiment config
    exp_config = {
        "configs": configs,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "model": MODEL_PATH,
        "speculator": SPECULATOR_PATH,
        "engine_params": ENGINE_PARAMS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULT_DIR / "experiment_config.json").write_text(json.dumps(exp_config, indent=2))

    all_results = []

    # Process A: Config 1 (no-SD baseline)
    if has_c1:
        log("=" * 60)
        log("PROCESS A: Config 1 No-SD Baseline")
        log("=" * 60)
        r = run_process_a(prompts, args.max_tokens, args.warmup, args.repetitions)
        all_results.append(r)
        # Save partial results
        (RESULT_DIR / "partial_results.json").write_text(
            json.dumps(all_results, indent=2, default=str))

    # Process B: Configs 2-5 (EAGLE-3 variants)
    if eagle_configs:
        log("=" * 60)
        log(f"PROCESS B: Configs {eagle_configs} (EAGLE-3 + SpecMoE)")
        log("=" * 60)
        rs = run_process_b(eagle_configs, prompts, args.max_tokens, args.warmup, args.repetitions)
        all_results.extend(rs)
        (RESULT_DIR / "partial_results.json").write_text(
            json.dumps(all_results, indent=2, default=str))

    # Save final results
    (RESULT_DIR / "final_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))

    # Print summary
    print_summary(all_results, args.max_tokens)

    log(f"All results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
