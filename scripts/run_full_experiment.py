#!/usr/bin/env python3
"""
SpecMoE Full Experiment Suite v4 — Real Dataset + Two-Process
=============================================================
Uses real ShareGPT prompts (not random tokens) so speculative decoding
draft models can predict effectively — essential for meaningful SD results.

Architecture:
  Process A — Config 1 (no-SD):  vllm serve + bench_serving_sharegpt.py
  Process B — Configs 2-5 (EAGLE-3):
      Serve      : specmoe_server_v2.py  + bench_serving_sharegpt.py (single load)
      Throughput  : vllm bench throughput --dataset-name sharegpt (per config)

Configurations:
  1  Baseline (no-SD)        Pure autoregressive decoding
  2  EAGLE-3 vanilla         Standard speculative decoding (K=3)
  3  +SpecFusedMoE           EAGLE-3 + cross-token expert deduplication
  4  +SDD                    EAGLE-3 + dedup + SDD early termination
  5  Full SpecMoE            EAGLE-3 + dedup + SDD + expert cache

Usage:
  python scripts/run_full_experiment.py --configs 1,2,3,4,5 --modes serve,throughput
  python scripts/run_full_experiment.py --configs 2,3,4,5 --modes serve
  python scripts/run_full_experiment.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# == Constants ================================================================
MODEL_PATH = "/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR_PATH = "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATASET_PATH = "/root/MoE-SD/data/combined_sharegpt.json"
RESULT_DIR = Path("/root/MoE-SD/results/full_experiment")
VLLM_BIN = "/opt/miniconda3/envs/moe-sd/bin/vllm"
CONDA_PYTHON = "/opt/miniconda3/envs/moe-sd/bin/python"
BENCH_CLIENT = str(Path(__file__).parent / "bench_serving_sharegpt.py")
SERVE_PORT = 8192

ENGINE_PARAMS = {
    "gpu_memory_utilization": 0.90,
    "cpu_offload_gb": 30,
    "max_model_len": 4096,
    "dtype": "bfloat16",
    "enforce_eager": True,
    "trust_remote_code": True,
}

# Serve benchmark: real prompts, 50 samples, max 128 output tokens, QPS=1.0
SERVE_PARAMS = {
    "num_prompts": 50,
    "max_tokens": 128,
    "request_rate": 1.0,
    "seed": 42,
}

# Throughput benchmark: sharegpt dataset, 50 prompts, max 128 output tokens
THROUGHPUT_PARAMS = {
    "num_prompts": 50,
    "output_len": 128,
}

EAGLE3_SPEC_CONFIG = {
    "method": "eagle3",
    "model": SPECULATOR_PATH,
    "num_speculative_tokens": 3,
}

CONFIG_LABELS = {
    1: "1_no_sd_baseline",
    2: "2_eagle3_vanilla",
    3: "3_specmoe_dedup",
    4: "4_specmoe_dedup_sdd",
    5: "5_specmoe_full",
}


# == Utilities ================================================================

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
            print(f"    ... still waiting ({attempt} attempts)")
        time.sleep(2)
    return False


def post_json(url, data):
    """POST JSON to a URL and return response dict."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def make_server_cmd(speculative_config=None):
    """Build vllm serve command."""
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
    if speculative_config:
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])
    return cmd


def run_bench_serve(label, out_dir):
    """Run sharegpt serving benchmark client, return parsed results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        CONDA_PYTHON, BENCH_CLIENT,
        "--base-url", f"http://127.0.0.1:{SERVE_PORT}",
        "--model", MODEL_PATH,
        "--dataset", DATASET_PATH,
        "--num-prompts", str(SERVE_PARAMS["num_prompts"]),
        "--max-tokens", str(SERVE_PARAMS["max_tokens"]),
        "--request-rate", str(SERVE_PARAMS["request_rate"]),
        "--seed", str(SERVE_PARAMS["seed"]),
        "--result-dir", str(out_dir),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
    elapsed = time.time() - t0

    (out_dir / "bench_stdout.log").write_text(proc.stdout)
    (out_dir / "bench_stderr.log").write_text(proc.stderr)

    print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)

    if proc.returncode != 0:
        print(f"    [FAILED] bench exit={proc.returncode}")
        for line in proc.stderr.strip().split("\n")[-10:]:
            print(f"      {line}")
        return {"label": label, "mode": "serve", "status": "bench_failed", "elapsed_s": elapsed}

    # Read saved metrics
    metrics_file = out_dir / "serve_metrics.json"
    results = {}
    if metrics_file.exists():
        results = json.loads(metrics_file.read_text())

    results.update({"label": label, "mode": "serve", "status": "ok", "bench_elapsed_s": elapsed})
    return results


def print_serve_metrics(results):
    for key, fmt, lbl in [
        ("output_throughput_tok_s", ".2f", "Output tput"),
        ("request_throughput_req_s", ".3f", "Req tput"),
        ("mean_ttft_ms", ".1f", "Mean TTFT"),
        ("mean_tpot_ms", ".1f", "Mean TPOT"),
        ("mean_itl_ms", ".1f", "Mean ITL"),
        ("duration_s", ".1f", "Duration"),
    ]:
        if key in results:
            print(f"    {lbl:<14}: {results[key]:{fmt}}")


# == Process A: Config 1 (no-SD baseline) ====================================

def process_a_serve():
    """Config 1 serve: start vanilla vllm serve -> sharegpt bench -> stop."""
    label = CONFIG_LABELS[1]
    out_dir = RESULT_DIR / label / "serve"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  [Process A / SERVE] {label}")
    print(f"  Dataset: {DATASET_PATH} ({SERVE_PARAMS['num_prompts']} prompts)")
    print(f"{'='*72}")

    server_cmd = make_server_cmd()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    print("  Starting vanilla vLLM server...")
    server_log = open(out_dir / "server.log", "w")
    server_proc = subprocess.Popen(
        server_cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
    )

    try:
        if not wait_for_server(SERVE_PORT, timeout=600):
            print("  [FAILED] Server timeout")
            server_proc.terminate()
            server_proc.wait(timeout=30)
            server_log.close()
            return {"label": label, "mode": "serve", "status": "server_timeout"}

        print("  Server ready. Running ShareGPT benchmark...")
        result = run_bench_serve(label, out_dir)

        if result.get("status") == "ok":
            print(f"  [OK]")
            print_serve_metrics(result)
        return result
    finally:
        print("  Stopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
        server_log.close()
        time.sleep(3)


def process_a_throughput():
    """Config 1 throughput: vllm bench throughput --dataset-name sharegpt."""
    label = CONFIG_LABELS[1]
    out_dir = RESULT_DIR / label / "throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "throughput.json"

    print(f"\n{'='*72}")
    print(f"  [Process A / THROUGHPUT] {label}")
    print(f"{'='*72}")

    cmd = [
        VLLM_BIN, "bench", "throughput",
        "--model", MODEL_PATH,
        "--dataset-name", "sharegpt",
        "--dataset-path", DATASET_PATH,
        "--num-prompts", str(THROUGHPUT_PARAMS["num_prompts"]),
        "--output-len", str(THROUGHPUT_PARAMS["output_len"]),
        "--output-json", str(out_json),
        "--gpu-memory-utilization", str(ENGINE_PARAMS["gpu_memory_utilization"]),
        "--cpu-offload-gb", str(ENGINE_PARAMS["cpu_offload_gb"]),
        "--max-model-len", str(ENGINE_PARAMS["max_model_len"]),
        "--dtype", ENGINE_PARAMS["dtype"],
        "--trust-remote-code",
    ]
    if ENGINE_PARAMS.get("enforce_eager"):
        cmd.append("--enforce-eager")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)

    print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)

    if proc.returncode != 0:
        print(f"  [FAILED] exit={proc.returncode}")
        for line in proc.stderr.strip().split("\n")[-15:]:
            print(f"    {line}")
        return {"label": label, "mode": "throughput", "status": "failed", "wall_time_s": elapsed}

    results = {}
    if out_json.exists():
        results = json.loads(out_json.read_text())

    # Parse stdout for throughput
    for line in proc.stdout.split("\n"):
        if "Throughput:" in line:
            try:
                parts = line.split("Throughput:")[1].strip().split()
                results["throughput_req_s"] = float(parts[0])
                if "tokens/s" in line:
                    for i, p in enumerate(parts):
                        if "tokens/s" in p or (i > 0 and parts[i-1].replace('.','').isdigit()):
                            try:
                                results["throughput_tok_s"] = float(parts[i-1])
                            except (ValueError, IndexError):
                                pass
            except (ValueError, IndexError):
                pass

    results.update({"label": label, "mode": "throughput", "status": "ok", "wall_time_s": elapsed})
    print(f"  [OK] Wall time: {elapsed:.1f}s")
    return results


# == Process B: Configs 2-5 (EAGLE-3, single model load) =====================

def process_b_serve(eagle_configs):
    """Start specmoe_server_v2 once -> cycle sharegpt bench for each config -> stop."""
    print(f"\n{'='*72}")
    print(f"  [Process B / SERVE] Configs {eagle_configs} (single model load)")
    print(f"  Dataset: {DATASET_PATH} ({SERVE_PARAMS['num_prompts']} prompts)")
    print(f"{'='*72}")

    server_cmd = [
        CONDA_PYTHON, str(Path(__file__).parent / "specmoe_server_v2.py"),
        MODEL_PATH,
        "--port", str(SERVE_PORT),
        "--gpu-memory-utilization", str(ENGINE_PARAMS["gpu_memory_utilization"]),
        "--cpu-offload-gb", str(ENGINE_PARAMS["cpu_offload_gb"]),
        "--max-model-len", str(ENGINE_PARAMS["max_model_len"]),
        "--dtype", ENGINE_PARAMS["dtype"],
        "--trust-remote-code",
        "--speculative-config", json.dumps(EAGLE3_SPEC_CONFIG),
    ]
    if ENGINE_PARAMS.get("enforce_eager"):
        server_cmd.append("--enforce-eager")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    server_log_dir = RESULT_DIR / "process_b_serve"
    server_log_dir.mkdir(parents=True, exist_ok=True)
    server_log = open(server_log_dir / "server.log", "w")

    print("  Starting SpecMoE-v2 server (EAGLE-3 + hook)...")
    server_proc = subprocess.Popen(
        server_cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
        cwd="/root/MoE-SD",
    )

    results = []
    try:
        if not wait_for_server(SERVE_PORT, timeout=600):
            print("  [FAILED] Server timeout")
            server_proc.terminate()
            server_proc.wait(timeout=30)
            server_log.close()
            return [{"label": "process_b", "mode": "serve", "status": "server_timeout"}]

        print("  Server ready. Starting benchmark cycle...\n")

        for cfg_id in eagle_configs:
            label = CONFIG_LABELS[cfg_id]
            out_dir = RESULT_DIR / label / "serve"

            print(f"  -- Config {cfg_id}: {label} --")

            # Configure hook via REST
            resp = post_json(
                f"http://127.0.0.1:{SERVE_PORT}/specmoe/configure",
                {"cfg_id": cfg_id},
            )
            print(f"    Hook: {resp}")

            # Run benchmark
            result = run_bench_serve(label, out_dir)
            results.append(result)

            if result.get("status") == "ok":
                print_serve_metrics(result)
            print()

    finally:
        print("  Stopping SpecMoE-v2 server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
        server_log.close()
        time.sleep(3)

    return results


def process_b_throughput(eagle_configs):
    """Run Configs 2-5 throughput in a single subprocess with one model load."""
    print(f"\n{'='*72}")
    print(f"  [Process B / THROUGHPUT] Configs {eagle_configs} (single model load)")
    print(f"{'='*72}")

    harness_code = _generate_throughput_harness(eagle_configs)
    harness_dir = RESULT_DIR / "process_b_throughput"
    harness_dir.mkdir(parents=True, exist_ok=True)
    harness_path = harness_dir / "harness.py"
    harness_path.write_text(harness_code)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

    print(f"  Running throughput harness for configs {eagle_configs}...")
    t0 = time.time()
    proc = subprocess.run(
        [CONDA_PYTHON, str(harness_path)],
        env=env, capture_output=True, text=True,
        timeout=7200, cwd="/root/MoE-SD",
    )
    elapsed = time.time() - t0

    (harness_dir / "stdout.log").write_text(proc.stdout)
    (harness_dir / "stderr.log").write_text(proc.stderr)

    if len(proc.stdout) > 4000:
        print(proc.stdout[-4000:])
    else:
        print(proc.stdout)

    if proc.returncode != 0:
        print(f"  [FAILED] exit={proc.returncode}")
        for line in proc.stderr.strip().split("\n")[-15:]:
            print(f"    {line}")
        return [{"label": "process_b", "mode": "throughput", "status": "failed"}]

    results = []
    for cfg_id in eagle_configs:
        label = CONFIG_LABELS[cfg_id]
        rfile = RESULT_DIR / label / "throughput" / "throughput.json"
        if rfile.exists():
            r = json.loads(rfile.read_text())
            r.update({"label": label, "mode": "throughput", "status": "ok"})
            results.append(r)
        else:
            results.append({"label": label, "mode": "throughput", "status": "no_result"})

    return results


def _generate_throughput_harness(eagle_configs):
    """Generate self-contained throughput harness — single model load, hook switching."""
    return f'''#!/usr/bin/env python3
"""Auto-generated: Single-load throughput harness for Configs {eagle_configs}."""
import gc, json, os, sys, time
import torch

sys.path.insert(0, "/root/MoE-SD")

MODEL_PATH = "{MODEL_PATH}"
SPECULATOR_PATH = "{SPECULATOR_PATH}"
DATASET_PATH = "{DATASET_PATH}"
RESULT_DIR = "/root/MoE-SD/results/full_experiment"
CONFIGS = {eagle_configs}
OUTPUT_LEN = {THROUGHPUT_PARAMS["output_len"]}
NUM_PROMPTS = {THROUGHPUT_PARAMS["num_prompts"]}

CONFIG_LABELS = {{
    2: "2_eagle3_vanilla",
    3: "3_specmoe_dedup",
    4: "4_specmoe_dedup_sdd",
    5: "5_specmoe_full",
}}


def load_sharegpt_prompts():
    """Load prompts from ShareGPT dataset."""
    import random
    data = json.load(open(DATASET_PATH))
    rng = random.Random(42)
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
        if len(prompts) >= NUM_PROMPTS:
            break
    return prompts


def configure_hook(hook, cfg_id):
    """Configure hook for a specific config."""
    hook.set_verify_mode(False)
    hook._spec_moe = None
    hook._sdd = None
    hook._expert_cache = None
    hook._total_intercepts = 0
    hook._total_specmoe_calls = 0
    hook._total_passthrough_calls = 0

    if cfg_id <= 2:
        return

    if cfg_id >= 3:
        from adapters.triton_spec_moe import SpecFusedMoEDispatcher
        hook.configure(spec_moe=SpecFusedMoEDispatcher())

    if cfg_id >= 4:
        from adapters.layer_early_terminator import SDDConfig, SpeculationDivergenceDetector
        hook._sdd = SpeculationDivergenceDetector(
            config=SDDConfig(min_check_layer=8, method="combined", consecutive_threshold=3),
            num_layers=48)

    if cfg_id >= 5:
        from adapters.expert_cache import ExpertCacheConfig, ExpertWeightCache
        hook._expert_cache = ExpertWeightCache(config=ExpertCacheConfig(
            gpu_budget_bytes=8 * 1024**3, eviction_policy="lru",
            enable_prefetch=True, pin_cpu_memory=True))

    hook.set_verify_mode(True, batch_size=4)


def main():
    from vllm import LLM, SamplingParams
    from adapters.fused_moe_hook import FusedMoEHook

    prompts = load_sharegpt_prompts()
    print(f"[Harness] Loaded {{len(prompts)}} ShareGPT prompts")

    spec_config = {{
        "method": "eagle3",
        "model": SPECULATOR_PATH,
        "num_speculative_tokens": 3,
    }}

    print("[Harness] Loading model with EAGLE-3 speculative decoding...")
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization={ENGINE_PARAMS["gpu_memory_utilization"]},
        cpu_offload_gb={ENGINE_PARAMS["cpu_offload_gb"]},
        max_model_len={ENGINE_PARAMS["max_model_len"]},
        dtype="{ENGINE_PARAMS["dtype"]}",
        enforce_eager={ENGINE_PARAMS["enforce_eager"]},
        trust_remote_code=True,
        speculative_config=spec_config,
    )
    print("[Harness] Model loaded.")

    hook = FusedMoEHook()
    hook.install()
    print("[Harness] Hook installed.")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LEN)

    for cfg_id in CONFIGS:
        label = CONFIG_LABELS[cfg_id]
        print(f"\\n[Harness] =============== Config {{cfg_id}}: {{label}} ===============")

        configure_hook(hook, cfg_id)

        out_dir = os.path.join(RESULT_DIR, label, "throughput")
        os.makedirs(out_dir, exist_ok=True)

        # Warmup (3 prompts)
        print("  Warmup...")
        _ = llm.generate(prompts[:3], sampling_params)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Measure
        print(f"  Running {{len(prompts)}} prompts...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in outputs if o.outputs
        )
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)

        req_throughput = len(prompts) / elapsed
        tok_throughput = total_output_tokens / elapsed
        total_tok_throughput = (total_input_tokens + total_output_tokens) / elapsed

        result = {{
            "elapsed_s": elapsed,
            "num_requests": len(prompts),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput": req_throughput,
            "output_throughput_tok_s": tok_throughput,
            "total_throughput_tok_s": total_tok_throughput,
            "hook_stats": {{
                "total_intercepts": hook._total_intercepts,
                "specmoe_calls": hook._total_specmoe_calls,
                "passthrough_calls": hook._total_passthrough_calls,
            }},
        }}

        with open(os.path.join(out_dir, "throughput.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"  [OK] {{elapsed:.1f}}s | {{tok_throughput:.2f}} out tok/s | {{total_tok_throughput:.2f}} total tok/s")

    hook.uninstall()
    del llm
    gc.collect()
    print("\\n[Harness] Done.")


if __name__ == "__main__":
    main()
'''


# == Summary ==================================================================

def print_summary(all_results):
    serve_results = [r for r in all_results if r.get("mode") == "serve"]
    tput_results = [r for r in all_results if r.get("mode") == "throughput"]

    if serve_results:
        print(f"\n{'='*90}")
        print(f"  SERVE BENCHMARK (ShareGPT, {SERVE_PARAMS['num_prompts']} prompts, QPS={SERVE_PARAMS['request_rate']})")
        print(f"{'='*90}")
        hdr = (f"{'Config':<24} {'Status':<7} {'Out tput':>9} {'Req tput':>9} "
               f"{'TTFT':>9} {'TPOT':>9} {'ITL':>9} {'Dur':>8}")
        print(hdr)
        print("-" * 90)
        baseline_tput = None
        for r in serve_results:
            label = r.get("label", "?")
            status = r.get("status", "?")
            if status != "ok":
                print(f"{label:<24} {status:<7}")
                continue
            out_t = r.get("output_throughput_tok_s", 0)
            req_t = r.get("request_throughput_req_s", 0)
            ttft = r.get("mean_ttft_ms", 0)
            tpot = r.get("mean_tpot_ms", 0)
            itl = r.get("mean_itl_ms", 0)
            dur = r.get("duration_s", 0)
            if baseline_tput is None and "no_sd" in label:
                baseline_tput = out_t
            delta = ""
            if baseline_tput and baseline_tput > 0 and out_t > 0:
                pct = (out_t - baseline_tput) / baseline_tput * 100
                delta = f" ({pct:+.1f}%)"
            print(f"{label:<24} {status:<7} {out_t:>8.2f}{delta:>8} {req_t:>9.3f} "
                  f"{ttft:>9.1f} {tpot:>9.1f} {itl:>9.1f} {dur:>7.1f}s")

    if tput_results:
        print(f"\n{'='*80}")
        print(f"  THROUGHPUT BENCHMARK (ShareGPT, {THROUGHPUT_PARAMS['num_prompts']} prompts, batch)")
        print(f"{'='*80}")
        print(f"{'Config':<24} {'Status':<7} {'Out tok/s':>10} {'Total tok/s':>12} {'vs Base':>9}")
        print("-" * 80)
        baseline = None
        for r in tput_results:
            label = r.get("label", "?")
            status = r.get("status", "?")
            if status != "ok":
                print(f"{label:<24} {status:<7}")
                continue
            out_t = r.get("output_throughput_tok_s", r.get("throughput_tok_s", 0))
            tot_t = r.get("total_throughput_tok_s", 0)
            if baseline is None and "no_sd" in label:
                baseline = out_t
            sp = out_t / baseline if baseline and baseline > 0 else 0
            print(f"{label:<24} {status:<7} {out_t:>10.2f} {tot_t:>12.2f} {sp:>8.2f}x")

    print(f"\n{'='*90}")


# == Main =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SpecMoE Full Experiment v4 (ShareGPT + Two-Process)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=str, default="1,2,3,4,5",
                        help="1=no-SD, 2=EAGLE-3, 3=+Dedup, 4=+SDD, 5=Full")
    parser.add_argument("--modes", type=str, default="serve,throughput",
                        help="serve,throughput (default: both)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = [int(x.strip()) for x in args.configs.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    eagle_configs = sorted(c for c in configs if c >= 2)
    has_config1 = 1 in configs

    print("=" * 72)
    print("  SpecMoE Full Experiment Protocol v4")
    print("  ** Real ShareGPT Dataset + Two-Process Architecture **")
    print("=" * 72)
    print(f"  Model:     {MODEL_PATH.split('/')[-1]}")
    print(f"  Dataset:   {DATASET_PATH}")
    print(f"  Hardware:  RTX A6000 (48GB) + {ENGINE_PARAMS['cpu_offload_gb']}GB CPU offload")
    print(f"  Configs:   {configs}")
    print(f"  Modes:     {modes}")
    if has_config1:
        print(f"  Process A: Config 1 (no-SD baseline)")
    if eagle_configs:
        print(f"  Process B: Configs {eagle_configs} (single EAGLE-3 model load)")
    print("=" * 72)

    if args.dry_run:
        print("DRY RUN -- exiting")
        return

    all_results = []

    # -- Process A: Config 1 --
    if has_config1:
        if "serve" in modes:
            r = process_a_serve()
            all_results.append(r)
            (RESULT_DIR / "partial_results.json").write_text(
                json.dumps(all_results, indent=2, default=str))
        if "throughput" in modes:
            r = process_a_throughput()
            all_results.append(r)
            (RESULT_DIR / "partial_results.json").write_text(
                json.dumps(all_results, indent=2, default=str))

    # -- Process B: Configs 2-5 --
    if eagle_configs:
        if "serve" in modes:
            rs = process_b_serve(eagle_configs)
            all_results.extend(rs)
            (RESULT_DIR / "partial_results.json").write_text(
                json.dumps(all_results, indent=2, default=str))
        if "throughput" in modes:
            rs = process_b_throughput(eagle_configs)
            all_results.extend(rs)
            (RESULT_DIR / "partial_results.json").write_text(
                json.dumps(all_results, indent=2, default=str))

    # -- Final summary --
    summary_path = RESULT_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAll results saved to {summary_path}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
