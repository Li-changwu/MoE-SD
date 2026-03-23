#!/usr/bin/env python3
"""Auto-generated: Single-load throughput harness for Configs [2, 3, 4, 5]."""
import gc, json, os, sys, time
import torch

sys.path.insert(0, "/root/MoE-SD")

MODEL_PATH = "/root/models/Qwen3-30B-A3B-Instruct-2507"
SPECULATOR_PATH = "/root/models/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
DATASET_PATH = "/root/MoE-SD/data/combined_sharegpt.json"
RESULT_DIR = "/root/MoE-SD/results/full_experiment"
CONFIGS = [2, 3, 4, 5]
OUTPUT_LEN = 128
NUM_PROMPTS = 50

CONFIG_LABELS = {
    2: "2_eagle3_vanilla",
    3: "3_specmoe_dedup",
    4: "4_specmoe_dedup_sdd",
    5: "5_specmoe_full",
}


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
    print(f"[Harness] Loaded {len(prompts)} ShareGPT prompts")

    spec_config = {
        "method": "eagle3",
        "model": SPECULATOR_PATH,
        "num_speculative_tokens": 3,
    }

    print("[Harness] Loading model with EAGLE-3 speculative decoding...")
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.9,
        cpu_offload_gb=30,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=True,
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
        print(f"\n[Harness] =============== Config {cfg_id}: {label} ===============")

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
        print(f"  Running {len(prompts)} prompts...")
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

        result = {
            "elapsed_s": elapsed,
            "num_requests": len(prompts),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput": req_throughput,
            "output_throughput_tok_s": tok_throughput,
            "total_throughput_tok_s": total_tok_throughput,
            "hook_stats": {
                "total_intercepts": hook._total_intercepts,
                "specmoe_calls": hook._total_specmoe_calls,
                "passthrough_calls": hook._total_passthrough_calls,
            },
        }

        with open(os.path.join(out_dir, "throughput.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"  [OK] {elapsed:.1f}s | {tok_throughput:.2f} out tok/s | {total_tok_throughput:.2f} total tok/s")

    hook.uninstall()
    del llm
    gc.collect()
    print("\n[Harness] Done.")


if __name__ == "__main__":
    main()
