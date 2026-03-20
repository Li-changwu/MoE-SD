#!/usr/bin/env python3
"""
Expert Routing Overlap Measurement — Direct Router Hook
=========================================================

Loads Qwen3-30B-A3B via transformers (device_map="auto") and captures
per-token, per-layer router decisions via forward hooks on gate modules.

Analyzes:
  1. Pairwise Jaccard overlap between consecutive tokens (window=4, simulating K=3 verify)
  2. Union size (unique experts per verify batch)
  3. Dedup potential (redundant expert computations)
  4. Per-layer overlap distribution

Model: Qwen3-30B-A3B-Instruct-2507 (128 experts, top-8, 48 layers)

Usage:
  python scripts/measure_routing_overlap.py --num-prompts 5 --max-tokens 32
  python scripts/measure_routing_overlap.py --analyze-only /tmp/routing_trace.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48
VERIFY_BATCH_SIZE = 4  # K=3 -> K+1 = 4

NO_THINK_SYSTEM = "You are a helpful assistant. /no_think"


def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ================================================================
# Routing Trace Collection
# ================================================================

class RouterTraceCollector:
    """Captures top-k expert routing decisions per token per layer."""

    def __init__(self, num_experts=128, top_k=8):
        self.num_experts = num_experts
        self.top_k = top_k
        self.events = []
        self._hooks = []
        self._current_request_id = "req_0"
        self._current_token_idx = 0
        self._recording = False  # only record decode-phase routing

    def set_context(self, request_id, token_idx):
        self._current_request_id = request_id
        self._current_token_idx = token_idx

    def _make_hook(self, layer_id):
        def hook_fn(module, input_args, output):
            if not self._recording:
                return

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            if logits.dim() != 2:
                return

            # Only record when batch_size == 1 (decode step)
            if logits.shape[0] != 1:
                return

            probs = torch.softmax(logits.float(), dim=-1)
            top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)

            self.events.append({
                "request_id": self._current_request_id,
                "token_idx": self._current_token_idx,
                "layer_id": layer_id,
                "experts": top_indices[0].cpu().tolist(),
                "probs": [round(p, 6) for p in top_probs[0].cpu().tolist()],
            })

        return hook_fn

    def attach(self, model):
        """Attach hooks to all MoE gate modules (model.layers.X.mlp.gate)."""
        count = 0
        for name, module in model.named_modules():
            # Match model.layers.X.mlp.gate but NOT model.layers.X.mlp.experts.Y.gate_proj
            if name.endswith(".mlp.gate") and isinstance(module, torch.nn.Linear):
                parts = name.split(".")
                layer_id = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_id = int(parts[i + 1])
                        except ValueError:
                            pass
                if layer_id is not None:
                    h = module.register_forward_hook(self._make_hook(layer_id))
                    self._hooks.append(h)
                    count += 1

        log(f"Attached {count} router hooks (expected {NUM_LAYERS})")
        return count

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        log(f"Saved {len(self.events)} events to {path}")


# ================================================================
# Data Loading
# ================================================================

def load_prompts(dataset_path, num_prompts, seed=42):
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


# ================================================================
# Model Loading & Generation with Trace Collection
# ================================================================

def collect_traces(model_path, dataset_path, num_prompts, max_tokens, trace_path, seed=42):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    log("Loading model with device_map='auto'...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"Model loaded in {time.time() - t0:.0f}s")

    # Print device map summary
    if hasattr(model, 'hf_device_map'):
        devices = set(str(v) for v in model.hf_device_map.values())
        log(f"Device map uses: {devices}")

    collector = RouterTraceCollector(num_experts=NUM_EXPERTS, top_k=TOP_K)
    num_hooks = collector.attach(model)
    if num_hooks == 0:
        log("No router hooks attached! Aborting.", "ERR")
        return

    prompts = load_prompts(dataset_path, num_prompts, seed)
    log(f"Loaded {len(prompts)} prompts")

    model.eval()
    total_tokens_generated = 0

    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": NO_THINK_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]

        log(f"Prompt {i}/{num_prompts}: {len(prompt)} chars, {prompt_len} tokens, "
            f"generating up to {max_tokens}...")

        # Phase 1: Prefill (don't record routing)
        collector._recording = False
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
        del outputs
        token_count = 1
        total_tokens_generated += 1

        # Phase 2: Decode (record routing for each token)
        collector._recording = True
        for step in range(1, max_tokens):
            collector.set_context(f"req_{i}", prompt_len + step)

            with torch.no_grad():
                outputs = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            del outputs
            token_count += 1
            total_tokens_generated += 1

            if next_token.item() == tokenizer.eos_token_id:
                break

        collector._recording = False
        log(f"  Generated {token_count} tokens, trace events so far: {len(collector.events)}")

        del past_key_values
        torch.cuda.empty_cache()

    collector.save(trace_path)
    collector.detach()

    log(f"Total: {total_tokens_generated} tokens across {len(prompts)} prompts")
    log(f"Trace events: {len(collector.events)} ({len(collector.events) // NUM_LAYERS} decode steps x {NUM_LAYERS} layers)")

    del model
    torch.cuda.empty_cache()


# ================================================================
# Analysis
# ================================================================

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def stats(vals):
    if not vals:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    import statistics
    s = sorted(vals)
    n = len(s)
    return {
        "mean": sum(s) / n,
        "median": s[n // 2],
        "std": statistics.stdev(s) if n > 1 else 0,
        "min": s[0],
        "max": s[-1],
        "p25": s[n // 4],
        "p75": s[3 * n // 4],
        "count": n,
    }


def analyze_trace(trace_path):
    """Analyze token-to-token expert routing overlap."""
    log(f"Loading trace from {trace_path}...")
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        log("No trace events!", "ERR")
        return None

    log(f"Loaded {len(events)} trace events")

    # Organize: {(request_id, layer_id): [{token_idx, experts}, ...]}
    by_req_layer = defaultdict(list)
    for e in events:
        by_req_layer[(e["request_id"], e["layer_id"])].append(e)
    for key in by_req_layer:
        by_req_layer[key].sort(key=lambda x: x["token_idx"])

    # --- Analysis 1: Sliding window overlap (simulating K=3 verify batch) ---
    all_pairwise_jaccard = []
    all_consecutive_jaccard = []
    all_union_sizes = []
    all_dedup_ratios = []
    all_raw_overlap_counts = []
    per_layer_jaccard = defaultdict(list)
    per_layer_union = defaultdict(list)
    per_layer_raw_overlap = defaultdict(list)

    window_size = VERIFY_BATCH_SIZE

    for (req_id, layer_id), token_events in by_req_layer.items():
        if len(token_events) < window_size:
            continue

        for start in range(len(token_events) - window_size + 1):
            window = token_events[start:start + window_size]
            expert_sets = [set(e["experts"]) for e in window]

            for ii in range(window_size):
                for jj in range(ii + 1, window_size):
                    jac = jaccard(expert_sets[ii], expert_sets[jj])
                    raw_overlap = len(expert_sets[ii] & expert_sets[jj])
                    all_pairwise_jaccard.append(jac)
                    all_raw_overlap_counts.append(raw_overlap)
                    per_layer_jaccard[layer_id].append(jac)
                    per_layer_raw_overlap[layer_id].append(raw_overlap)

            for ii in range(window_size - 1):
                jac = jaccard(expert_sets[ii], expert_sets[ii + 1])
                all_consecutive_jaccard.append(jac)

            union_all = set()
            for s in expert_sets:
                union_all |= s
            all_union_sizes.append(len(union_all))
            per_layer_union[layer_id].append(len(union_all))

            total_slots = window_size * TOP_K
            dedup_ratio = len(union_all) / total_slots
            all_dedup_ratios.append(dedup_ratio)

    # --- Analysis 2: Adjacent token overlap ---
    all_adjacent_jaccard = []
    all_adjacent_raw = []
    per_layer_adjacent = defaultdict(list)

    for (req_id, layer_id), token_events in by_req_layer.items():
        for ii in range(len(token_events) - 1):
            s1 = set(token_events[ii]["experts"])
            s2 = set(token_events[ii + 1]["experts"])
            jac = jaccard(s1, s2)
            raw = len(s1 & s2)
            all_adjacent_jaccard.append(jac)
            all_adjacent_raw.append(raw)
            per_layer_adjacent[layer_id].append(raw)

    results = {
        "total_events": len(events),
        "window_size": window_size,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,

        "pairwise_jaccard": stats(all_pairwise_jaccard),
        "consecutive_jaccard": stats(all_consecutive_jaccard),
        "union_size": stats(all_union_sizes),
        "dedup_ratio": stats(all_dedup_ratios),
        "raw_overlap_per_pair": stats(all_raw_overlap_counts),

        "adjacent_jaccard": stats(all_adjacent_jaccard),
        "adjacent_raw_overlap": stats(all_adjacent_raw),

        "per_layer": {},
    }

    for layer_id in sorted(per_layer_jaccard.keys()):
        results["per_layer"][layer_id] = {
            "jaccard": stats(per_layer_jaccard[layer_id]),
            "union_size": stats(per_layer_union[layer_id]),
            "raw_overlap": stats(per_layer_raw_overlap[layer_id]),
            "adjacent_raw": stats(per_layer_adjacent.get(layer_id, [])),
        }

    return results


def print_report(results):
    """Print formatted analysis report."""
    print()
    print("=" * 86)
    print("  EXPERT ROUTING OVERLAP ANALYSIS - Qwen3-30B-A3B-Instruct-2507")
    print(f"  {results['num_experts']} experts, top-{results['top_k']}, "
          f"window={results['window_size']} (EAGLE-3 K={results['window_size']-1})")
    print("=" * 86)
    print(f"  Total trace events: {results['total_events']}")
    print()

    pj = results["pairwise_jaccard"]
    ro = results["raw_overlap_per_pair"]
    print("-" * 86)
    print("  1. PAIRWISE OVERLAP (any two tokens in verify window)")
    print("-" * 86)
    print(f"     Jaccard:  mean={pj['mean']:.4f} ({pj['mean']*100:.1f}%), "
          f"median={pj['median']:.4f}, std={pj['std']:.4f}")
    print(f"     Range:    [{pj['min']:.4f}, {pj['max']:.4f}], "
          f"P25-P75=[{pj['p25']:.4f}, {pj['p75']:.4f}]")
    print(f"     Raw shared experts: mean={ro['mean']:.2f}/{TOP_K}, "
          f"median={ro['median']:.1f}, max={ro['max']:.0f}")
    print(f"     ({pj['count']} pairs measured)")
    print()

    aj = results["adjacent_jaccard"]
    ar = results["adjacent_raw_overlap"]
    print("-" * 86)
    print("  2. ADJACENT TOKEN OVERLAP (token[i] vs token[i+1])")
    print("-" * 86)
    print(f"     Jaccard:  mean={aj['mean']:.4f} ({aj['mean']*100:.1f}%), "
          f"median={aj['median']:.4f}, std={aj['std']:.4f}")
    print(f"     Raw shared: mean={ar['mean']:.2f}/{TOP_K}, "
          f"median={ar['median']:.1f}")
    print()

    us = results["union_size"]
    max_slots = results['window_size'] * TOP_K
    print("-" * 86)
    print(f"  3. UNION SIZE (unique experts per {results['window_size']}-token verify batch)")
    print(f"     Theoretical: [{TOP_K}, {max_slots}] "
          f"(all same -> {TOP_K}, all different -> {max_slots})")
    print("-" * 86)
    print(f"     Mean:   {us['mean']:.1f} / {max_slots} slots "
          f"({us['mean']/max_slots*100:.1f}% unique)")
    print(f"     Median: {us['median']:.1f}, Range: [{us['min']:.0f}, {us['max']:.0f}]")
    print(f"     P25-P75: [{us['p25']:.0f}, {us['p75']:.0f}]")
    print()

    dr = results["dedup_ratio"]
    redundancy = (1 - dr['mean']) * 100
    print("-" * 86)
    print("  4. DEDUPLICATION POTENTIAL")
    print("-" * 86)
    print(f"     Unique/Total ratio: mean={dr['mean']:.4f}")
    print(f"     REDUNDANCY: {redundancy:.1f}% of expert computations are "
          f"duplicated across tokens in verify batch")
    print(f"     -> Expert dedup can save ~{redundancy:.0f}% of fused_moe "
          f"compute in verify phase")
    print()

    if results["per_layer"]:
        print("-" * 86)
        print("  5. PER-LAYER BREAKDOWN")
        print("-" * 86)

        layer_data = []
        for lid in sorted(results["per_layer"].keys()):
            d = results["per_layer"][lid]
            layer_data.append((
                lid,
                d["jaccard"]["mean"],
                d["raw_overlap"]["mean"],
                d["union_size"]["mean"],
                d.get("adjacent_raw", {}).get("mean", 0),
            ))

        print(f"  {'Layer':>6} {'Jaccard':>10} {'SharedExp':>10} "
              f"{'UnionSize':>10} {'AdjShared':>10}")
        print(f"  {'------':>6} {'----------':>10} {'----------':>10} "
              f"{'----------':>10} {'----------':>10}")
        for lid, jac, shared, union_sz, adj in layer_data:
            print(f"  {lid:6d} {jac:10.4f} {shared:10.2f} "
                  f"{union_sz:10.1f} {adj:10.2f}")

        print()
        layer_data_sorted = sorted(layer_data, key=lambda x: x[1], reverse=True)
        print("  Top-5 MOST overlapping layers:")
        for lid, jac, shared, union_sz, _ in layer_data_sorted[:5]:
            print(f"    Layer {lid:2d}: Jaccard={jac:.4f} ({jac*100:.1f}%), "
                  f"shared={shared:.1f}, union={union_sz:.0f}/{max_slots}")
        print("  Top-5 LEAST overlapping layers:")
        for lid, jac, shared, union_sz, _ in layer_data_sorted[-5:]:
            print(f"    Layer {lid:2d}: Jaccard={jac:.4f} ({jac*100:.1f}%), "
                  f"shared={shared:.1f}, union={union_sz:.0f}/{max_slots}")
        print()

    print("=" * 86)
    print("  SUMMARY")
    print("=" * 86)
    print(f"  * {results['window_size']} tokens in each verify batch "
          f"select top-{TOP_K} from {NUM_EXPERTS} experts")
    print(f"  * Average {us['mean']:.1f} unique experts needed "
          f"(vs {max_slots} total dispatch slots)")
    print(f"  * {redundancy:.1f}% of expert computations are REDUNDANT")
    print(f"  * Each token pair shares {ro['mean']:.1f} experts "
          f"on average (Jaccard={pj['mean']:.3f})")
    if redundancy > 20:
        print(f"  -> HIGH OVERLAP: Expert deduplication is valuable!")
    elif redundancy > 10:
        print(f"  -> MODERATE OVERLAP: Some benefit from deduplication")
    else:
        print(f"  -> LOW OVERLAP: Limited deduplication potential")
    print("=" * 86)
    print()


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure expert routing overlap")
    parser.add_argument("--model", type=str,
                        default="/root/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--dataset", type=str,
                        default="/root/MoE-SD/data/combined_sharegpt.json")
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trace-path", type=str,
                        default="/tmp/routing_trace.jsonl")
    parser.add_argument("--output", type=str,
                        default="/root/MoE-SD/results/routing_overlap_analysis.json")
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Skip model loading, just analyze existing trace")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.analyze_only:
        results = analyze_trace(args.analyze_only)
    else:
        collect_traces(
            model_path=args.model,
            dataset_path=args.dataset,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            trace_path=args.trace_path,
            seed=args.seed,
        )
        results = analyze_trace(args.trace_path)

    if results:
        print_report(results)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        log(f"Analysis saved to {out_path}")


if __name__ == "__main__":
    main()
