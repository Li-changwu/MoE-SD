"""
Expert Routing Trace Collector — Hook into MoE Router Logits
=============================================================
Captures per-token, per-layer expert routing decisions during inference.
Supports both offline (transformers) and online (vLLM hook) collection.

Output format (JSONL):
  {
    "request_id": str,
    "token_idx": int,
    "layer_id": int,
    "experts": [int × top_k],
    "router_probs": [float × top_k],
    "phase": "prefill" | "decode"
  }

Usage:
  # Offline collection with transformers
  python -m collectors.expert_trace_hook \
      --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
      --dataset data/combined_sharegpt.json \
      --output results/moe_trace/expert_routing_trace.jsonl \
      --max-tokens 128 --num-prompts 20
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    request_id: str
    token_idx: int
    layer_id: int
    experts: list[int]
    router_probs: list[float]
    phase: str = "decode"

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "token_idx": self.token_idx,
            "layer_id": self.layer_id,
            "experts": self.experts,
            "router_probs": [round(p, 6) for p in self.router_probs],
            "phase": self.phase,
        }


class ExpertTraceCollector:
    """Collects MoE expert routing decisions via forward hooks."""

    def __init__(self, output_path: Optional[str] = None):
        self.events: list[TraceEvent] = []
        self.output_path = Path(output_path) if output_path else None
        self._hooks = []
        self._current_request_id = "default"
        self._current_token_idx = 0
        self._file_handle = None

    def _open_file(self):
        if self.output_path and self._file_handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.output_path, "w", encoding="utf-8")

    def _write_event(self, event: TraceEvent):
        self.events.append(event)
        if self._file_handle:
            self._file_handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self._file_handle.flush()

    def set_context(self, request_id: str, token_idx: int, phase: str = "decode"):
        self._current_request_id = request_id
        self._current_token_idx = token_idx
        self._current_phase = phase

    def _make_hook(self, layer_id: int):
        """Create a forward hook for a specific MoE layer's router."""

        def hook_fn(module, input_args, output):
            # Router output is logits of shape [batch, num_experts]
            # We need to extract top-k expert indices and their probabilities
            if isinstance(output, tuple):
                router_logits = output[0]
            else:
                router_logits = output

            if router_logits.dim() == 2:
                # [batch_size, num_experts]
                probs = torch.softmax(router_logits.float(), dim=-1)
                top_k = min(8, probs.shape[-1])
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

                for b in range(router_logits.shape[0]):
                    event = TraceEvent(
                        request_id=self._current_request_id,
                        token_idx=self._current_token_idx + b,
                        layer_id=layer_id,
                        experts=top_indices[b].cpu().tolist(),
                        router_probs=top_probs[b].cpu().tolist(),
                        phase=getattr(self, "_current_phase", "decode"),
                    )
                    self._write_event(event)

        return hook_fn

    def attach_to_model(self, model):
        """Attach hooks to all MoE router layers in a Qwen3MoE model."""
        self._open_file()
        hook_count = 0

        for name, module in model.named_modules():
            # Qwen3MoE: each decoder layer has .mlp.gate (the router)
            if hasattr(module, "gate") and "mlp" in name:
                # Extract layer index from name like "model.layers.0.mlp"
                parts = name.split(".")
                layer_id = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_id = int(parts[i + 1])
                        except ValueError:
                            pass
                if layer_id is not None:
                    hook = module.gate.register_forward_hook(self._make_hook(layer_id))
                    self._hooks.append(hook)
                    hook_count += 1

        logger.info(f"Attached {hook_count} router hooks")
        return hook_count

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def get_events_df(self):
        """Return events as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([e.to_dict() for e in self.events])


@dataclass
class MAFResult:
    """Result of MAF computation for a given K."""
    K: int
    mean_maf: float
    std_maf: float
    p25_maf: float
    p75_maf: float
    per_layer_maf: dict = field(default_factory=dict)  # layer_id -> mean union size
    num_windows: int = 0


def compute_maf_from_trace(events: list[dict], K: int, num_experts_per_tok: int = 8) -> MAFResult:
    """
    Compute MAF(K) from expert routing trace data.

    MAF(K) = E[|∪_{i=0}^{K} E_i^(l)|] / k

    where E_i^(l) is the set of top-k experts for token i at layer l,
    and the expectation is over sliding windows of K+1 consecutive tokens.
    """
    import numpy as np

    # Group events by (request_id, layer_id) and sort by token_idx
    from collections import defaultdict
    grouped = defaultdict(list)
    for e in events:
        if e.get("phase", "decode") == "decode":  # Only decode tokens
            key = (e["request_id"], e["layer_id"])
            grouped[key].append(e)

    all_mafs = []
    per_layer_unions = defaultdict(list)

    for (req_id, layer_id), layer_events in grouped.items():
        layer_events.sort(key=lambda x: x["token_idx"])

        if len(layer_events) < K + 1:
            continue

        # Sliding window of K+1 consecutive tokens
        for start in range(len(layer_events) - K):
            window = layer_events[start: start + K + 1]
            expert_union = set()
            for evt in window:
                expert_union.update(evt["experts"])

            union_size = len(expert_union)
            maf = union_size / num_experts_per_tok
            all_mafs.append(maf)
            per_layer_unions[layer_id].append(union_size)

    if not all_mafs:
        return MAFResult(K=K, mean_maf=0, std_maf=0, p25_maf=0, p75_maf=0, num_windows=0)

    arr = np.array(all_mafs)
    per_layer_maf = {
        lid: float(np.mean(sizes))
        for lid, sizes in sorted(per_layer_unions.items())
    }

    return MAFResult(
        K=K,
        mean_maf=float(np.mean(arr)),
        std_maf=float(np.std(arr)),
        p25_maf=float(np.percentile(arr, 25)),
        p75_maf=float(np.percentile(arr, 75)),
        per_layer_maf=per_layer_maf,
        num_windows=len(all_mafs),
    )


def compute_mmaf_per_token(events: list[dict], num_experts_per_tok: int = 8) -> list[dict]:
    """
    Compute marginal MAF (mMAF) per token: the number of unique experts
    token t_j adds to the union beyond what previous tokens contributed.

    mMAF(t_j) = |E_j \\ ∪_{i<j} E_i| / k
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for e in events:
        if e.get("phase", "decode") == "decode":
            key = (e["request_id"], e["layer_id"])
            grouped[key].append(e)

    results = []
    for (req_id, layer_id), layer_events in grouped.items():
        layer_events.sort(key=lambda x: x["token_idx"])
        running_union = set()

        for evt in layer_events:
            current_experts = set(evt["experts"])
            marginal = current_experts - running_union
            mmaf = len(marginal) / num_experts_per_tok

            results.append({
                "request_id": req_id,
                "layer_id": layer_id,
                "token_idx": evt["token_idx"],
                "mmaf": round(mmaf, 4),
                "marginal_count": len(marginal),
                "union_size": len(running_union | current_experts),
            })
            running_union |= current_experts

    return results


def compute_theoretical_maf(K: int, k: int = 8, N: int = 128) -> float:
    """
    Theoretical MAF under i.i.d. uniform random routing.

    MAF_random(K) = N * (1 - (1 - k/N)^(K+1)) / k

    For Qwen3-30B: N=128, k=8
    """
    return N * (1 - (1 - k / N) ** (K + 1)) / k


def main():
    parser = argparse.ArgumentParser(description="Expert Trace Collection (Offline)")
    parser.add_argument("--model", required=True, help="Path to Qwen3-30B model")
    parser.add_argument("--dataset", required=True, help="Path to dataset (ShareGPT JSON or JSONL)")
    parser.add_argument("--output", default="results/moe_trace/expert_routing_trace.jsonl")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load dataset
    dataset_path = Path(args.dataset)
    if dataset_path.suffix == ".json":
        with open(dataset_path) as f:
            data = json.load(f)
        prompts = [d["conversations"][0]["value"] for d in data[:args.num_prompts]]
    elif dataset_path.suffix == ".jsonl":
        prompts = []
        with open(dataset_path) as f:
            for line in f:
                if len(prompts) >= args.num_prompts:
                    break
                item = json.loads(line.strip())
                prompts.append(item.get("prompt", item.get("text", "")))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    logger.info(f"Loaded {len(prompts)} prompts from {dataset_path}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    # Attach hooks
    collector = ExpertTraceCollector(output_path=args.output)
    num_hooks = collector.attach_to_model(model)
    logger.info(f"Attached {num_hooks} hooks to MoE router layers")

    if num_hooks == 0:
        logger.error("No MoE router hooks attached! Check model architecture.")
        # Try alternative hook pattern
        for name, module in model.named_modules():
            logger.info(f"  Module: {name} -> {type(module).__name__}")
        sys.exit(1)

    # Run inference and collect traces
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:80]}...")
        collector.set_context(request_id=f"req_{i}", token_idx=0, phase="prefill")

        chat_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        prefill_len = chat_input.shape[1]

        with torch.no_grad():
            # Prefill
            collector.set_context(request_id=f"req_{i}", token_idx=0, phase="prefill")
            outputs = model.generate(
                chat_input,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=1.0,
            )

        num_generated = outputs.shape[1] - prefill_len
        logger.info(f"  Generated {num_generated} tokens")

    collector.detach()
    logger.info(f"Collected {len(collector.events)} trace events")
    logger.info(f"Trace saved to {args.output}")

    # Compute MAF for different K values
    events_dicts = [e.to_dict() for e in collector.events]
    output_dir = Path(args.output).parent

    logger.info("Computing MAF statistics...")
    maf_rows = []
    for K in range(1, 6):
        result = compute_maf_from_trace(events_dicts, K=K)
        theoretical = compute_theoretical_maf(K)
        maf_rows.append({
            "K": K,
            "mean_MAF": round(result.mean_maf, 4),
            "std_MAF": round(result.std_maf, 4),
            "p25_MAF": round(result.p25_maf, 4),
            "p75_MAF": round(result.p75_maf, 4),
            "theoretical_MAF": round(theoretical, 4),
            "num_windows": result.num_windows,
        })
        logger.info(f"  MAF(K={K}): measured={result.mean_maf:.4f}, "
                     f"theoretical={theoretical:.4f}, windows={result.num_windows}")

    maf_path = output_dir / "maf_by_k.csv"
    import csv
    with open(maf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=maf_rows[0].keys())
        writer.writeheader()
        writer.writerows(maf_rows)
    logger.info(f"MAF results saved to {maf_path}")

    # Compute mMAF
    mmaf_results = compute_mmaf_per_token(events_dicts)
    mmaf_path = output_dir / "mmaf_distribution.csv"
    with open(mmaf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mmaf_results[0].keys())
        writer.writeheader()
        writer.writerows(mmaf_results)
    logger.info(f"mMAF distribution saved to {mmaf_path}")

    # Per-layer MAF for K=2
    result_k2 = compute_maf_from_trace(events_dicts, K=2)
    per_layer_path = output_dir / "maf_per_layer.csv"
    with open(per_layer_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "K", "mean_union_size"])
        writer.writeheader()
        for layer_id, mean_size in sorted(result_k2.per_layer_maf.items()):
            writer.writerow({"layer": layer_id, "K": 2, "mean_union_size": round(mean_size, 2)})
    logger.info(f"Per-layer MAF saved to {per_layer_path}")


if __name__ == "__main__":
    main()
