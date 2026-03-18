#!/usr/bin/env python3
"""
Experiment 0: Expert Routing Trace Collection
==============================================
前置实验 — 采集 Qwen3-30B 真实 Expert 路由数据。
所有后续验证实验均依赖此 trace 数据。

采集内容:
  - 每个 token 在每层的 top-8 expert indices + router probabilities
  - 区分 prefill / decode phase
  - 记录 token 生成顺序

运行方式:
  python scripts/validation/exp0_collect_expert_trace.py \
      --model /home/sage3/models/Qwen3-30B-A3B-Instruct-2507 \
      --dataset data/combined_sharegpt.json \
      --num-prompts 50 \
      --max-tokens 128 \
      --output results/validation/expert_trace.jsonl

预计耗时: ~10-15 min (50 prompts × 128 tokens, single A6000)
预计输出: ~50 × 128 × 48 ≈ 307,200 trace events (~60-100 MB JSONL)

对应 Issue: #31
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


def load_prompts(dataset_path: str, num_prompts: int) -> list[str]:
    """Load prompts from ShareGPT JSON or JSONL format."""
    path = Path(dataset_path)
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        prompts = []
        for d in data[:num_prompts]:
            if "conversations" in d:
                # ShareGPT format
                for turn in d["conversations"]:
                    if turn.get("from") in ("human", "user"):
                        prompts.append(turn["value"])
                        break
            elif "prompt" in d:
                prompts.append(d["prompt"])
        return prompts[:num_prompts]
    elif path.suffix == ".jsonl":
        prompts = []
        with open(path) as f:
            for line in f:
                if len(prompts) >= num_prompts:
                    break
                item = json.loads(line.strip())
                prompts.append(item.get("prompt", item.get("text", "")))
        return prompts
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def collect_trace(args):
    """Main trace collection pipeline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collectors.expert_trace_hook import ExpertTraceCollector

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    prompts = load_prompts(args.dataset, args.num_prompts)
    logger.info(f"Loaded {len(prompts)} prompts from {args.dataset}")

    # Load model
    logger.info(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully")

    # Print model MoE info
    config = model.config
    logger.info(f"Model config: num_experts={config.num_experts}, "
                f"top_k={config.num_experts_per_tok}, "
                f"num_layers={config.num_hidden_layers}")

    # Attach hooks
    collector = ExpertTraceCollector(output_path=args.output)
    num_hooks = collector.attach_to_model(model)
    logger.info(f"Attached {num_hooks} router hooks")

    if num_hooks == 0:
        logger.error("No MoE router hooks attached! Dumping module names...")
        for name, module in model.named_modules():
            if "gate" in name or "router" in name:
                logger.info(f"  Candidate: {name} -> {type(module).__name__}")
        sys.exit(1)

    # Inference loop
    total_tokens = 0
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")

        # Prepare input
        chat_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        prefill_len = chat_input.shape[1]
        collector.set_context(request_id=f"req_{i:04d}", token_idx=0, phase="prefill")

        with torch.no_grad():
            outputs = model.generate(
                chat_input,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=1.0,
            )

        num_generated = outputs.shape[1] - prefill_len
        total_tokens += num_generated
        elapsed = time.time() - t0
        logger.info(f"  Generated {num_generated} tokens "
                     f"(total: {total_tokens}, {total_tokens/elapsed:.1f} tok/s)")

    collector.detach()
    elapsed = time.time() - t0
    logger.info(f"Collection complete: {len(collector.events)} events, "
                f"{total_tokens} tokens, {elapsed:.1f}s")

    # Summary statistics
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "total_generated_tokens": total_tokens,
        "total_events": len(collector.events),
        "elapsed_seconds": round(elapsed, 1),
        "trace_file": args.output,
    }

    summary_path = output_dir / "trace_collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp0: Expert Trace Collection")
    parser.add_argument("--model", default="/home/sage3/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--dataset", default="data/combined_sharegpt.json")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="results/validation/expert_trace.jsonl")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output).parent / "exp0_trace.log"),
        ],
    )

    collect_trace(args)


if __name__ == "__main__":
    main()
