#!/usr/bin/env python3
"""
Dataset Downloader for Public Benchmarks

Downloads and preprocesses public datasets for vLLM benchmark evaluation.
Supports:
- ShareGPT (via HuggingFace datasets)
- Alpaca
- Dolly
- Custom JSONL files
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' (HuggingFace) not installed. Install with: pip install datasets")

DATA_DIR = Path(__file__).parent.parent / "data"


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def estimate_token_count(text: str) -> int:
    """Estimate token count from text (~4 chars/token for English, ~1.5 for Chinese)."""
    char_count = len(text)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    if chinese_chars > char_count * 0.5:
        return max(1, int(char_count / 1.5))
    else:
        return max(1, int(char_count / 4))


def download_sharegpt_subset(output_path: Path, max_samples: int = 1000, 
                             min_prompt_len: int = 50, max_prompt_len: int = 2048,
                             min_completion_len: int = 50, max_completion_len: int = 1024,
                             seed: int = 42):
    """Download a subset of ShareGPT dataset."""
    if not DATASETS_AVAILABLE:
        print("Error: HuggingFace 'datasets' library required.")
        print("Install with: pip install datasets")
        sys.exit(1)
    
    print(f"Downloading ShareGPT dataset (subset: {max_samples} samples)...")
    
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", 
                               split="train", trust_remote_code=True)
        
        print(f"Loaded {len(dataset)} samples from ShareGPT")
        
        processed = []
        seen_hashes = set()
        
        for item in dataset:
            try:
                conv = item.get("conversations", [])
                if len(conv) < 2:
                    continue
                
                prompt = conv[0].get("value", "") if isinstance(conv[0], dict) else str(conv[0])
                completion = conv[1].get("value", "") if isinstance(conv[1], dict) else str(conv[1])
                
                prompt_len = len(prompt)
                completion_len = len(completion)
                
                if not (min_prompt_len <= prompt_len <= max_prompt_len):
                    continue
                if not (min_completion_len <= completion_len <= max_completion_len):
                    continue
                
                hash_key = compute_hash(prompt + completion)
                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)
                
                processed.append({
                    "prompt": prompt.strip(),
                    "completion": completion.strip(),
                    "source": "ShareGPT",
                    "length_input": estimate_token_count(prompt),
                    "length_output": estimate_token_count(completion),
                    "original_length_prompt": prompt_len,
                    "original_length_completion": completion_len
                })
                
                if len(processed) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Warning: Skipping sample due to error: {e}")
                continue
        
        print(f"Processed {len(processed)} valid samples")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved to {output_path}")
        print(f"Statistics:")
        print(f"  - Samples: {len(processed)}")
        if processed:
            avg_prompt = sum(p['length_input'] for p in processed) / len(processed)
            avg_comp = sum(p['length_output'] for p in processed) / len(processed)
            print(f"  - Avg input tokens: {avg_prompt:.1f}")
            print(f"  - Avg output tokens: {avg_comp:.1f}")
        
    except Exception as e:
        print(f"Error downloading ShareGPT: {e}")
        sys.exit(1)


def download_alpaca_subset(output_path: Path, max_samples: int = 1000,
                           min_prompt_len: int = 20, max_prompt_len: int = 1024,
                           min_completion_len: int = 20, max_completion_len: int = 512,
                           seed: int = 42):
    """Download Stanford Alpaca dataset subset."""
    if not DATASETS_AVAILABLE:
        print("Error: HuggingFace 'datasets' library required.")
        sys.exit(1)
    
    print(f"Downloading Alpaca dataset (subset: {max_samples} samples)...")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        print(f"Loaded {len(dataset)} samples from Alpaca")
        
        processed = []
        seen_hashes = set()
        
        for item in dataset:
            try:
                instruction = item.get("instruction", "").strip()
                input_text = item.get("input", "").strip()
                output_text = item.get("output", "").strip()
                
                if input_text:
                    prompt = f"{instruction}\n\nInput:\n{input_text}"
                else:
                    prompt = instruction
                
                completion = output_text
                
                prompt_len = len(prompt)
                completion_len = len(completion)
                
                if not (min_prompt_len <= prompt_len <= max_prompt_len):
                    continue
                if not (min_completion_len <= completion_len <= max_completion_len):
                    continue
                
                hash_key = compute_hash(prompt + completion)
                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)
                
                processed.append({
                    "prompt": prompt,
                    "completion": completion,
                    "source": "Alpaca",
                    "length_input": estimate_token_count(prompt),
                    "length_output": estimate_token_count(completion),
                    "original_length_prompt": prompt_len,
                    "original_length_completion": completion_len
                })
                
                if len(processed) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Warning: Skipping sample due to error: {e}")
                continue
        
        print(f"Processed {len(processed)} valid samples")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error downloading Alpaca: {e}")
        sys.exit(1)


def download_dolly_subset(output_path: Path, max_samples: int = 1000,
                          min_prompt_len: int = 20, max_prompt_len: int = 1024,
                          min_completion_len: int = 20, max_completion_len: int = 512,
                          seed: int = 42):
    """Download Dolly-15k dataset subset."""
    if not DATASETS_AVAILABLE:
        print("Error: HuggingFace 'datasets' library required.")
        sys.exit(1)
    
    print(f"Downloading Dolly-15k dataset (subset: {max_samples} samples)...")
    
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
        print(f"Loaded {len(dataset)} samples from Dolly-15k")
        
        processed = []
        seen_hashes = set()
        
        for item in dataset:
            try:
                instruction = item.get("instruction", "").strip()
                context = item.get("context", "").strip()
                response = item.get("response", "").strip()
                
                if context:
                    prompt = f"{instruction}\n\nContext:\n{context}"
                else:
                    prompt = instruction
                
                completion = response
                
                prompt_len = len(prompt)
                completion_len = len(completion)
                
                if not (min_prompt_len <= prompt_len <= max_prompt_len):
                    continue
                if not (min_completion_len <= completion_len <= max_completion_len):
                    continue
                
                hash_key = compute_hash(prompt + completion)
                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)
                
                processed.append({
                    "prompt": prompt,
                    "completion": completion,
                    "source": "Dolly-15k",
                    "length_input": estimate_token_count(prompt),
                    "length_output": estimate_token_count(completion),
                    "original_length_prompt": prompt_len,
                    "original_length_completion": completion_len
                })
                
                if len(processed) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Warning: Skipping sample due to error: {e}")
                continue
        
        print(f"Processed {len(processed)} valid samples")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error downloading Dolly: {e}")
        sys.exit(1)


def create_custom_subset(input_path: Path, output_path: Path, max_samples: int = 1000,
                         prompt_field: str = "prompt", completion_field: str = "completion",
                         source_name: str = "custom"):
    """Create a subset from a custom JSONL file."""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing custom dataset from {input_path}...")
    
    processed = []
    seen_hashes = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if len(processed) >= max_samples:
                break
            
            try:
                item = json.loads(line.strip())
                
                prompt = item.get(prompt_field, "")
                completion = item.get(completion_field, "")
                
                if not prompt or not completion:
                    print(f"Warning: Line {line_num} missing prompt or completion")
                    continue
                
                hash_key = compute_hash(str(prompt) + str(completion))
                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)
                
                processed.append({
                    "prompt": str(prompt).strip(),
                    "completion": str(completion).strip(),
                    "source": source_name,
                    "length_input": estimate_token_count(str(prompt)),
                    "length_output": estimate_token_count(str(completion)),
                    "original_length_prompt": len(str(prompt)),
                    "original_length_completion": len(str(completion))
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}")
                continue
    
    print(f"Processed {len(processed)} valid samples")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download public datasets for benchmarking")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sharegpt", "alpaca", "dolly", "custom"],
                        help="Dataset to download")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: data/{dataset}_subset.jsonl)")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples to extract")
    parser.add_argument("--min-prompt-len", type=int, default=50,
                        help="Minimum prompt length (characters)")
    parser.add_argument("--max-prompt-len", type=int, default=2048,
                        help="Maximum prompt length (characters)")
    parser.add_argument("--min-completion-len", type=int, default=50,
                        help="Minimum completion length (characters)")
    parser.add_argument("--max-completion-len", type=int, default=1024,
                        help="Maximum completion length (characters)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file for custom dataset")
    parser.add_argument("--prompt-field", type=str, default="prompt",
                        help="Field name for prompt in custom dataset")
    parser.add_argument("--completion-field", type=str, default="completion",
                        help="Field name for completion in custom dataset")
    parser.add_argument("--source-name", type=str, default="custom",
                        help="Source name for custom dataset")
    
    args = parser.parse_args()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DATA_DIR / f"{args.dataset}_subset.jsonl"
    
    if args.dataset == "sharegpt":
        download_sharegpt_subset(
            output_path=output_path,
            max_samples=args.max_samples,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            min_completion_len=args.min_completion_len,
            max_completion_len=args.max_completion_len,
            seed=args.seed
        )
    elif args.dataset == "alpaca":
        download_alpaca_subset(
            output_path=output_path,
            max_samples=args.max_samples,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            min_completion_len=args.min_completion_len,
            max_completion_len=args.max_completion_len,
            seed=args.seed
        )
    elif args.dataset == "dolly":
        download_dolly_subset(
            output_path=output_path,
            max_samples=args.max_samples,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            min_completion_len=args.min_completion_len,
            max_completion_len=args.max_completion_len,
            seed=args.seed
        )
    elif args.dataset == "custom":
        if not args.input:
            print("Error: --input required for custom dataset")
            sys.exit(1)
        create_custom_subset(
            input_path=Path(args.input),
            output_path=output_path,
            max_samples=args.max_samples,
            prompt_field=args.prompt_field,
            completion_field=args.completion_field,
            source_name=args.source_name
        )
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main()
