import argparse
import datetime as dt
import json
import subprocess
import tempfile
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_config(path: Path) -> dict:
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read yaml configs")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported config extension: {path}")


def run_cmd(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def load_dataset_file(dataset_path: Path) -> list[dict]:
    """
    Load prompts from a dataset file (JSONL format).
    
    Expected format per line:
    {
        "prompt": str,
        "completion": str (optional, for reference),
        "length_input": int (optional, estimated tokens),
        "length_output": int (optional, estimated tokens)
    }
    
    Returns list of prompt dictionaries.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    prompts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                item = json.loads(line)
                
                if isinstance(item, dict):
                    if 'prompt' in item:
                        prompts.append(item)
                    elif 'text' in item:
                        prompts.append({'prompt': item['text'], **item})
                    else:
                        print(f"Warning: Line {line_num} missing 'prompt' field, skipping")
                elif isinstance(item, str):
                    prompts.append({'prompt': item})
                else:
                    print(f"Warning: Line {line_num} unexpected format, skipping")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}, skipping")
    
    if not prompts:
        raise ValueError(f"No valid prompts found in {dataset_path}")
    
    print(f"Loaded {len(prompts)} prompts from {dataset_path}")
    return prompts


def result_dir(root: Path, method: str, profile: str, mode: str, cfg_hash: str) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / method / profile / mode / f"{stamp}_{cfg_hash}"


def short_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    import hashlib
    return hashlib.sha1(raw).hexdigest()[:10]


def run_bench(config: dict):
    model = config["model"]
    base_url = config.get("base_url", "http://127.0.0.1:8000/v1")
    endpoint = config.get("endpoint", "/chat/completions")
    result_root = Path(config.get("result_root", "results/raw"))
    method = config["method"]
    profile = config["workload_profile"]

    meta = {
        "model": model,
        "method": method,
        "workload_profile": profile,
        "seed": config.get("seed", 42),
        "git_commit": git_commit(),
    }

    common = {
        "num_prompts": int(config.get("num_prompts", 32)),
        "prompt_len": int(config.get("prompt_len", 512)),
        "output_len": int(config.get("output_len", 128)),
        "request_rate": float(config.get("request_rate", 2.0)),
    }
    
    # Check if using dataset file
    dataset_file = config.get("dataset_file")
    if dataset_file:
        print(f"Using dataset: {dataset_file}")
        profile = f"{method}_dataset_{Path(dataset_file).stem}"

    engine = {
        "max_model_len": config.get("max_model_len"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "cpu_offload_gb": config.get("cpu_offload_gb"),
        "swap_space": config.get("swap_space"),
    }

    cfg_hash = short_hash({"meta": meta, "common": common, "dataset_file": dataset_file})

    for mode in config.get("modes", ["serve", "latency", "throughput"]):
        out_dir = result_dir(result_root, method, profile, mode, cfg_hash)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{mode}.json"

        cmd = ["vllm", "bench", mode, "--model", model]

        if mode == "serve":
            cmd.extend(["--save-result", "--result-dir", str(out_dir)])
        else:
            cmd.extend(["--output-json", str(out_json)])

        # Handle dataset-based or synthetic workload
        if dataset_file:
            # Load prompts from dataset file
            dataset_path = Path(dataset_file)
            if not dataset_path.is_absolute():
                dataset_path = Path(__file__).parent.parent / dataset_path
            
            prompts = load_dataset_file(dataset_path)
            
            # Limit to num_prompts if specified
            num_prompts = common["num_prompts"]
            if len(prompts) > num_prompts:
                prompts = prompts[:num_prompts]
            
            # Create temporary file with selected prompts
            temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
            with open(temp_file, 'w', encoding='utf-8') as f:
                for prompt_item in prompts:
                    f.write(json.dumps(prompt_item, ensure_ascii=False) + '\n')
            
            print(f"Using {len(prompts)} prompts from dataset (temp: {temp_file})")
            
            # vLLM bench serve supports --dataset parameter
            if mode == "serve":
                cmd.extend([
                    "--backend",
                    "openai-chat",
                    "--base-url",
                    base_url,
                    "--endpoint",
                    endpoint,
                    "--dataset-name", "custom",
                    "--dataset-path",
                    str(temp_file),
                    "--request-rate",
                    str(common["request_rate"]),
                ])
            elif mode == "throughput":
                cmd.extend([
                    "--dataset-path",
                    str(temp_file),
                ])
            # Note: latency mode doesn't support dataset, falls back to synthetic
            else:
                print(f"Warning: {mode} mode doesn't support dataset, using synthetic")
                cmd.extend([
                    "--input-len",
                    str(common["prompt_len"]),
                    "--output-len",
                    str(common["output_len"]),
                ])
        else:
            # Synthetic random workload (legacy behavior)
            if mode == "serve":
                cmd.extend([
                    "--backend",
                    "openai-chat",
                    "--base-url",
                    base_url,
                    "--endpoint",
                    endpoint,
                    "--num-prompts",
                    str(common["num_prompts"]),
                    "--random-input-len",
                    str(common["prompt_len"]),
                    "--random-output-len",
                    str(common["output_len"]),
                    "--request-rate",
                    str(common["request_rate"]),
                    "--seed",
                    str(meta["seed"]),
                ])
            elif mode == "latency":
                cmd.extend([
                    "--input-len",
                    str(common["prompt_len"]),
                    "--output-len",
                    str(common["output_len"]),
                    "--num-iters",
                    str(int(config.get("latency_num_iters", 3))),
                    "--num-iters-warmup",
                    str(int(config.get("latency_num_iters_warmup", 1))),
                ])
            elif mode == "throughput":
                cmd.extend([
                    "--num-prompts",
                    str(common["num_prompts"]),
                    "--random-input-len",
                    str(common["prompt_len"]),
                    "--random-output-len",
                    str(common["output_len"]),
                    "--seed",
                    str(meta["seed"]),
                ])
            else:
                raise ValueError(f"unsupported mode: {mode}")

        if mode in {"latency", "throughput"}:
            if engine["max_model_len"] is not None:
                cmd.extend(["--max-model-len", str(engine["max_model_len"])])
            if engine["gpu_memory_utilization"] is not None:
                cmd.extend(["--gpu-memory-utilization", str(engine["gpu_memory_utilization"])])
            if engine["cpu_offload_gb"] is not None:
                cmd.extend(["--cpu-offload-gb", str(engine["cpu_offload_gb"])])
            if engine["swap_space"] is not None:
                cmd.extend(["--swap-space", str(engine["swap_space"])])

        if "speculative_config" in config and config["speculative_config"]:
            cmd.extend(["--speculative-config", json.dumps(config["speculative_config"])])
        
        run_cmd(cmd)

        metadata_path = out_dir / "metadata.json"
        metadata = {
            **meta,
            **common,
            **engine,
            "mode": mode,
            "config_hash": cfg_hash,
            "speculative_config": config.get("speculative_config"),
            "dataset_file": dataset_file,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="yaml/json benchmark config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_bench(config)


if __name__ == "__main__":
    main()
