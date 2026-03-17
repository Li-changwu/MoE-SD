import argparse
import datetime as dt
import json
import subprocess
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
    engine = {
        "max_model_len": config.get("max_model_len"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "cpu_offload_gb": config.get("cpu_offload_gb"),
        "swap_space": config.get("swap_space"),
    }

    cfg_hash = short_hash({"meta": meta, "common": common})

    for mode in config.get("modes", ["serve", "latency", "throughput"]):
        out_dir = result_dir(result_root, method, profile, mode, cfg_hash)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{mode}.json"

        cmd = ["vllm", "bench", mode, "--model", model]

        if mode == "serve":
            cmd.extend(["--save-result", "--result-dir", str(out_dir)])
        else:
            cmd.extend(["--output-json", str(out_json)])

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
