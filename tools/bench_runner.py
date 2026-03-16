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

    cfg_hash = short_hash({"meta": meta, "common": common})

    for mode in config.get("modes", ["serve", "latency", "throughput"]):
        out_dir = result_dir(result_root, method, profile, mode, cfg_hash)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["vllm", "bench", mode, "--model", model, "--save-result", "--result-dir", str(out_dir)]

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

        if "speculative_config" in config and config["speculative_config"]:
            cmd.extend(["--speculative-config", json.dumps(config["speculative_config"])])

        run_cmd(cmd)

        metadata_path = out_dir / "metadata.json"
        metadata = {
            **meta,
            **common,
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
