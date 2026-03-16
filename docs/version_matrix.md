# Version Matrix

This matrix is frozen for baseline and controller comparisons.

## Runtime Versions

| Component | Version |
| --- | --- |
| Python | 3.10 |
| CUDA Toolkit | 12.4 |
| Driver | 550.xx |
| NCCL | 2.21 |
| vLLM | 0.8.5 |
| Speculators | eagle3-compatible build |
| PyTorch | 2.6.0 |
| transformers | 4.51.0 |
| flash-attn | 2.7.4.post1 |

## Reproducibility Contract

- Use `requirements.txt` as the package lock.
- Use `env.lock` as the system/runtime lock.
- Use `Dockerfile` for portable setup.
- Run `scripts/bootstrap.sh` then `scripts/env_report.sh` on every new host.
- Keep `docs/env_report.txt` as the environment evidence for benchmark runs.

## Validation Checklist

- `vllm serve` starts successfully.
- `vllm bench serve` runs and writes raw json.
- `docs/env_report.txt` is generated and archived with benchmark results.
