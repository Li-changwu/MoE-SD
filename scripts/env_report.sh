#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="docs"
mkdir -p "$OUT_DIR"

{
  echo "# Environment Report"
  echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo
  echo "## nvidia-smi"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "## python packages"
  python3 - <<'PY'
import importlib
mods = ["vllm", "torch", "transformers", "flash_attn", "numpy", "pandas", "pyarrow"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m}=={getattr(mod, '__version__', 'unknown')}")
    except Exception:
        print(f"{m}==not-installed")
PY
} > "$OUT_DIR/env_report.txt"

echo "Wrote $OUT_DIR/env_report.txt"
