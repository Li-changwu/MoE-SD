#!/usr/bin/env bash
set -euo pipefail

echo "==> initialize repository layout"

mkdir -p \
  adapters \
  controllers \
  collectors \
  configs/models \
  configs/workloads \
  configs/policies \
  configs/experiments \
  scripts \
  tools \
  tests \
  results/raw \
  results/parsed \
  results/figures \
  docs

touch \
  adapters/__init__.py \
  controllers/__init__.py \
  collectors/__init__.py \
  tools/__init__.py

cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.env

# Build
build/
dist/
*.egg-info/

# Results
results/raw/*
results/parsed/*
results/figures/*

# Exceptions: keep directory structure
!results/raw/.gitkeep
!results/parsed/.gitkeep
!results/figures/.gitkeep

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
EOF

mkdir -p results/raw results/parsed results/figures
touch results/raw/.gitkeep results/parsed/.gitkeep results/figures/.gitkeep

cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vllm-moe-sd-scheduler"
version = "0.1.0"
description = "Plugin-style scheduling framework for memory-constrained MoE + speculative decoding on vLLM"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools]
packages = ["adapters", "controllers", "collectors", "tools"]
EOF

cat > docs/repo_structure.md <<'EOF'
# Repository Structure

- adapters/: vLLM/speculators version adapters
- controllers/: scheduling logic, governor, memory partition, prefetch policy
- collectors/: acceptance, MoE trace, memory trace collectors
- configs/: models, workloads, policies, experiments
- scripts/: runnable shell entrypoints
- tools/: parsing, plotting, utility scripts
- results/: raw and parsed experiment results
- docs/: design notes and reproducibility documents
EOF

echo "==> done"
