# Integration Boundary

This document defines plugin/adapter/patch boundaries for `vllm-moe-sd-scheduler`.

## Goals

- Keep scheduler integration minimally intrusive.
- Ensure all behavior changes are gated by feature flags.
- Allow fast rollback to native EAGLE-3 or no-SD paths.

## Integration Layers

1. Plugin Layer (preferred)
- Injects controller decisions through runtime hooks.
- Reads state from collectors and benchmark metadata.
- Must not modify vLLM internal scheduler source code.

2. Adapter Layer (allowed)
- Version compatibility adapters under `adapters/`.
- Handles API differences across vLLM/speculator versions.
- Must preserve equivalent behavior for identical inputs.

3. Patch Layer (restricted)
- Allowed only when plugin/adapter cannot expose required signals.
- Every patch must be documented with rationale and rollback plan.
- Patch usage must be explicitly marked in experiment metadata.

## Patch-Allowed Cases

- Missing lifecycle hook for request/step state collection.
- Missing callback needed for stable fallback switching.
- Missing counters required for acceptance or memory accounting.

## Patch-Forbidden Cases

- Implementing policy logic directly inside vLLM core.
- Editing code solely for convenience when adapter can solve it.
- Any change without reproducibility note and rollback instruction.

## Runtime Contract

- Feature flags default to disabled.
- Disabled controller must keep native behavior unchanged.
- Observation-only mode is allowed for safe rollout.

## Deliverable Mapping

- package skeleton: `vllm_moe_sd_scheduler/`
- entry points: `vllm_moe_sd_scheduler/entrypoints.py`, `vllm_moe_sd_scheduler/cli.py`
- feature flags: `vllm_moe_sd_scheduler/feature_flags.py`
- config schema: `docs/config_protocol.md`
