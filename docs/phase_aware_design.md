# Phase-Aware Governor v1

Phase-aware governor v1 separates prefill and decode objectives.

## Design Goal

- prefill: protect TTFT
- decode: improve TPOT / throughput

## Phase Rules

### Prefill

- lower speculation K cap
- higher KV reserve bias
- avoid aggressive speculative overhead

### Decode

- higher K cap when memory pressure is safe
- balanced KV/speculative budgets
- acceptance low => reduce K

## Deliverables

- `controllers/phase_aware_governor.py`
- `configs/policies/phase_aware_v1.yaml`
- phase-aware reason strings in decision outputs

## Validation Checklist

- compare prefill/decode decision traces on same workload
- show phase-dependent K shift in trace logs
- benchmark regression should include long-prompt and long-output profiles
