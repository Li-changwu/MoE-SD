# Acceptance-Aware Prefetch v1

This policy scores candidates with:

`score = p(expert_needed) * p(token_accepted) * benefit - cost`

## Output Levels

- `hard`: immediate prefetch
- `soft`: opportunistic prefetch
- `defer`: no prefetch

## Deliverables

- `controllers/prefetch_policy.py`
- `configs/policies/prefetch_v1.yaml`

## Notes

- v1 is heuristic and intentionally lightweight.
- wasted bytes / hit-miss counters are included for instrumentation compatibility.
