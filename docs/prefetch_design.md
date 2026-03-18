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

# Frontier-Aware Prefetch v2

v2 extends the v1 score with speculative-frontier structure:

`score = base + shared_bonus + reuse_bonus - depth_penalty - isolation_penalty - waste_penalty`

Where:

- `shared_bonus`: boosts experts shared by multiple speculative branches
- `reuse_bonus`: boosts experts likely to be reused by nearby requests/steps
- `depth_penalty`: discounts experts only needed by deeper speculative tokens
- `isolation_penalty`: discounts experts that appear in only one branch
- `waste_penalty`: estimates bytes likely to be wasted if the branch dies

Intuition:

- hard-prefetch the union core of the speculative frontier
- soft-prefetch uncertain but moderately shared experts
- defer deep isolated experts that are likely to become rejected-token waste
