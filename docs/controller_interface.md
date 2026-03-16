# Controller Interface

This document freezes the controller API used by all scheduling policies.

## Python Interface

Defined in `controllers/interface.py`:

- `on_request_arrival(req_meta)`
- `on_prefill_begin(state)`
- `on_prefill_end(state)`
- `on_decode_step(step_meta)`
- `decide_speculation_k(state) -> dict`
- `decide_memory_partition(state) -> dict`
- `decide_prefetch(expert_candidates, state) -> dict`

## State Levels

- request-level: request metadata and static workload fields
- phase-level: prefill/decode current phase context
- step-level: per decode-step counters and memory/acceptance signals

## Output Contract

### decide_speculation_k

```json
{
  "k": 4,
  "apply": true,
  "reason": "decode_phase_acceptance_high"
}
```

### decide_memory_partition

```json
{
  "expert_budget_mb": 2048,
  "speculative_budget_mb": 768,
  "kv_reserve_mb": 4096,
  "apply": true,
  "reason": "memory_pressure_medium"
}
```

### decide_prefetch

```json
{
  "hard": [3, 8],
  "soft": [12],
  "defer": [1, 5],
  "apply": true,
  "reason": "acceptance_high_io_cost_balanced"
}
```

## Native Compatibility

- `NoOpController` is the default safe implementation.
- When controller is disabled, all decisions return `apply=false`.
- This guarantees no behavior change for native baseline runs.
