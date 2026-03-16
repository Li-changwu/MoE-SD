# Static Governor v0

Static governor v0 provides a minimal closed-loop controller using deterministic rules.

## Scope

- controls speculation K
- controls coarse memory partition
- writes explicit reason strings for every decision
- does not implement advanced prefetch logic

## Inputs

- phase (`prefill` / `decode`)
- GPU memory used/total
- acceptance rate
- KV cache growth speed

## K Decision Rules

- prefill starts conservative (`prefill_max_k`)
- decode can use higher cap (`decode_max_k`)
- high memory pressure => `K=1`
- medium memory pressure => `K<=2`
- low acceptance => reduce K

## Memory Partition Rules

Default split:
- KV reserve: 55%
- expert budget: 25%
- speculative budget: 20%

Adjustments:
- medium/high pressure increases KV reserve and shrinks speculative budget
- fast KV growth adds extra KV reserve
- low acceptance shrinks speculative budget

## Output Example

```json
{
  "k": 1,
  "apply": true,
  "reason": "phase=decode|high_memory_pressure|low_acceptance"
}
```

```json
{
  "expert_budget_mb": 9216,
  "speculative_budget_mb": 3072,
  "kv_reserve_mb": 24576,
  "apply": true,
  "reason": "pressure=high|high_pressure_shrink_spec|fast_kv_growth"
}
```

## Validation

- see `tests/test_static_governor.py`
- compare against native EAGLE-3 via benchmark before merge-candidate judgement
