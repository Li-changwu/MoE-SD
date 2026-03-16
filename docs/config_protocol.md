# Config Protocol

Each experiment config should include:

- model
- tokenizer (optional)
- speculative_method
- num_speculative_tokens
- workload_profile
- memory_budget
- policy_name
- trace_flags
- output_dir
- feature_flags

## Minimal Scheduler Config (YAML)

```yaml
model: Qwen/Qwen3-30B-A3B-Instruct-2507
workload_profile: online_medium
policy_name: native
feature_flags:
	enable_controller: false
	enable_prefetch: false
	enable_memory_partition: false
	observation_only: true
```

## Feature Flag Contract

- `enable_controller`: enable scheduler decision path.
- `enable_prefetch`: allow prefetch policy action.
- `enable_memory_partition`: allow runtime budget adjustments.
- `observation_only`: collect and log decisions without applying actions.

Defaults are conservative and must preserve native behavior.

## Validation Rules

- Missing `model` or `workload_profile` is invalid.
- If `enable_controller=false`, other action flags must be treated as no-op.
- Any patch-based integration must be recorded in experiment metadata.
