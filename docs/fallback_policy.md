# Fallback / Hot-Switch Policy

This policy ensures benchmark runs continue when controller or collector paths fail.

## Supported Modes

- `controller`: full scheduler path
- `native_eagle3`: fallback to native EAGLE-3
- `no_sd`: fallback to non-spec decode baseline
- `observe_only`: keep collecting traces without applying decisions

## Fallback Chain

1. controller error -> native_eagle3
2. native_eagle3 error -> no_sd
3. unknown error path -> observe_only

## Force Mode

`force_mode` can override runtime mode for emergency recovery:
- `native_eagle3`
- `no_sd`
- `observe_only`

## Logging Contract

Every fallback event must emit trace fields:
- request_id
- requested_mode
- resolved_mode
- fallback_applied
- reason

Silent fallback is forbidden.
