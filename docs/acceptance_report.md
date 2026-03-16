# Acceptance Report

This report summarizes EAGLE-3 acceptance behavior by request and decode step.

## Inputs

- acceptance trace: request_id, step_id, proposed_tokens, accepted_tokens
- optional workload metadata: prompt_len, output_len, request_rate, temperature
- optional benchmark summary for request alignment

## Output Artifacts

- `results/acceptance/step_acceptance.parquet`
- `results/acceptance/request_acceptance.parquet`
- `results/acceptance/bucket_acceptance.parquet`
- `results/acceptance/acceptance_by_prompt_bucket.png`
- `results/acceptance/acceptance_by_output_bucket.png`
- `results/acceptance/acceptance_by_qps_bucket.png`

## How To Run

```bash
python collectors/acceptance_collector.py \
  --trace <acceptance_trace.jsonl> \
  --bench-summary results/parsed/summary.csv \
  --output-dir results/acceptance
```

## Questions This Report Answers

- How acceptance changes across prompt/output buckets.
- How acceptance changes as request rate increases.
- Whether request-level acceptance can align with benchmark records.
