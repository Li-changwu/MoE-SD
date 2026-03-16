# MoE Trace Report

This report summarizes Qwen3 MoE routing traces.

## Inputs

- moe trace with fields: request_id, token_idx, layer_id, experts (top-k list)
- optional phase field (`prefill` / `decode`)

## Output Artifacts

- `results/moe_trace/expert_heat.parquet`
- `results/moe_trace/overlap.parquet`
- `results/moe_trace/reuse_distance.parquet`
- `results/moe_trace/expert_heatmap.png`
- `results/moe_trace/overlap_distribution.png`
- `results/moe_trace/reuse_distance_distribution.png`

## How To Run

```bash
python collectors/moe_trace_collector.py \
  --trace <moe_trace.jsonl> \
  --output-dir results/moe_trace
```

## Questions This Report Answers

- Which experts are hottest per layer.
- How much token-to-token overlap exists.
- What reuse-distance pattern appears during decode.
