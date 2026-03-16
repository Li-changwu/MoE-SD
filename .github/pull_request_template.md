## Summary
- change_summary:
- linked_issue:
- linked_experiment_id:

## Workload & Baseline
- workload_profile:
- baseline_experiment_id:
- comparison_scope:

## Metrics Delta (vs baseline)
- delta_ttft_p95_pct:
- delta_tpot_p95_pct:
- delta_throughput_pct:
- delta_goodput_pct:
- delta_gpu_mem_peak_pct:

## Decision
- result_label: (win / partial_win / neutral / regression / invalid)
- score_main:
- is_merge_candidate: (true / false)
- final_conclusion:

## Risks
- fallback_count impact:
- oom_count impact:
- scope limitations:

## Artifacts
- result_path:
- dashboard_html: docs/dashboard/optimization_dashboard.html
- regression_table: results/parsed/regression_stability.csv

## Checklist
- [ ] 实验产物已按规范落盘（meta/summary/compare/trace）
- [ ] experiment_registry.csv 已更新
- [ ] README dashboard snapshot 已刷新
- [ ] 涉及回归时已关联 regression issue
- [ ] 本 PR 结论与 result_label 一致
