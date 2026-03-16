---
name: Experiment Report
about: 记录一次标准化实验，并可直接汇总进决策看板
title: "EXP-YYYYMMDD-XXX: "
labels: ["dashboard"]
assignees: []
---

## 1) 改动信息
- experiment_id: EXP-YYYYMMDD-XXX
- owner:
- issue_id:
- branch:
- commit_hash:
- optimization_target: (TTFT / TPOT / throughput / goodput)
- optimization_module:
- change_summary:
- hypothesis:

## 2) 实验条件
- model:
- spec_method:
- policy_name:
- workload_profile:
- hardware_profile:
- seed:

## 3) 基线信息
- baseline_experiment_id:
- baseline_type: (no_sd / native_eagle3 / previous_best)
- comparison_scope:

## 4) 核心结果
- ttft_p95_ms:
- tpot_p95_ms:
- throughput_tok_per_s:
- goodput:
- gpu_mem_peak_mb:
- fallback_count:
- oom_count:

## 5) 相对基线变化
- delta_ttft_p95_pct:
- delta_tpot_p95_pct:
- delta_throughput_pct:
- delta_goodput_pct:
- delta_gpu_mem_peak_pct:

## 6) 结论
- result_label: (win / partial_win / neutral / regression / invalid)
- primary_gain:
- primary_cost:
- final_conclusion:
- is_merge_candidate: (true / false)

## 7) 产物链接
- result_path:
- dashboard_html:
- regression_table:

## Checklist
- [ ] 已写入 results/experiments/EXP-... 目录
- [ ] 已执行 make append-exp
- [ ] 已执行 make update-readme-dashboard
- [ ] 若 result_label=regression，已加标签 regression
- [ ] 若 is_merge_candidate=true，已加标签 merge-candidate
