# no-SD Repeat Stability (3 runs)

Config: prompt=128, output=64, num_prompts=8, request_rate=1, max_model_len=256.

| mode | metric | run1 | run2 | run3 | mean | std | CV% |
|---|---:|---:|---:|---:|---:|---:|---:|
| serve | mean_ttft_ms | 11094.5985 | 8738.6175 | 8867.3767 | 9566.8642 | 1081.5494 | 11.31 |
| serve | output_throughput_tok_s | 7.1590 | 7.6052 | 6.6500 | 7.1381 | 0.3902 | 5.47 |
| serve | mean_tpot_ms | 872.9577 | 860.8207 | 986.9131 | 906.8972 | 56.7963 | 6.26 |
| latency | avg_latency_ms | 63794.7896 | 61249.1767 | 62797.0736 | 62613.6800 | 1047.3016 | 1.67 |
| throughput | tokens_per_second | 23.2881 | 23.2879 | 23.2641 | 23.2800 | 0.0113 | 0.05 |
| throughput | requests_per_second | 0.1213 | 0.1213 | 0.1212 | 0.1213 | 0.0001 | 0.05 |
