# Benchmark Contract

All performance validation must use `vllm bench`.

Primary metrics:
- TTFT
- TPOT
- ITL
- throughput
- goodput

Every result must record:
- model
- benchmark mode
- workload profile
- request rate
- input length
- output length
- seed
- git commit
