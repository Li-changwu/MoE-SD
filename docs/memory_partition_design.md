# Dynamic Memory Partition Controller v2

Dynamic controller adapts expert/speculative/KV budgets to runtime pressure.

## Inputs

- GPU memory usage ratio
- acceptance rate
- runtime step id

## Behavior

- medium/high pressure => move budget to KV reserve
- low acceptance => reduce speculative budget
- smooth budgets to reduce oscillation
- apply updates at fixed step interval

## Deliverables

- `controllers/memory_partition_controller.py`
- `configs/policies/memory_partition_v2.yaml`
