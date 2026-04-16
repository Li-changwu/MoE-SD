"""AdapMoE dynamic cache allocation via latency-minimizing DP.

Compared with the previous rough utility heuristic, this implementation uses a
per-layer cost model and solves the integer allocation with dynamic
programming, matching the formulation style in the AdapMoE codebase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerRuntimeStat:
    name: str
    capacity: int
    miss_rate: float
    prefetch_acc: float
    single_ratio: float


def _layer_demand(stat: LayerRuntimeStat) -> float:
    miss_term = max(0.0, min(1.0, stat.miss_rate))
    prefetch_term = 1.0 - max(0.0, min(1.0, stat.prefetch_acc))
    multi_term = 1.0 - max(0.0, min(1.0, stat.single_ratio))
    return max(1e-6, 0.45 * miss_term + 0.30 * prefetch_term + 0.25 * multi_term)


def _layer_latency_cost(demand: float, slots: int) -> float:
    """
    Latency proxy for one layer given allocated slots.

    Demand captures miss pressure. More slots reduce miss penalty with
    diminishing returns (1/sqrt(slots)).
    """
    s = max(1, slots)
    return demand / (s ** 0.5)


def allocate_slots_dp(
    stats: list[LayerRuntimeStat],
    total_slots: int,
    min_slots_per_layer: int,
) -> dict[str, int]:
    """Allocate layer cache slots by minimizing total modeled latency."""
    n = len(stats)
    if n == 0:
        return {}

    mins = [max(1, min(min_slots_per_layer, s.capacity)) for s in stats]
    base = sum(mins)
    if base >= total_slots:
        return {s.name: mins[i] for i, s in enumerate(stats)}

    budget = total_slots - base
    demands = [_layer_demand(s) for s in stats]

    inf = 1e30
    # dp[i][b] = min cost using first i layers with b extra slots.
    dp = [[inf] * (budget + 1) for _ in range(n + 1)]
    choose = [[0] * (budget + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        cap_i = max(1, stats[i - 1].capacity)
        min_i = mins[i - 1]
        max_extra_i = max(0, cap_i - min_i)
        d = demands[i - 1]
        for b in range(0, budget + 1):
            best = inf
            best_x = 0
            lim = min(max_extra_i, b)
            for x in range(0, lim + 1):
                prev = dp[i - 1][b - x]
                if prev >= inf / 2:
                    continue
                slots_i = min_i + x
                val = prev + _layer_latency_cost(d, slots_i)
                if val < best:
                    best = val
                    best_x = x
            dp[i][b] = best
            choose[i][b] = best_x

    # Backtrack best feasible budget.
    b = min(range(budget + 1), key=lambda k: dp[n][k])
    alloc = {}
    for i in range(n, 0, -1):
        x = choose[i][b]
        s = stats[i - 1]
        alloc[s.name] = mins[i - 1] + x
        b -= x

    return alloc
