#!/usr/bin/env python3
"""Compare v1 and v2 prefetch policies on representative speculative frontiers."""

from __future__ import annotations

from controllers.interface import Phase, RequestState, RuntimeState
from controllers.prefetch_policy import (
    AcceptanceAwarePrefetchPolicy,
    FrontierAwarePrefetchPolicy,
)


def build_state(acceptance_rate: float) -> RuntimeState:
    request = RequestState(
        request_id="aggregate",
        prompt_len=256,
        output_len=128,
        request_rate=50.0,
        phase=Phase.DECODE,
    )
    return RuntimeState(
        request=request,
        step_id=1,
        gpu_mem_used_mb=26000.0,
        gpu_mem_total_mb=48000.0,
        kv_cache_mb=12000.0,
        acceptance_rate=acceptance_rate,
    )


def pretty_bytes(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    return f"{mib:.1f} MiB"


def run_case(name: str, candidates: list[dict], acceptance_rate: float) -> None:
    state = build_state(acceptance_rate)
    policy_v1 = AcceptanceAwarePrefetchPolicy()
    policy_v2 = FrontierAwarePrefetchPolicy()

    out_v1 = policy_v1.decide(candidates, state)
    out_v2 = policy_v2.decide(candidates, state)

    print(f"\n=== {name} ===")
    print(f"acceptance_rate={acceptance_rate:.2f}")
    print(
        "v1:",
        f"hard={out_v1['hard']} soft={out_v1['soft']} defer={out_v1['defer']}",
        f"waste={pretty_bytes(int(out_v1['wasted_prefetched_bytes']))}",
    )
    print(
        "v2:",
        f"hard={out_v2['hard']} soft={out_v2['soft']} defer={out_v2['defer']}",
        f"waste={pretty_bytes(int(out_v2['wasted_prefetched_bytes']))}",
        f"reason={out_v2['reason']}",
    )


def main() -> None:
    shared_frontier = [
        {
            "expert_id": 7,
            "p_expert_needed": 0.94,
            "p_token_accepted": 0.45,
            "benefit": 1.20,
            "cost": 0.08,
            "shared_count": 4,
            "frontier_size": 4,
            "avg_depth": 1.0,
            "frontier_depth": 3.0,
            "expert_reuse": 0.95,
        },
        {
            "expert_id": 19,
            "p_expert_needed": 0.88,
            "p_token_accepted": 0.45,
            "benefit": 1.10,
            "cost": 0.10,
            "shared_count": 3,
            "frontier_size": 4,
            "avg_depth": 1.5,
            "frontier_depth": 3.0,
            "expert_reuse": 0.80,
        },
        {
            "expert_id": 42,
            "p_expert_needed": 0.86,
            "p_token_accepted": 0.45,
            "benefit": 1.05,
            "cost": 0.10,
            "shared_count": 1,
            "frontier_size": 4,
            "avg_depth": 3.0,
            "frontier_depth": 3.0,
            "expert_reuse": 0.10,
        },
        {
            "expert_id": 73,
            "p_expert_needed": 0.62,
            "p_token_accepted": 0.45,
            "benefit": 1.00,
            "cost": 0.12,
            "shared_count": 1,
            "frontier_size": 4,
            "avg_depth": 2.5,
            "frontier_depth": 3.0,
            "expert_reuse": 0.15,
        },
    ]

    low_acceptance_frontier = [
        {
            "expert_id": 11,
            "p_expert_needed": 0.90,
            "p_token_accepted": 0.28,
            "benefit": 1.10,
            "cost": 0.08,
            "shared_count": 2,
            "frontier_size": 4,
            "avg_depth": 2.0,
            "frontier_depth": 3.0,
            "expert_reuse": 0.70,
        },
        {
            "expert_id": 58,
            "p_expert_needed": 0.84,
            "p_token_accepted": 0.28,
            "benefit": 1.00,
            "cost": 0.08,
            "shared_count": 1,
            "frontier_size": 4,
            "avg_depth": 3.0,
            "frontier_depth": 3.0,
            "expert_reuse": 0.05,
        },
        {
            "expert_id": 63,
            "p_expert_needed": 0.78,
            "p_token_accepted": 0.28,
            "benefit": 1.00,
            "cost": 0.10,
            "shared_count": 1,
            "frontier_size": 4,
            "avg_depth": 3.0,
            "frontier_depth": 3.0,
            "expert_reuse": 0.10,
        },
    ]

    run_case("shared-frontier / moderate acceptance", shared_frontier, 0.45)
    run_case("deep-isolated / low acceptance", low_acceptance_frontier, 0.28)


if __name__ == "__main__":
    main()
