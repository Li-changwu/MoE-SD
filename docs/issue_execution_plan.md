# Issue Execution Plan

This plan is generated from all open issues and follows dependency order.

## Wave 0 (P0 foundation)

- #1 Freeze environment matrix and bootstrap chain.
- #2 Reproduce no-SD baseline.
- #3 Reproduce native EAGLE-3 baseline.
- #4 Unify benchmark harness.
- #15 Freeze workload matrix.
- #19 Unify Makefile/script entry points.
- #20 Freeze result naming convention.

## Wave 1 (Observability)

- #5 EAGLE-3 acceptance collector.
- #6 Qwen3 MoE expert trace collector.
- #7 Memory breakdown analyzer.

## Wave 2 (Architecture + Minimum Control Loop)

- #8 Package boundary and integration protocol.
- #9 Controller interface freeze.
- #10 Static governor v0.
- #14 Fallback / hot-switch.

## Wave 3 (Core mechanisms)

- #11 Phase-aware governor v1.
- #12 Acceptance-aware prefetch v1.
- #13 Dynamic memory partition v2.

## Wave 4 (Paper-level experiments)

- #16 Main comparison experiments.
- #17 Ablations.
- #18 Artifact packaging.

## Progress Update Policy

For each issue:

1. Post kickoff comment with implementation scope.
2. Link PR/commit and produced artifacts.
3. Update checklist progress with concrete evidence paths.
4. Close issue only when Definition of Done is met.
