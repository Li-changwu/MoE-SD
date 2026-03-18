"""
Expert Temporal Locality Analyzer
==================================
Analyzes expert access patterns across SD draft→verify rounds
to quantify temporal locality and cache hit potential.

Three key metrics:
  1. Inter-round overlap: |experts(r) ∩ experts(r-1)| / |experts(r)|
  2. Expert reuse distance: rounds between consecutive uses of same expert
  3. Draft-target correlation: |E_draft ∩ E_target| / |E_target|
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LocalityStats:
    """Summary statistics for expert temporal locality."""
    mean_interround_overlap: float = 0.0
    std_interround_overlap: float = 0.0
    mean_reuse_distance: float = 0.0
    mean_draft_target_correlation: float = 0.0
    cache_hit_rate_estimate: float = 0.0
    num_rounds: int = 0


class ExpertTemporalLocalityAnalyzer:
    """
    Tracks expert access patterns across SD verify rounds and
    computes temporal locality metrics.
    """

    def __init__(self, num_experts: int = 128, top_k: int = 8, num_layers: int = 48):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_layers = num_layers

        # Per-round expert sets: round_id -> layer_id -> set of expert ids
        self._round_experts: list[dict[int, set[int]]] = []
        # Draft routing predictions: round_id -> layer_id -> set of expert ids
        self._draft_routing: list[dict[int, set[int]]] = []
        # Expert last-seen round: (layer_id, expert_id) -> round_id
        self._last_seen: dict[tuple[int, int], int] = {}
        # Reuse distances
        self._reuse_distances: list[int] = []
        # Overlap values
        self._overlaps: list[float] = []
        # Draft-target correlations
        self._correlations: list[float] = []

    def record_verify_round(
        self,
        round_id: int,
        expert_indices: dict[int, list[list[int]]],  # layer -> [token_experts_list, ...]
        draft_indices: Optional[dict[int, list[list[int]]]] = None,
    ):
        """
        Record experts used in a verify round.

        Args:
            round_id: Sequence number of this verify round
            expert_indices: {layer_id: [[experts_token0], [experts_token1], ...]}
            draft_indices: Same format, from draft model's predicted routing
        """
        # Compute expert set per layer for this round
        round_layer_experts = {}
        for layer_id, token_expert_lists in expert_indices.items():
            expert_set = set()
            for experts in token_expert_lists:
                expert_set.update(experts)
            round_layer_experts[layer_id] = expert_set

        self._round_experts.append(round_layer_experts)

        # Record draft routing
        if draft_indices:
            draft_layer_experts = {}
            for layer_id, token_expert_lists in draft_indices.items():
                expert_set = set()
                for experts in token_expert_lists:
                    expert_set.update(experts)
                draft_layer_experts[layer_id] = expert_set
            self._draft_routing.append(draft_layer_experts)
        else:
            self._draft_routing.append({})

        # Compute inter-round overlap
        if len(self._round_experts) >= 2:
            prev = self._round_experts[-2]
            curr = self._round_experts[-1]
            overlaps_this_round = []
            for layer_id in curr:
                if layer_id in prev and curr[layer_id] and prev[layer_id]:
                    intersection = len(curr[layer_id] & prev[layer_id])
                    union = len(curr[layer_id] | prev[layer_id])
                    if union > 0:
                        overlaps_this_round.append(intersection / len(curr[layer_id]))
            if overlaps_this_round:
                mean_overlap = sum(overlaps_this_round) / len(overlaps_this_round)
                self._overlaps.append(mean_overlap)

        # Compute draft-target correlation
        if draft_indices and round_layer_experts:
            draft_experts_this = self._draft_routing[-1]
            correlations_this = []
            for layer_id in round_layer_experts:
                if layer_id in draft_experts_this:
                    target_set = round_layer_experts[layer_id]
                    draft_set = draft_experts_this[layer_id]
                    if target_set:
                        corr = len(target_set & draft_set) / len(target_set)
                        correlations_this.append(corr)
            if correlations_this:
                self._correlations.append(sum(correlations_this) / len(correlations_this))

        # Update reuse distances
        for layer_id, expert_set in round_layer_experts.items():
            for eid in expert_set:
                key = (layer_id, eid)
                if key in self._last_seen:
                    distance = round_id - self._last_seen[key]
                    self._reuse_distances.append(distance)
                self._last_seen[key] = round_id

    def compute_statistics(self) -> LocalityStats:
        """Compute aggregate locality statistics."""
        import numpy as np

        mean_overlap = float(np.mean(self._overlaps)) if self._overlaps else 0.0
        std_overlap = float(np.std(self._overlaps)) if self._overlaps else 0.0
        mean_reuse = float(np.mean(self._reuse_distances)) if self._reuse_distances else 0.0
        mean_corr = float(np.mean(self._correlations)) if self._correlations else 0.0

        # Estimate cache hit rate: fraction of experts in current round that were also in previous round
        # A simple model: hit_rate ≈ mean_overlap
        cache_hit_estimate = mean_overlap

        return LocalityStats(
            mean_interround_overlap=round(mean_overlap, 4),
            std_interround_overlap=round(std_overlap, 4),
            mean_reuse_distance=round(mean_reuse, 2),
            mean_draft_target_correlation=round(mean_corr, 4),
            cache_hit_rate_estimate=round(cache_hit_estimate, 4),
            num_rounds=len(self._round_experts),
        )

    def generate_report(self) -> dict:
        """Generate full analysis report."""
        import numpy as np

        stats = self.compute_statistics()
        report = {
            "summary": {
                "inter_round_overlap": stats.mean_interround_overlap,
                "inter_round_overlap_std": stats.std_interround_overlap,
                "mean_reuse_distance": stats.mean_reuse_distance,
                "draft_target_correlation": stats.mean_draft_target_correlation,
                "estimated_cache_hit_rate": stats.cache_hit_rate_estimate,
                "num_rounds_analyzed": stats.num_rounds,
            },
            "recommendations": [],
        }

        # Recommendations
        if stats.mean_interround_overlap > 0.4:
            report["recommendations"].append(
                "✅ Inter-round overlap > 40%: Expert cache is valuable"
            )
        else:
            report["recommendations"].append(
                "⚠️ Inter-round overlap < 40%: Expert cache benefit limited"
            )

        if stats.mean_draft_target_correlation > 0.5:
            report["recommendations"].append(
                "✅ Draft-target correlation > 50%: Prefetch from draft routing is effective"
            )
        else:
            report["recommendations"].append(
                "⚠️ Draft-target correlation < 50%: Draft-guided prefetch less effective"
            )

        if self._reuse_distances:
            arr = np.array(self._reuse_distances)
            report["reuse_distance_distribution"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "reuse_within_1_round_pct": float(np.mean(arr <= 1)) * 100,
                "reuse_within_3_rounds_pct": float(np.mean(arr <= 3)) * 100,
            }

        if self._overlaps:
            arr = np.array(self._overlaps)
            report["overlap_distribution"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }

        return report

    def export_csv(self, output_dir: str):
        """Export analysis data to CSV files."""
        import csv
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Inter-round overlap
        if self._overlaps:
            with open(out / "inter_round_overlap.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "overlap"])
                for i, overlap in enumerate(self._overlaps):
                    w.writerow([i + 1, round(overlap, 4)])

        # Reuse distances
        if self._reuse_distances:
            with open(out / "reuse_distance_dist.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["distance", "count"])
                from collections import Counter
                counts = Counter(self._reuse_distances)
                for dist in sorted(counts):
                    w.writerow([dist, counts[dist]])

        # Draft-target correlation
        if self._correlations:
            with open(out / "draft_target_correlation.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "correlation"])
                for i, corr in enumerate(self._correlations):
                    w.writerow([i, round(corr, 4)])


def analyze_from_trace_file(
    trace_path: str,
    K: int = 3,
    num_experts: int = 128,
    top_k: int = 8,
    num_layers: int = 48,
) -> dict:
    """
    Run locality analysis from an expert trace JSONL file.

    Simulates SD verify rounds by grouping consecutive K+1 tokens as a round.
    """
    import json
    from pathlib import Path

    trace_file = Path(trace_path)
    events = []
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Group by request_id
    by_request = defaultdict(list)
    for e in events:
        if e.get("phase", "decode") == "decode":
            by_request[e["request_id"]].append(e)

    analyzer = ExpertTemporalLocalityAnalyzer(
        num_experts=num_experts, top_k=top_k, num_layers=num_layers
    )

    round_id = 0
    for req_id, req_events in by_request.items():
        req_events.sort(key=lambda x: x["token_idx"])

        # Group into verify rounds of K+1 tokens
        for start in range(0, len(req_events) - K, K + 1):
            window = req_events[start: start + K + 1]

            # Build layer -> [token_experts...] mapping
            layer_experts = defaultdict(list)
            for evt in window:
                layer_experts[evt["layer_id"]].append(evt["experts"][:top_k])

            analyzer.record_verify_round(round_id=round_id, expert_indices=dict(layer_experts))
            round_id += 1

    return analyzer.generate_report()
