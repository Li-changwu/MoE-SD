#!/usr/bin/env python3
"""
S0: SVD Low-Rank Hypothesis Verification for BDE (Base-Delta Expert Decomposition)

Loads real expert weights from Qwen3-30B-A3B-Instruct-2507 and analyzes:
1. Inter-expert similarity (do experts share a common base?)
2. Delta low-rankness (how fast do singular values decay?)
3. Energy retention at various ranks (rank-32/64/128/256)
4. Per-layer variation (are all layers similar?)

Usage:
    python scripts/analyze_expert_svd.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

MODEL_DIR = "/root/models/Qwen3-30B-A3B-Instruct-2507"

# Layers to analyze (sample across depth)
LAYERS_TO_ANALYZE = [0, 6, 12, 18, 24, 30, 36, 42, 47]

# Ranks to evaluate
RANKS = [8, 16, 32, 64, 128, 256]


def load_shard_index():
    """Load the safetensors index to find which shard contains each tensor."""
    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def load_expert_weights(weight_map, layer_idx, proj_name):
    """
    Load all 128 expert weight matrices for a given layer and projection.
    
    proj_name: "gate_proj", "up_proj", or "down_proj"
    Returns: tensor of shape [128, out_dim, in_dim]
    """
    experts = []
    # Determine shard file for first expert to get shape
    key_template = f"model.layers.{layer_idx}.mlp.experts.{{eid}}.{proj_name}.weight"
    
    # Group experts by shard file for efficient loading
    shard_to_eids = {}
    for eid in range(128):
        key = key_template.format(eid=eid)
        shard_file = weight_map.get(key)
        if shard_file is None:
            raise KeyError(f"Weight key not found: {key}")
        if shard_file not in shard_to_eids:
            shard_to_eids[shard_file] = []
        shard_to_eids[shard_file].append((eid, key))
    
    # Load from each shard
    expert_dict = {}
    for shard_file, eid_keys in shard_to_eids.items():
        shard_path = os.path.join(MODEL_DIR, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for eid, key in eid_keys:
                expert_dict[eid] = f.get_tensor(key).float()  # float32 for SVD precision
    
    # Stack in order
    experts = [expert_dict[i] for i in range(128)]
    return torch.stack(experts)  # [128, out_dim, in_dim]


def analyze_layer(weight_map, layer_idx):
    """Analyze one layer's expert weights."""
    results = {"layer": layer_idx}
    
    for proj_name, proj_label in [
        ("gate_proj", "w_gate"),
        ("up_proj", "w_up"),
        ("down_proj", "w_down"),
    ]:
        print(f"    Loading {proj_label}...", end=" ", flush=True)
        t0 = time.time()
        W = load_expert_weights(weight_map, layer_idx, proj_name)
        n_experts, out_dim, in_dim = W.shape
        print(f"[{n_experts}x{out_dim}x{in_dim}] ({time.time()-t0:.1f}s)")
        
        # 1. Compute base (mean across experts)
        W_base = W.mean(dim=0)  # [out_dim, in_dim]
        
        # 2. Compute deltas
        deltas = W - W_base.unsqueeze(0)  # [128, out_dim, in_dim]
        
        # 3. Inter-expert similarity metrics
        # Frobenius norm of base vs mean delta norm
        base_norm = torch.norm(W_base).item()
        delta_norms = torch.norm(deltas.reshape(128, -1), dim=1)  # [128]
        mean_delta_norm = delta_norms.mean().item()
        max_delta_norm = delta_norms.max().item()
        delta_ratio = mean_delta_norm / base_norm  # smaller = more similar
        
        # 4. Pairwise cosine similarity (sample 20 pairs for speed)
        rng = np.random.RandomState(42)
        pairs = [(rng.randint(128), rng.randint(128)) for _ in range(20)]
        cos_sims = []
        for i, j in pairs:
            if i == j:
                continue
            cos = torch.nn.functional.cosine_similarity(
                W[i].reshape(1, -1), W[j].reshape(1, -1)
            ).item()
            cos_sims.append(cos)
        avg_cos_sim = np.mean(cos_sims) if cos_sims else 0.0
        
        # 5. SVD analysis on a representative subset of deltas
        # Full SVD on all 128 is expensive; sample 16 experts
        sample_eids = [0, 8, 16, 24, 32, 48, 64, 80, 96, 112, 4, 12, 20, 28, 36, 44]
        
        energy_at_rank = {r: [] for r in RANKS}
        singular_values_list = []
        
        print(f"    SVD on 16 sampled experts...", end=" ", flush=True)
        t0 = time.time()
        
        for eid in sample_eids:
            delta = deltas[eid]  # [out_dim, in_dim]
            # Economy SVD (only compute min(out_dim, in_dim) singular values)
            try:
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            except Exception as e:
                print(f"SVD failed for expert {eid}: {e}")
                continue
            
            total_energy = (S ** 2).sum().item()
            singular_values_list.append(S.numpy())
            
            for r in RANKS:
                if r <= len(S):
                    retained = (S[:r] ** 2).sum().item()
                    energy_at_rank[r].append(retained / total_energy if total_energy > 0 else 0)
                else:
                    energy_at_rank[r].append(1.0)
        
        print(f"({time.time()-t0:.1f}s)")
        
        # Aggregate results
        proj_results = {
            "shape": f"{out_dim}x{in_dim}",
            "base_norm": base_norm,
            "mean_delta_norm": mean_delta_norm,
            "max_delta_norm": max_delta_norm,
            "delta_base_ratio": delta_ratio,
            "avg_pairwise_cosine": avg_cos_sim,
            "energy_retention": {
                r: {
                    "mean": np.mean(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                }
                for r, vals in energy_at_rank.items()
                if vals
            },
        }
        
        # Singular value decay curve (averaged)
        if singular_values_list:
            avg_sv = np.mean(singular_values_list, axis=0)
            # Normalize
            avg_sv_norm = avg_sv / avg_sv[0] if avg_sv[0] > 0 else avg_sv
            proj_results["sv_decay_top20"] = avg_sv_norm[:20].tolist()
        
        results[proj_label] = proj_results
    
    return results


def print_results(all_results):
    """Pretty-print the analysis results."""
    print("\n" + "=" * 80)
    print("BDE Low-Rank Hypothesis Verification Results")
    print("=" * 80)
    
    for layer_result in all_results:
        layer_idx = layer_result["layer"]
        print(f"\n{'─' * 70}")
        print(f"Layer {layer_idx}")
        print(f"{'─' * 70}")
        
        for proj_key in ["w_gate", "w_up", "w_down"]:
            r = layer_result[proj_key]
            print(f"\n  {proj_key} ({r['shape']}):")
            print(f"    ‖W_base‖ = {r['base_norm']:.2f}")
            print(f"    ‖ΔW‖ mean/max = {r['mean_delta_norm']:.4f} / {r['max_delta_norm']:.4f}")
            print(f"    ‖ΔW‖/‖W_base‖ = {r['delta_base_ratio']:.4f} ({r['delta_base_ratio']*100:.2f}%)")
            print(f"    Avg pairwise cosine = {r['avg_pairwise_cosine']:.4f}")
            
            print(f"    Energy retention (Frobenius norm² ratio):")
            for rank in RANKS:
                if rank in r["energy_retention"]:
                    e = r["energy_retention"][rank]
                    print(f"      rank={rank:>3d}: {e['mean']*100:6.2f}% (min={e['min']*100:.2f}%, max={e['max']*100:.2f}%)")
            
            if "sv_decay_top20" in r:
                sv = r["sv_decay_top20"]
                print(f"    SV decay (σ_i/σ_0): ", end="")
                indices = [0, 1, 3, 7, 15, 19]
                for i in indices:
                    if i < len(sv):
                        print(f"σ_{i}={sv[i]:.3f}  ", end="")
                print()
    
    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: Energy Retention at Key Ranks (averaged across all layers)")
    print(f"{'=' * 80}")
    print(f"{'Rank':>6s}  {'gate_proj':>12s}  {'up_proj':>12s}  {'down_proj':>12s}  {'Average':>12s}")
    print(f"{'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    
    for rank in RANKS:
        vals = []
        for proj_key in ["w_gate", "w_up", "w_down"]:
            layer_vals = []
            for lr in all_results:
                if rank in lr[proj_key]["energy_retention"]:
                    layer_vals.append(lr[proj_key]["energy_retention"][rank]["mean"])
            if layer_vals:
                avg = np.mean(layer_vals)
                vals.append(avg)
                print(f"  {avg*100:5.1f}%", end="     ")
            else:
                vals.append(0)
                print(f"  {'N/A':>5s}", end="     ")
        
        overall = np.mean(vals) if vals else 0
        print(f"  r={rank:>3d}", end="")
        print(f"   {overall*100:5.1f}%")
    
    # BDE feasibility verdict
    print(f"\n{'=' * 80}")
    print("BDE FEASIBILITY VERDICT")
    print(f"{'=' * 80}")
    
    # Check rank=64 energy across all projections
    r64_energies = []
    for lr in all_results:
        for proj_key in ["w_gate", "w_up", "w_down"]:
            if 64 in lr[proj_key]["energy_retention"]:
                r64_energies.append(lr[proj_key]["energy_retention"][64]["mean"])
    
    avg_r64 = np.mean(r64_energies) if r64_energies else 0
    min_r64 = np.min(r64_energies) if r64_energies else 0
    
    print(f"  Rank-64 energy retention: avg={avg_r64*100:.1f}%, min={min_r64*100:.1f}%")
    
    if avg_r64 >= 0.95:
        print(f"  ✅ BDE IS HIGHLY FEASIBLE — rank-64 retains ≥95% energy")
        print(f"     Expected HBM reduction: ~6.9× per layer")
    elif avg_r64 >= 0.90:
        print(f"  ⚠️  BDE IS MODERATELY FEASIBLE — rank-64 retains {avg_r64*100:.0f}% energy")
        print(f"     May need rank-128 for acceptable quality")
    elif avg_r64 >= 0.80:
        print(f"  ⚠️  BDE NEEDS HIGH RANK — rank-64 only retains {avg_r64*100:.0f}% energy")
        print(f"     HBM savings will be limited")
    else:
        print(f"  ❌ BDE NOT FEASIBLE — experts are too dissimilar")
        print(f"     Fall back to W8A16 quantization")
    
    # Also check delta/base ratio
    delta_ratios = []
    for lr in all_results:
        for proj_key in ["w_gate", "w_up", "w_down"]:
            delta_ratios.append(lr[proj_key]["delta_base_ratio"])
    avg_ratio = np.mean(delta_ratios)
    print(f"\n  Average ‖ΔW‖/‖W_base‖ = {avg_ratio*100:.2f}%")
    if avg_ratio < 0.1:
        print(f"  ✅ Experts are very similar (delta < 10% of base)")
    elif avg_ratio < 0.3:
        print(f"  ⚠️  Moderate expert diversity (delta = {avg_ratio*100:.0f}% of base)")
    else:
        print(f"  ❌ High expert diversity (delta = {avg_ratio*100:.0f}% of base)")


def main():
    print("BDE Low-Rank Hypothesis Verification")
    print(f"Model: {MODEL_DIR}")
    print(f"Layers to analyze: {LAYERS_TO_ANALYZE}")
    print(f"Ranks: {RANKS}")
    print()
    
    print("Loading safetensors index...")
    weight_map = load_shard_index()
    print(f"  {len(weight_map)} tensors across shards")
    
    all_results = []
    
    for i, layer_idx in enumerate(LAYERS_TO_ANALYZE):
        print(f"\n[{i+1}/{len(LAYERS_TO_ANALYZE)}] Analyzing Layer {layer_idx}...")
        t0 = time.time()
        result = analyze_layer(weight_map, layer_idx)
        all_results.append(result)
        print(f"  Layer {layer_idx} done in {time.time()-t0:.1f}s")
    
    print_results(all_results)
    
    # Save raw results
    output_path = "/root/MoE-SD/results/bde_svd_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types for JSON
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=to_serializable)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
