"""AdapMoE baseline adapter.

Implements the core mechanisms from PKU-SEC-Lab/AdapMoE and maps them
to ELMM's offloaded FusedMoE runtime:
1) DP-based per-layer cache allocation.
2) Sensitivity-threshold adaptive gating (top-k -> top-1 fallback).
3) Next-layer gate based expert prefetching.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Any

import torch


_DP_ACCS = [
    0,
    0.5663265306122449,
    0.7210884353741497,
    0.7891156462585034,
    0.8656462585034014,
    0.858843537414966,
    0.8299319727891157,
    0.8401360544217688,
    0.8486394557823129,
    0.8010204081632653,
    0.8979591836734694,
    0.8979591836734694,
    0.8928571428571429,
    0.9098639455782312,
    0.9251700680272109,
    0.8809523809523809,
    0.9149659863945578,
    0.9387755102040817,
    0.9047619047619048,
    0.8792517006802721,
    0.8758503401360545,
    0.8928571428571429,
    0.8792517006802721,
    0.8758503401360545,
    0.8877551020408163,
    0.8843537414965986,
    0.9217687074829932,
    0.9132653061224489,
    0.8860544217687075,
    0.8996598639455783,
    0.9030612244897959,
    0.8503401360544217,
]

_DP_TOP1_RATIOS = [
    1.901311971939101e-05,
    0.03475535161783361,
    0.004884125918519905,
    0.0009399270927619579,
    0.0003762266347225816,
    0.002507425069433102,
    0.007886289458982268,
    0.034134956771462194,
    0.034509078817228075,
    0.04913390811153893,
    0.012986285530511104,
    0.019571446563590767,
    0.029581493074767616,
    0.0054871764196417,
    0.015524880500307208,
    0.028414047587928246,
    0.06388771550511758,
    0.0709986002526634,
    0.14071690014890662,
    0.2802636362185971,
    0.2986692745856989,
    0.365106444317853,
    0.4144767403128437,
    0.35560653361774475,
    0.4951908113054345,
    0.5209842744834382,
    0.5478910940291751,
    0.6180380387530074,
    0.7799792571119152,
    0.8839920792738672,
    0.9753751835420208,
    0.8577125736121682,
]

_HESSIAN_WEIGHTS = [
    46.69189453125,
    17.303466796875,
    13.0157470703125,
    7.640838623046875,
    4.169464111328125,
    2.2296905517578125,
    1.2559890747070312,
    0.8444786071777344,
    0.6837844848632812,
    0.5602836608886719,
    0.5125999450683594,
    0.4780292510986328,
    0.44536590576171875,
    0.4355907440185547,
    0.38361549377441406,
    0.30994415283203125,
    0.23305416107177734,
    0.1760721206665039,
    0.13840198516845703,
    0.1137852668762207,
    0.10472536087036133,
    0.09542703628540039,
    0.08624792098999023,
    0.07712841033935547,
    0.06937980651855469,
    0.06109476089477539,
    0.0502467155456543,
    0.042557716369628906,
    0.03349781036376953,
    0.025272369384765625,
    0.020682811737060547,
    0.02294778823852539,
]


def _dp_objective(i_1based: int, size: int, adap_gate: bool) -> float:
    s = min(max(size, 0), 8)
    acc = _DP_ACCS[i_1based - 1]
    top1 = _DP_TOP1_RATIOS[i_1based - 1] if adap_gate else 0.0
    f1 = (1 - s / 8.0) * (1 - acc)
    f2 = 2 * max(0.0, (8 - s) * (7 - s) / 56.0) * (1 - acc)
    f3 = max((8 - s) * (7 - s) / 56.0, 0.0) * acc
    f4 = 2 * (8 - s) * s / 56.0 * (1 - acc)
    return top1 * f1 + (1 - top1) * (f2 + f3 + f4)


def _dp_cache_sizes(total_budget: int, adap_gate: bool) -> list[int]:
    n = len(_DP_ACCS)
    dp = [[0.0] * (total_budget + 1) for _ in range(n + 1)]
    take = [[0] * (total_budget + 1) for _ in range(n + 1)]

    for i in range(n):
        dp[i + 1][0] = dp[i][0] + _dp_objective(i + 1, 0, adap_gate)

    for i in range(1, n + 1):
        for j in range(1, total_budget + 1):
            best = float("inf")
            best_k = 1
            for k in range(1, j + 1):
                v = dp[i - 1][j - k] + _dp_objective(i, k, adap_gate)
                if v < best:
                    best = v
                    best_k = k
            dp[i][j] = best
            take[i][j] = best_k

    sizes = [0] * n
    remain = total_budget
    for i in range(n, 0, -1):
        k = take[i][remain]
        sizes[i - 1] = k
        remain -= k
        if remain <= 0:
            break
    return sizes


def _resample_list(values: list[float], target_len: int) -> list[float]:
    if not values:
        return [0.0] * target_len
    if len(values) == target_len:
        return list(values)
    out: list[float] = []
    src_n = len(values)
    for i in range(target_len):
        src_i = int(round(i * (src_n - 1) / max(target_len - 1, 1)))
        out.append(values[src_i])
    return out


class AdapMoEController:
    def __init__(self, elmm_manager: Any):
        self._elmm = elmm_manager
        self._orig_select: dict[str, Any] = {}
        self._hit_to_top1 = defaultdict(int)
        self._prefetch_horizon = int(os.environ.get("ADAPMOE_PREFETCH_HORIZON", "1"))
        self._prefetch_topk = int(os.environ.get("ADAPMOE_PREFETCH_TOPK", "2"))
        self._enable_adap_gate = os.environ.get("ADAPMOE_ADAPTGATE", "1") == "1"

        threshold_base = float(os.environ.get("ADAPMOE_THRESHOLD_BASE", "0.005"))
        thresholds = [math.sqrt(max(threshold_base, 0.0) / max(w, 1e-8)) for w in _HESSIAN_WEIGHTS]
        self._threshold_by_layer = _resample_list(thresholds, len(self._elmm._ordered_layers))

    def _apply_dp_cache_allocation(self):
        if os.environ.get("ADAPMOE_DP_ENABLE", "1") != "1":
            return
        total_slots = sum(c._max_slots for c in self._elmm._layer_caches.values())
        layer_names = self._elmm._ordered_layers
        if not layer_names or total_slots <= 0:
            return

        # AdapMoE DP in the original code uses 32-layer Mixtral table.
        # We map it to the target model depth by deterministic resampling.
        dp_sizes = _dp_cache_sizes(total_slots, self._enable_adap_gate)
        target = _resample_list([float(x) for x in dp_sizes], len(layer_names))
        target_int = [max(1, int(round(x))) for x in target]

        # Fix sum mismatch caused by rounding.
        diff = total_slots - sum(target_int)
        if diff > 0:
            for i in range(diff):
                target_int[i % len(target_int)] += 1
        elif diff < 0:
            need = -diff
            idx = 0
            while need > 0 and idx < len(target_int) * 4:
                j = idx % len(target_int)
                if target_int[j] > 1:
                    target_int[j] -= 1
                    need -= 1
                idx += 1

        for lname, nslots in zip(layer_names, target_int):
            cache = self._elmm._layer_caches.get(lname)
            if cache is not None:
                cache.resize(nslots)

        print(
            f"[AdapMoE] Applied DP cache allocation: total_slots={total_slots}, "
            f"layers={len(layer_names)}",
            file=sys.stderr,
            flush=True,
        )

    def _extract_hidden_states(self, args: tuple, kwargs: dict) -> torch.Tensor | None:
        if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            return kwargs["hidden_states"]
        if args and isinstance(args[0], torch.Tensor):
            return args[0]
        return None

    def _predict_and_prefetch_next_layers(self, layer_idx: int, hidden_states: torch.Tensor):
        if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
            return
        layer_names = self._elmm._ordered_layers
        for h in range(1, self._prefetch_horizon + 1):
            nxt_idx = layer_idx + h
            if nxt_idx >= len(layer_names):
                break
            nxt_name = layer_names[nxt_idx]
            nxt_mod = self._elmm._patched_modules.get(nxt_name)
            if nxt_mod is None or not hasattr(nxt_mod, "gate"):
                continue
            try:
                with torch.no_grad():
                    gate_out = nxt_mod.gate(hidden_states)
                    if isinstance(gate_out, tuple):
                        logits = gate_out[0]
                    else:
                        logits = gate_out
                    probs = torch.softmax(logits, dim=1, dtype=torch.float)
                    k = min(max(self._prefetch_topk, 1), probs.shape[1])
                    top_ids = torch.topk(probs, k, dim=-1).indices[0].tolist()

                # Follow AdapMoE's check-then-prefetch style: prefetch at most
                # one currently-offloaded expert from predicted top-k candidates.
                cache = self._elmm._layer_caches.get(nxt_name)
                target = None
                if cache is not None:
                    for eid in top_ids:
                        if not cache.contains(eid):
                            target = eid
                            break
                elif top_ids:
                    target = top_ids[0]

                if target is not None:
                    self._elmm.prefetch_experts(nxt_name, [target])
            except Exception:
                continue

    def install(self):
        self._apply_dp_cache_allocation()
        for idx, lname in enumerate(self._elmm._ordered_layers):
            module = self._elmm._patched_modules.get(lname)
            if module is None or not hasattr(module, "router"):
                continue
            router = module.router
            if not hasattr(router, "select_experts"):
                continue
            if getattr(router, "_adapmoe_wrapped", False):
                continue

            original = router.select_experts
            self._orig_select[lname] = original
            threshold = self._threshold_by_layer[idx]
            ctrl = self

            def wrapped_select_experts(
                *args,
                _original=original,
                _threshold=threshold,
                _idx=idx,
                _lname=lname,
                **kwargs,
            ):
                topk_weights, topk_ids = _original(*args, **kwargs)
                hidden_states = ctrl._extract_hidden_states(args, kwargs)

                # Adaptive Sensitivity-based Expert Gating (official logic):
                # zero-out low-confidence secondary experts and renormalize.
                if (
                    ctrl._enable_adap_gate
                    and isinstance(topk_weights, torch.Tensor)
                    and isinstance(topk_ids, torch.Tensor)
                    and topk_weights.ndim == 2
                    and topk_weights.shape[1] >= 2
                    and hidden_states is not None
                    and isinstance(hidden_states, torch.Tensor)
                    and hidden_states.ndim == 2
                    and hidden_states.shape[0] == 1
                    and _threshold > 0
                ):
                    mask = topk_weights[:, 1] < _threshold
                    if bool(mask.any().item()):
                        topk_weights = topk_weights.clone()
                        topk_ids = topk_ids.clone()
                        # Match official top-1 fallback semantics while preserving
                        # fixed tensor shape required by runtime.
                        topk_ids[mask, 1:] = topk_ids[mask, :1]
                        topk_weights[mask, 1:] = 0
                        denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-8)
                        topk_weights = topk_weights / denom
                        ctrl._hit_to_top1[_lname] += int(mask.sum().item())

                if hidden_states is not None:
                    ctrl._predict_and_prefetch_next_layers(_idx, hidden_states)

                return topk_weights, topk_ids

            router.select_experts = wrapped_select_experts
            router._adapmoe_wrapped = True

        print(
            f"[AdapMoE] Activated: adap_gate={self._enable_adap_gate}, "
            f"prefetch_horizon={self._prefetch_horizon}, prefetch_topk={self._prefetch_topk}",
            file=sys.stderr,
            flush=True,
        )


_adapmoe_state: AdapMoEController | None = None


def activate_adapmoe(elmm_manager: Any) -> AdapMoEController | None:
    global _adapmoe_state
    if elmm_manager is None or not getattr(elmm_manager, "_installed", False):
        print("[AdapMoE] ELMM is not installed; skip activation", file=sys.stderr, flush=True)
        return None
    ctrl = AdapMoEController(elmm_manager)
    ctrl.install()
    _adapmoe_state = ctrl
    return ctrl
