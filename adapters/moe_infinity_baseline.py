"""MoE-Infinity baseline adapter.

Core logic mapped from EfficientMoE/MoE-Infinity:
1) Activation tracing (per-layer expert matrix accumulation).
2) Similar-trace based future expert prediction.
3) Activation-aware prefetching for future layers.
4) Priority-based eviction score using frequency + predicted demand.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


def _row_normalize(x: torch.Tensor) -> torch.Tensor:
    d = x.sum(dim=1, keepdim=True)
    d = torch.where(d > 0, d, torch.ones_like(d))
    return x / d


@dataclass
class _TraceEntry:
    matrix: torch.Tensor
    access: int = 1


class MoEInfinityController:
    def __init__(self, elmm_manager: Any):
        self._elmm = elmm_manager
        self._num_layers = len(elmm_manager._ordered_layers)
        self._num_experts = max(
            (m.get("num_experts", 0) for m in elmm_manager._layer_meta.values()),
            default=0,
        )
        self._capacity = int(os.environ.get("MOE_INFINITY_TRACE_CAPACITY", "256"))
        self._prefetch_topk = int(os.environ.get("MOE_INFINITY_PREFETCH_TOPK", "4"))
        self._prefetch_per_layer = int(os.environ.get("MOE_INFINITY_PREFETCH_PER_LAYER", "1"))
        self._prefetch_max_layers = int(os.environ.get("MOE_INFINITY_PREFETCH_MAX_LAYERS", "2"))
        self._prefetch_min_step = int(os.environ.get("MOE_INFINITY_PREFETCH_MIN_STEP", "16"))
        self._prefetch_score_threshold = float(
            os.environ.get("MOE_INFINITY_PREFETCH_SCORE_THRESHOLD", "1e-3")
        )
        self._enable_external_victim = (
            os.environ.get("MOE_INFINITY_EXTERNAL_VICTIM", "1") == "1"
        )
        self._victim_refresh_interval = int(
            os.environ.get("MOE_INFINITY_VICTIM_REFRESH_INTERVAL", "32")
        )
        self._candidate_refresh_interval = int(
            os.environ.get("MOE_INFINITY_CANDIDATE_REFRESH_INTERVAL", "8")
        )
        self._plan_update_stride = int(
            os.environ.get("MOE_INFINITY_PLAN_UPDATE_STRIDE", "4")
        )
        self._plan_pressure_threshold = float(
            os.environ.get("MOE_INFINITY_PLAN_PRESSURE_THRESHOLD", "0.25")
        )
        self._neighbor_window = int(
            os.environ.get("MOE_INFINITY_NEIGHBOR_WINDOW", "2")
        )

        self._trace_collection: list[_TraceEntry] = []
        self._current_trace = torch.zeros(self._num_layers, self._num_experts, dtype=torch.float32)
        self._pred_matrix = torch.zeros_like(self._current_trace)
        self._layer_decay = lambda x, l, L: -1.0 / (L + 1) * (x - l) + 1.0

        self._visit_freq: dict[tuple[int, int], int] = defaultdict(int)
        self._total_visits = 0
        self._step = 0
        self._freq_matrix = torch.zeros_like(self._current_trace)

        self._orig_select: dict[str, Any] = {}
        self._priority_row_cache: dict[int, tuple[int, torch.Tensor]] = {}
        self._priority_row_dirty: set[int] = set(range(self._num_layers))
        self._candidate_cache: dict[int, tuple[int, list[int], int]] = {}
        self._step_protected: dict[int, tuple[int, set[int]]] = {}

    def _get_step_protected(self, layer_id: int) -> set[int]:
        state = self._step_protected.get(layer_id)
        if state is None:
            return set()
        step, eids = state
        if step != self._step:
            return set()
        return set(eids)

    def _protect_for_current_step(self, layer_id: int, eids: list[int] | set[int]):
        if not eids:
            return
        current = self._get_step_protected(layer_id)
        current.update(int(eid) for eid in eids)
        self._step_protected[layer_id] = (self._step, current)

    def on_loaded_expert(self, layer_id: int, eid: int):
        self._protect_for_current_step(layer_id, [eid])

    def _should_refresh_plan(self, layer_id: int, cache: Any, incoming_eids: list[int]) -> bool:
        state = self._candidate_cache.get(layer_id)
        if state is None:
            return True
        age = self._step - state[0]
        if age >= max(self._plan_update_stride, 1):
            return True
        if cache is None:
            return False
        pending = 0
        for eid in incoming_eids:
            if eid not in cache._slot_map:
                pending += 1
        if pending <= 0:
            return False
        denom = float(max(len(cache._slot_map), 1))
        pressure = pending / denom
        return pressure >= self._plan_pressure_threshold

    def _compute_priority_row(self, layer_id: int) -> torch.Tensor:
        if self._num_layers <= 0 or self._num_experts <= 0:
            return torch.zeros(self._num_experts, dtype=torch.float32)

        # Approximate MoE-Infinity global cache ranking under ELMM's per-layer
        # cache: pool demand from current layer + nearby future layers.
        last_layer = min(self._num_layers, layer_id + self._neighbor_window + 1)

        freq_row = torch.zeros(self._num_experts, dtype=torch.float32)
        pred_row = torch.zeros(self._num_experts, dtype=torch.float32)
        weight_sum = 0.0
        for l in range(layer_id, last_layer):
            dist = float(l - layer_id)
            w = 1.0 / (1.0 + dist)
            freq_row += self._freq_matrix[l] * w
            pred_row += self._pred_matrix[l] * w
            weight_sum += w
        if weight_sum > 0:
            inv = 1.0 / weight_sum
            freq_row *= inv
            pred_row *= inv

        freq_row = freq_row / torch.clamp(freq_row.sum(), min=1.0) + 1e-6

        if float(pred_row.sum().item()) <= 0:
            pred_row.fill_(1.0)
        pred_row = pred_row / torch.clamp(pred_row.sum(), min=1.0) + 1e-6

        # For same-layer eviction, topology term is constant and does not
        # affect ranking; keep row-level multiplicative structure.
        return pred_row * freq_row

    def _get_priority_row(self, layer_id: int) -> torch.Tensor:
        cached = self._priority_row_cache.get(layer_id)
        if (
            cached is None
            or layer_id in self._priority_row_dirty
            or (self._step - cached[0]) >= self._victim_refresh_interval
        ):
            row = self._compute_priority_row(layer_id)
            self._priority_row_cache[layer_id] = (self._step, row)
            self._priority_row_dirty.discard(layer_id)
            return row
        return cached[1]

    def _refresh_candidate_cache(self, layer_id: int, cache: Any):
        if cache is None:
            return
        current = list(cache._slot_map.keys())
        if not current:
            self._candidate_cache[layer_id] = (self._step, [], 0)
            return
        row = self._get_priority_row(layer_id)
        current.sort(key=lambda eid: float(row[eid].item()) if row.numel() else 0.0)
        self._candidate_cache[layer_id] = (self._step, current, 0)

    def _maybe_periodic_candidate_refresh(self, layer_id: int):
        if not self._enable_external_victim:
            return
        if self._step % max(self._candidate_refresh_interval, 1) != 0:
            return
        if layer_id < 0 or layer_id >= len(self._elmm._ordered_layers):
            return
        lname = self._elmm._ordered_layers[layer_id]
        cache = self._elmm._layer_caches.get(lname)
        self._refresh_candidate_cache(layer_id, cache)

    def _extract_hidden_states(self, args: tuple, kwargs: dict) -> torch.Tensor | None:
        if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            return kwargs["hidden_states"]
        if args and isinstance(args[0], torch.Tensor):
            return args[0]
        return None

    def _begin_new_sequence_if_needed(self, layer_idx: int, hidden_states: torch.Tensor | None):
        # HumanEval bench runs one prompt at a time with batch=1.
        # We use first-layer prefill as sequence boundary.
        if layer_idx != 0 or hidden_states is None:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[0] <= 1:
            return
        if float(self._current_trace.sum().item()) > 0:
            self._push_current_trace()
            self._current_trace.zero_()

    def _push_current_trace(self):
        entry = _TraceEntry(matrix=self._current_trace.clone(), access=1)
        if len(self._trace_collection) < self._capacity:
            self._trace_collection.append(entry)
            return
        # replace least-accessed entry (MoE-Infinity style bounded trace memory)
        min_i = 0
        min_v = self._trace_collection[0].access
        for i, e in enumerate(self._trace_collection):
            if e.access < min_v:
                min_i = i
                min_v = e.access
        self._trace_collection[min_i] = entry

    def _predict_future_matrix(self, layer_idx: int) -> torch.Tensor:
        if not self._trace_collection:
            return torch.zeros_like(self._current_trace)

        q = _row_normalize(self._current_trace)
        best_i = -1
        best_dist = float("inf")
        for i, entry in enumerate(self._trace_collection):
            c = entry.matrix.clone()
            c[: layer_idx + 1, :] = 1e-9
            c = _row_normalize(c)
            cos = torch.nn.functional.cosine_similarity(q, c, dim=1, eps=1e-6)
            dist = float((1.0 - cos.mean()).item())
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i < 0:
            return torch.zeros_like(self._current_trace)

        self._trace_collection[best_i].access += 1
        pred = self._trace_collection[best_i].matrix.clone()
        pred[:layer_idx, :] = 0
        for l in range(layer_idx, self._num_layers):
            pred[l] = (pred[l] + 1e-8) * self._layer_decay(l, layer_idx, self._num_layers)
        return pred

    def _prefetch_from_prediction(self, layer_idx: int):
        if self._pred_matrix.numel() == 0:
            return
        # Let trace memory warm up before issuing speculative prefetches.
        if self._step < self._prefetch_min_step:
            return

        candidates: list[tuple[float, int, int]] = []
        last_layer = min(self._num_layers, layer_idx + 1 + self._prefetch_max_layers)
        for l in range(layer_idx + 1, last_layer):
            row = self._pred_matrix[l]
            if float(row.sum().item()) <= 0:
                continue
            k = min(self._prefetch_per_layer, row.shape[0])
            vals, ids = torch.topk(row, k=k)
            for v, e in zip(vals.tolist(), ids.tolist()):
                if v >= self._prefetch_score_threshold:
                    candidates.append((v, l, e))
        candidates.sort(reverse=True)
        if not candidates:
            return
        candidates = candidates[: self._prefetch_topk]
        by_layer: dict[int, list[int]] = defaultdict(list)
        for _v, l, e in candidates:
            by_layer[l].append(e)
        for l, eids in by_layer.items():
            lname = self._elmm._ordered_layers[l]
            cache = self._elmm._layer_caches.get(lname)
            # Conservative mode: avoid prefetch-triggered evictions.
            if cache is not None and not cache._free_slots:
                continue
            self._elmm.prefetch_experts(lname, eids)

    def on_access_batch(self, layer_id: int, unique_list: list[int]):
        self._step += 1
        self._step_protected[layer_id] = (self._step, set(int(eid) for eid in unique_list))
        for eid in unique_list:
            self._visit_freq[(layer_id, eid)] += 1
            self._total_visits += 1
            if 0 <= layer_id < self._num_layers and 0 <= eid < self._num_experts:
                self._freq_matrix[layer_id, eid] += 1.0
        self._priority_row_dirty.add(layer_id)
        self._maybe_periodic_candidate_refresh(layer_id)

    def select_victim(
        self,
        layer_name: str,
        layer_id: int,
        cache: Any,
        unique_set: set[int],
        incoming_eid: int,
    ) -> int | None:
        del layer_name, incoming_eid
        protected = set(unique_set)
        protected |= self._get_step_protected(layer_id)
        candidates = [eid for eid in cache._slot_map.keys() if eid not in protected]
        if not candidates:
            candidates = list(cache._slot_map.keys())
        if not candidates:
            return None

        state = self._candidate_cache.get(layer_id)
        if state is None or (self._step - state[0]) >= self._candidate_refresh_interval:
            self._refresh_candidate_cache(layer_id, cache)
            state = self._candidate_cache.get(layer_id)

        if state is not None:
            ts, cand_list, cursor = state
            n = len(cand_list)
            checked = 0
            while checked < n:
                idx = (cursor + checked) % n
                eid = cand_list[idx]
                if eid in cache._slot_map and eid not in protected:
                    self._candidate_cache[layer_id] = (ts, cand_list, (idx + 1) % n)
                    return eid
                checked += 1
            self._candidate_cache[layer_id] = (ts, cand_list, cursor)

        # Conservative fallback: plain LRU among non-required experts.
        for eid in cache._slot_map.keys():
            if eid not in protected:
                return eid
        return next(iter(cache._slot_map.keys()), None)

    def select_victims_batch(
        self,
        layer_name: str,
        layer_id: int,
        cache: Any,
        unique_set: set[int],
        incoming_eids: list[int],
    ) -> dict[int, int | None]:
        del layer_name
        plan: dict[int, int | None] = {}
        if cache is None or not incoming_eids:
            return plan
        if cache._free_slots:
            return plan

        state = self._candidate_cache.get(layer_id)
        if self._should_refresh_plan(layer_id, cache, incoming_eids):
            self._refresh_candidate_cache(layer_id, cache)
            state = self._candidate_cache.get(layer_id)

        ordered: list[int] = []
        protected = set(unique_set)
        protected |= self._get_step_protected(layer_id)
        if state is not None:
            _ts, cand_list, _cursor = state
            ordered = [eid for eid in cand_list if eid in cache._slot_map and eid not in protected]
        if not ordered:
            ordered = [eid for eid in cache._slot_map.keys() if eid not in protected]

        if not ordered:
            return plan

        victim_cursor = 0
        planned_victims: set[int] = set()
        for eid in incoming_eids:
            if eid in cache._slot_map or eid in protected:
                continue
            victim = None
            while victim_cursor < len(ordered):
                cand = ordered[victim_cursor]
                victim_cursor += 1
                if cand in planned_victims or cand in protected:
                    continue
                victim = cand
                break
            if victim is None:
                for cand in cache._slot_map.keys():
                    if cand in planned_victims or cand in protected:
                        continue
                    victim = cand
                    break
            plan[eid] = victim
            if victim is None:
                continue
            planned_victims.add(victim)
            protected.add(eid)
            self._protect_for_current_step(layer_id, [eid])

        return plan

    def install(self):
        self._elmm._external_on_access_batch = self.on_access_batch
        if self._enable_external_victim:
            self._elmm._external_select_victim = self.select_victim
            self._elmm._external_select_victims_batch = self.select_victims_batch
            self._elmm._external_on_load_expert = self.on_loaded_expert
        else:
            self._elmm._external_select_victim = None
            self._elmm._external_select_victims_batch = None
            self._elmm._external_on_load_expert = None

        for layer_idx, lname in enumerate(self._elmm._ordered_layers):
            module = self._elmm._patched_modules.get(lname)
            if module is None or not hasattr(module, "router"):
                continue
            router = module.router
            if not hasattr(router, "select_experts"):
                continue
            if getattr(router, "_moeinf_wrapped", False):
                continue

            orig = router.select_experts
            self._orig_select[lname] = orig
            ctrl = self

            def wrapped_select_experts(*args, **kwargs):
                hs = ctrl._extract_hidden_states(args, kwargs)
                ctrl._begin_new_sequence_if_needed(layer_idx, hs)

                topk_weights, topk_ids = orig(*args, **kwargs)
                if isinstance(topk_ids, torch.Tensor) and topk_ids.numel() > 0:
                    with torch.no_grad():
                        counts = torch.bincount(
                            topk_ids.reshape(-1), minlength=ctrl._num_experts
                        ).float().cpu()
                        ctrl._current_trace[layer_idx] += counts
                        ctrl._pred_matrix = ctrl._predict_future_matrix(layer_idx)
                        ctrl._priority_row_dirty.add(layer_idx)
                        ctrl._prefetch_from_prediction(layer_idx)
                return topk_weights, topk_ids

            router.select_experts = wrapped_select_experts
            router._moeinf_wrapped = True

        print(
            f"[MoE-Infinity] Activated: trace_capacity={self._capacity}, "
            f"prefetch_topk={self._prefetch_topk}, per_layer={self._prefetch_per_layer}, "
            f"external_victim={self._enable_external_victim}, "
            f"victim_refresh={self._victim_refresh_interval}, "
            f"plan_stride={self._plan_update_stride}, "
            f"neighbor_window={self._neighbor_window}",
            file=sys.stderr,
            flush=True,
        )


_moeinf_state: MoEInfinityController | None = None


def activate_moe_infinity(elmm_manager: Any) -> MoEInfinityController | None:
    global _moeinf_state
    if elmm_manager is None or not getattr(elmm_manager, "_installed", False):
        print("[MoE-Infinity] ELMM is not installed; skip activation", file=sys.stderr, flush=True)
        return None
    ctrl = MoEInfinityController(elmm_manager)
    ctrl.install()
    _moeinf_state = ctrl
    return ctrl
