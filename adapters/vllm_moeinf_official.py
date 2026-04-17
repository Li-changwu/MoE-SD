"""Pure vLLM MoE-Infinity (official-like) controller.

This module intentionally does NOT depend on ELMM internals.

Mapped core logic from official MoE-Infinity runtime:
1) Activation trace update (per-layer expert matrix accumulation)
2) Similar-trace future expert prediction (cosine distance)
3) Prediction-driven prefetch planning for future layers
4) Priority-style shadow eviction based on predicted demand + frequency

Because vLLM does not expose expert swap APIs, we simulate cache residency via
per-layer shadow sets and report policy-level metrics.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


def _row_normalize(x: torch.Tensor) -> torch.Tensor:
	denom = x.sum(dim=1, keepdim=True)
	denom = torch.where(denom > 0, denom, torch.ones_like(denom))
	return x / denom


@dataclass
class _TraceEntry:
	matrix: torch.Tensor
	access: int = 1


class VLLMMoEInfOfficialController:
	def __init__(self, model: Any):
		self._model = model

		self._trace_capacity = int(os.environ.get("MOE_INFINITY_TRACE_CAPACITY", "256"))
		self._prefetch_topk = int(os.environ.get("MOE_INFINITY_PREFETCH_TOPK", "4"))
		self._prefetch_per_layer = int(os.environ.get("MOE_INFINITY_PREFETCH_PER_LAYER", "1"))
		self._prefetch_max_layers = int(os.environ.get("MOE_INFINITY_PREFETCH_MAX_LAYERS", "2"))
		self._prefetch_min_step = int(os.environ.get("MOE_INFINITY_PREFETCH_MIN_STEP", "16"))
		self._prefetch_score_threshold = float(
			os.environ.get("MOE_INFINITY_PREFETCH_SCORE_THRESHOLD", "1e-3")
		)

		# Overhead-control knobs; default=1 keeps algorithm behavior unchanged.
		self._trace_update_stride = max(
			1, int(os.environ.get("MOE_INFINITY_TRACE_UPDATE_STRIDE", "1"))
		)
		self._predict_update_stride = max(
			1, int(os.environ.get("MOE_INFINITY_PREDICT_UPDATE_STRIDE", "1"))
		)
		self._shadow_policy_stride = max(
			1, int(os.environ.get("MOE_INFINITY_SHADOW_POLICY_STRIDE", "1"))
		)

		self._log_interval = int(os.environ.get("MOE_INFINITY_LOG_INTERVAL", "100"))
		self._score_refresh_interval = max(
			1, int(os.environ.get("MOE_INFINITY_SCORE_REFRESH_INTERVAL", "32"))
		)
		self._neighbor_window = max(
			0, int(os.environ.get("MOE_INFINITY_NEIGHBOR_WINDOW", "2"))
		)

		self._ordered_layer_names: list[str] = []
		self._modules: list[Any] = []
		self._routers: list[Any] = []
		self._orig_select: dict[str, Any] = {}

		self._num_layers = 0
		self._num_experts = 0

		self._trace_collection: list[_TraceEntry] = []
		self._current_trace = torch.zeros(0, 0, dtype=torch.float32)
		self._pred_matrix = torch.zeros(0, 0, dtype=torch.float32)
		self._freq_matrix = torch.zeros(0, 0, dtype=torch.float32)

		self._step = 0
		self._trace_updates = 0
		self._predict_updates = 0
		self._prefetch_planned = 0
		self._shadow_hits = 0
		self._shadow_misses = 0

		self._total_shadow_slots = int(os.environ.get("MOE_INFINITY_SHADOW_TOTAL_SLOTS", "128"))
		self._slots_by_layer: list[int] = []
		self._shadow_cache: dict[int, set[int]] = defaultdict(set)
		self._step_protected: dict[int, tuple[int, set[int]]] = {}

		self._priority_row_cache: dict[int, tuple[int, torch.Tensor]] = {}
		self._priority_row_dirty: set[int] = set()

	def _extract_hidden_states(self, args: tuple, kwargs: dict) -> torch.Tensor | None:
		if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
			return kwargs["hidden_states"]
		if args and isinstance(args[0], torch.Tensor):
			return args[0]
		return None

	def _init_shadow_slots(self):
		if self._num_layers <= 0:
			self._slots_by_layer = []
			return
		total = max(self._total_shadow_slots, self._num_layers)
		base = max(1, total // self._num_layers)
		self._slots_by_layer = [base] * self._num_layers
		rem = total - base * self._num_layers
		for i in range(rem):
			self._slots_by_layer[i % self._num_layers] += 1

	def _ensure_num_experts(self, required: int):
		if required <= self._num_experts:
			return
		new_n = max(required, max(8, self._num_experts * 2))
		old_n = self._num_experts
		self._num_experts = new_n

		def _expand(mat: torch.Tensor, rows: int) -> torch.Tensor:
			if rows <= 0:
				return torch.zeros(0, new_n, dtype=torch.float32)
			if mat.numel() == 0:
				return torch.zeros(rows, new_n, dtype=torch.float32)
			out = torch.zeros(rows, new_n, dtype=torch.float32)
			out[:, :old_n] = mat[:, :old_n]
			return out

		self._current_trace = _expand(self._current_trace, self._num_layers)
		self._pred_matrix = _expand(self._pred_matrix, self._num_layers)
		self._freq_matrix = _expand(self._freq_matrix, self._num_layers)

		for entry in self._trace_collection:
			expanded = torch.zeros(self._num_layers, new_n, dtype=torch.float32)
			if old_n > 0:
				expanded[:, :old_n] = entry.matrix[:, :old_n]
			entry.matrix = expanded

		self._priority_row_cache.clear()
		self._priority_row_dirty = set(range(self._num_layers))

	def _begin_new_sequence_if_needed(self, layer_idx: int, hidden_states: torch.Tensor | None):
		# Typical single-request benchmark: layer-0 prefill with T>1 means a new prompt.
		if layer_idx != 0 or hidden_states is None:
			return
		if hidden_states.ndim != 2 or hidden_states.shape[0] <= 1:
			return
		if self._current_trace.numel() > 0 and float(self._current_trace.sum().item()) > 0:
			self._push_current_trace()
			self._current_trace.zero_()

	def _push_current_trace(self):
		entry = _TraceEntry(matrix=self._current_trace.clone(), access=1)
		if len(self._trace_collection) < self._trace_capacity:
			self._trace_collection.append(entry)
			return
		min_i = 0
		min_v = self._trace_collection[0].access
		for i, e in enumerate(self._trace_collection):
			if e.access < min_v:
				min_v = e.access
				min_i = i
		self._trace_collection[min_i] = entry

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
		cur = self._get_step_protected(layer_id)
		cur.update(int(x) for x in eids)
		self._step_protected[layer_id] = (self._step, cur)

	def _compute_priority_row(self, layer_id: int) -> torch.Tensor:
		if self._num_layers <= 0 or self._num_experts <= 0:
			return torch.zeros(self._num_experts, dtype=torch.float32)
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
		return pred_row * freq_row

	def _get_priority_row(self, layer_id: int) -> torch.Tensor:
		cached = self._priority_row_cache.get(layer_id)
		if (
			cached is None
			or layer_id in self._priority_row_dirty
			or (self._step - cached[0]) >= self._score_refresh_interval
		):
			row = self._compute_priority_row(layer_id)
			self._priority_row_cache[layer_id] = (self._step, row)
			self._priority_row_dirty.discard(layer_id)
			return row
		return cached[1]

	def _pick_victim(self, layer_id: int, protected: set[int]) -> int | None:
		cache = self._shadow_cache[layer_id]
		if not cache:
			return None
		row = self._get_priority_row(layer_id)

		best = None
		best_score = float("inf")
		for eid in cache:
			if eid in protected:
				continue
			score = float(row[eid].item()) if row.numel() else 0.0
			if score < best_score:
				best_score = score
				best = eid
		if best is not None:
			return best

		# Conservative fallback if all are protected.
		return next(iter(cache), None)

	def _shadow_insert(self, layer_id: int, eid: int, protected: set[int]) -> bool:
		cache = self._shadow_cache[layer_id]
		if eid in cache:
			return True
		cap = self._slots_by_layer[layer_id] if layer_id < len(self._slots_by_layer) else 1
		if len(cache) >= cap:
			victim = self._pick_victim(layer_id, protected)
			if victim is None:
				return False
			cache.discard(victim)
		cache.add(eid)
		return True

	def _update_trace_from_ids(self, layer_idx: int, topk_ids: torch.Tensor):
		if self._step % self._trace_update_stride != 0:
			return
		flat = topk_ids.reshape(-1).to(torch.int64)
		if flat.numel() <= 0:
			return
		max_id = int(flat.max().item()) + 1
		self._ensure_num_experts(max_id)
		counts = torch.bincount(flat, minlength=self._num_experts).to(torch.float32).cpu()
		self._current_trace[layer_idx] += counts
		self._trace_updates += 1

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
			decay = -1.0 / (self._num_layers + 1) * (l - layer_idx) + 1.0
			pred[l] = (pred[l] + 1e-8) * decay
		return pred

	def _maybe_update_prediction(self, layer_idx: int):
		if self._step % self._predict_update_stride != 0:
			return
		self._pred_matrix = self._predict_future_matrix(layer_idx)
		self._predict_updates += 1
		self._priority_row_dirty.add(layer_idx)

	def _on_demand_access(self, layer_idx: int, topk_ids: torch.Tensor):
		unique_eids = [int(x) for x in torch.unique(topk_ids.reshape(-1)).tolist()]
		self._protect_for_current_step(layer_idx, unique_eids)
		protected = self._get_step_protected(layer_idx)
		cache = self._shadow_cache[layer_idx]

		for eid in unique_eids:
			if eid in cache:
				self._shadow_hits += 1
			else:
				self._shadow_misses += 1
				self._shadow_insert(layer_idx, eid, protected)
			if 0 <= layer_idx < self._num_layers and 0 <= eid < self._num_experts:
				self._freq_matrix[layer_idx, eid] += 1.0

		self._priority_row_dirty.add(layer_idx)

	def _plan_prefetch(self, layer_idx: int):
		if self._step < self._prefetch_min_step:
			return
		if self._step % self._shadow_policy_stride != 0:
			return
		if self._pred_matrix.numel() == 0:
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
					candidates.append((float(v), int(l), int(e)))

		if not candidates:
			return

		candidates.sort(reverse=True)
		candidates = candidates[: self._prefetch_topk]

		grouped: dict[int, list[int]] = defaultdict(list)
		for _score, l, e in candidates:
			grouped[l].append(e)

		for l, eids in grouped.items():
			protected = self._get_step_protected(l)
			for eid in eids:
				if eid in self._shadow_cache[l]:
					continue
				ok = self._shadow_insert(l, eid, protected)
				if ok:
					self._prefetch_planned += 1
					self._protect_for_current_step(l, [eid])

	def _log_stats(self):
		if self._log_interval <= 0 or self._step % self._log_interval != 0:
			return
		total = self._shadow_hits + self._shadow_misses
		hr = (self._shadow_hits / total) if total > 0 else 0.0
		print(
			f"[MoEInf-vLLM] step={self._step} shadow_hit_rate={hr:.3f} "
			f"prefetch_planned={self._prefetch_planned} "
			f"shadow_hits={self._shadow_hits} shadow_misses={self._shadow_misses} "
			f"trace_updates={self._trace_updates} predict_updates={self._predict_updates}",
			file=sys.stderr,
			flush=True,
		)

	def install(self):
		module_map = dict(self._model.named_modules())
		for name, module in module_map.items():
			router = getattr(module, "router", None)
			if router is None or not hasattr(router, "select_experts"):
				continue
			if getattr(router, "_moeinf_vllm_official_wrapped", False):
				continue
			self._ordered_layer_names.append(name)
			self._modules.append(module)
			self._routers.append(router)

		self._num_layers = len(self._routers)
		if self._num_layers == 0:
			print("[MoEInf-vLLM] no router found; skip install", file=sys.stderr, flush=True)
			return

		self._current_trace = torch.zeros(self._num_layers, 0, dtype=torch.float32)
		self._pred_matrix = torch.zeros(self._num_layers, 0, dtype=torch.float32)
		self._freq_matrix = torch.zeros(self._num_layers, 0, dtype=torch.float32)
		self._priority_row_dirty = set(range(self._num_layers))
		self._init_shadow_slots()

		for layer_idx, router in enumerate(self._routers):
			orig = router.select_experts
			lname = self._ordered_layer_names[layer_idx]
			self._orig_select[lname] = orig
			ctrl = self

			def wrapped_select_experts(
				*args,
				_orig=orig,
				_layer_idx=layer_idx,
				**kwargs,
			):
				topk_weights, topk_ids = _orig(*args, **kwargs)
				ctrl._step += 1

				hidden_states = ctrl._extract_hidden_states(args, kwargs)
				ctrl._begin_new_sequence_if_needed(_layer_idx, hidden_states)

				if isinstance(topk_ids, torch.Tensor) and topk_ids.numel() > 0:
					max_seen = int(topk_ids.max().item()) + 1
					ctrl._ensure_num_experts(max_seen)
					ctrl._update_trace_from_ids(_layer_idx, topk_ids)
					ctrl._on_demand_access(_layer_idx, topk_ids)
					ctrl._maybe_update_prediction(_layer_idx)
					ctrl._plan_prefetch(_layer_idx)

				ctrl._log_stats()
				return topk_weights, topk_ids

			router.select_experts = wrapped_select_experts
			router._moeinf_vllm_official_wrapped = True

		print(
			f"[MoEInf-vLLM] Installed official-like controller: layers={self._num_layers}, "
			f"trace_capacity={self._trace_capacity}, prefetch_topk={self._prefetch_topk}, "
			f"per_layer={self._prefetch_per_layer}, max_layers={self._prefetch_max_layers}, "
			f"trace_stride={self._trace_update_stride}, "
			f"predict_stride={self._predict_update_stride}, "
			f"policy_stride={self._shadow_policy_stride}",
			file=sys.stderr,
			flush=True,
		)


_state: VLLMMoEInfOfficialController | None = None


def activate_vllm_moeinf_official(model: Any) -> VLLMMoEInfOfficialController | None:
	global _state
	if model is None:
		return None
	ctrl = VLLMMoEInfOfficialController(model)
	ctrl.install()
	_state = ctrl
	return ctrl
