"""Pure vLLM AdapMoE (official-like) controller.

This controller intentionally does NOT depend on ELMM.
It reproduces AdapMoE's core runtime logic on top of vLLM router hooks:
1) sensitivity-threshold adaptive gating (top-2 -> top-1 fallback semantics)
2) DP-based per-layer cache-size planning (shadow policy only)
3) next-layer gate based check-then-prefetch behavior (shadow simulation)

Because vLLM native runtime does not expose expert swap APIs, this module uses
shadow residency sets for faithful policy-level reproduction and instrumentation.
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


class VLLMAdapMoEOfficialController:
	def __init__(self, model: Any):
		self._model = model
		self._prefetch_horizon = int(os.environ.get("ADAPMOE_PREFETCH_HORIZON", "1"))
		self._prefetch_topk = int(os.environ.get("ADAPMOE_PREFETCH_TOPK", "2"))
		self._enable_adap_gate = os.environ.get("ADAPMOE_ADAPTGATE", "1") == "1"
		self._enable_dp = os.environ.get("ADAPMOE_DP_ENABLE", "1") == "1"
		# Official run.py uses --size=64 as default main cache budget.
		self._total_slots = int(os.environ.get("ADAPMOE_SHADOW_TOTAL_SLOTS", "64"))
		self._log_interval = int(os.environ.get("ADAPMOE_LOG_INTERVAL", "100"))
		self._debug_diag = os.environ.get("ADAPMOE_DEBUG_DIAG", "0") == "1"
		self._debug_max_steps = int(os.environ.get("ADAPMOE_DEBUG_STEPS", "40"))
		self._debug_layer = int(os.environ.get("ADAPMOE_DEBUG_LAYER", "0"))

		self._ordered_layer_names: list[str] = []
		self._modules: list[Any] = []
		self._routers: list[Any] = []
		self._orig_select: dict[str, Any] = {}
		self._gate_by_layer: list[Any] = []
		self._gate_src_by_layer: list[str] = []

		self._threshold_by_layer: list[float] = []
		self._slots_by_layer: list[int] = []
		self._shadow_cache: dict[int, set[int]] = defaultdict(set)
		self._shadow_lru: dict[int, list[int]] = defaultdict(list)

		self._step = 0
		self._top1_fallback_hits = defaultdict(int)
		self._prefetch_planned = 0
		self._shadow_hits = 0
		self._shadow_misses = 0

	def _debug_should_log(self, layer_idx: int) -> bool:
		if not self._debug_diag:
			return False
		if self._step > self._debug_max_steps:
			return False
		return self._debug_layer < 0 or layer_idx == self._debug_layer

	def _debug_print(self, msg: str):
		if not self._debug_diag:
			return
		print(f"[AdapMoE-vLLM][diag] {msg}", file=sys.stderr, flush=True)

	def _as_callable_gate(self, candidate: Any) -> Any | None:
		if candidate is None:
			return None
		if callable(candidate):
			return candidate
		return None

	def _resolve_gate_for_layer(
		self,
		layer_idx: int,
		module_map: dict[str, Any],
	) -> tuple[Any | None, str]:
		lname = self._ordered_layer_names[layer_idx]
		mod = self._modules[layer_idx]

		candidates: list[tuple[Any, str]] = [
			(getattr(mod, "gate", None), "module.gate"),
			(getattr(mod, "_gate", None), "module._gate"),
			(getattr(getattr(mod, "router", None), "gate", None), "module.router.gate"),
			(getattr(getattr(mod, "router", None), "_gate", None), "module.router._gate"),
		]

		parent_name = lname
		if lname.endswith(".experts"):
			parent_name = lname[: -len(".experts")]
		elif "." in lname:
			parent_name = lname.rsplit(".", 1)[0]

		parent = module_map.get(parent_name)
		if parent is not None:
			candidates.extend(
				[
					(getattr(parent, "gate", None), f"parent({parent_name}).gate"),
					(getattr(parent, "_gate", None), f"parent({parent_name})._gate"),
				]
			)

		for cand, src in candidates:
			gate = self._as_callable_gate(cand)
			if gate is not None:
				return gate, src
		return None, "none"

	def _extract_hidden_states(self, args: tuple, kwargs: dict) -> torch.Tensor | None:
		if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
			return kwargs["hidden_states"]
		if args and isinstance(args[0], torch.Tensor):
			return args[0]
		return None

	def _flatten_hidden_states(self, hidden_states: torch.Tensor | None) -> torch.Tensor | None:
		if not isinstance(hidden_states, torch.Tensor):
			return None
		if hidden_states.ndim == 1:
			return hidden_states.view(1, -1)
		if hidden_states.ndim >= 2:
			return hidden_states.view(-1, hidden_states.shape[-1])
		return None

	def _init_thresholds(self, nlayers: int):
		threshold_base = float(os.environ.get("ADAPMOE_THRESHOLD_BASE", "0.005"))
		thresholds = [
			math.sqrt(max(threshold_base, 0.0) / max(w, 1e-8)) for w in _HESSIAN_WEIGHTS
		]
		self._threshold_by_layer = _resample_list(thresholds, nlayers)

	def _init_shadow_slots(self, nlayers: int):
		if nlayers <= 0:
			self._slots_by_layer = []
			return
		if not self._enable_dp:
			base = max(1, self._total_slots // nlayers)
			self._slots_by_layer = [base] * nlayers
			return

		dp_sizes = _dp_cache_sizes(max(self._total_slots, nlayers), self._enable_adap_gate)
		target = _resample_list([float(x) for x in dp_sizes], nlayers)
		slots = [max(1, int(round(x))) for x in target]

		diff = max(self._total_slots, nlayers) - sum(slots)
		if diff > 0:
			for i in range(diff):
				slots[i % len(slots)] += 1
		elif diff < 0:
			need = -diff
			idx = 0
			while need > 0 and idx < len(slots) * 4:
				j = idx % len(slots)
				if slots[j] > 1:
					slots[j] -= 1
					need -= 1
				idx += 1
		self._slots_by_layer = slots

	def _shadow_touch(self, layer_idx: int, eid: int):
		cache = self._shadow_cache[layer_idx]
		lru = self._shadow_lru[layer_idx]
		if eid in cache:
			self._shadow_hits += 1
			if eid in lru:
				lru.remove(eid)
			lru.append(eid)
			return
		self._shadow_misses += 1
		cache.add(eid)
		lru.append(eid)

		cap = self._slots_by_layer[layer_idx] if layer_idx < len(self._slots_by_layer) else 1
		while len(cache) > cap:
			victim = lru.pop(0)
			if victim in cache:
				cache.remove(victim)

	def _predict_and_prefetch_next_layers(self, layer_idx: int, hidden_states: torch.Tensor):
		if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
			if self._debug_should_log(layer_idx):
				self._debug_print(
					f"step={self._step} layer={layer_idx} prefetch_skip: "
					f"hidden_shape={tuple(hidden_states.shape)}"
				)
			return
		for h in range(1, self._prefetch_horizon + 1):
			nxt_idx = layer_idx + h
			if nxt_idx >= len(self._modules):
				if self._debug_should_log(layer_idx):
					self._debug_print(
						f"step={self._step} layer={layer_idx} prefetch_stop: "
						f"next_idx={nxt_idx} out_of_range"
					)
				break
			gate = self._gate_by_layer[nxt_idx] if nxt_idx < len(self._gate_by_layer) else None
			gate_src = self._gate_src_by_layer[nxt_idx] if nxt_idx < len(self._gate_src_by_layer) else "none"
			if gate is None:
				if self._debug_should_log(layer_idx):
					self._debug_print(
						f"step={self._step} layer={layer_idx} prefetch_gate_missing: "
						f"next_layer={nxt_idx} gate_src={gate_src}"
					)
				continue
			try:
				with torch.no_grad():
					gate_out = gate(hidden_states)
					logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
					probs = torch.softmax(logits, dim=1, dtype=torch.float)
					k = min(max(self._prefetch_topk, 1), probs.shape[1])
					top_ids = torch.topk(probs, k, dim=-1).indices[0].tolist()
					if self._debug_should_log(layer_idx):
						self._debug_print(
							f"step={self._step} layer={layer_idx} prefetch_gate_ok: "
							f"next_layer={nxt_idx} gate_src={gate_src} "
							f"logits_shape={tuple(logits.shape)} "
							f"top_ids={top_ids}"
						)

				target = None
				cache = self._shadow_cache[nxt_idx]
				for eid in top_ids:
					if eid not in cache:
						target = int(eid)
						break
				if target is not None:
					self._prefetch_planned += 1
					self._shadow_touch(nxt_idx, target)
					if self._debug_should_log(layer_idx):
						self._debug_print(
							f"step={self._step} layer={layer_idx} prefetch_plan: "
							f"next_layer={nxt_idx} target={target} cache_size={len(cache)}"
						)
				elif self._debug_should_log(layer_idx):
					self._debug_print(
						f"step={self._step} layer={layer_idx} prefetch_no_target: "
						f"next_layer={nxt_idx} cache_size={len(cache)}"
					)
			except Exception:
				if self._debug_should_log(layer_idx):
					self._debug_print(
						f"step={self._step} layer={layer_idx} prefetch_exception: next_layer={nxt_idx}"
					)
				continue

	def _log_stats(self):
		if self._log_interval <= 0 or self._step % self._log_interval != 0:
			return
		total = self._shadow_hits + self._shadow_misses
		hr = (self._shadow_hits / total) if total > 0 else 0.0
		top1_hits = sum(self._top1_fallback_hits.values())
		print(
			f"[AdapMoE-vLLM] step={self._step} shadow_hit_rate={hr:.3f} "
			f"top1_fallback={top1_hits} prefetch_planned={self._prefetch_planned} "
			f"shadow_hits={self._shadow_hits} shadow_misses={self._shadow_misses}",
			file=sys.stderr,
			flush=True,
		)

	def install(self):
		module_map = dict(self._model.named_modules())
		for name, module in module_map.items():
			router = getattr(module, "router", None)
			if router is None or not hasattr(router, "select_experts"):
				continue
			if getattr(router, "_adapmoe_vllm_official_wrapped", False):
				continue
			self._ordered_layer_names.append(name)
			self._modules.append(module)
			self._routers.append(router)

		nlayers = len(self._routers)
		if nlayers == 0:
			print("[AdapMoE-vLLM] no router found; skip install", file=sys.stderr, flush=True)
			return

		self._init_thresholds(nlayers)
		self._init_shadow_slots(nlayers)

		self._gate_by_layer = []
		self._gate_src_by_layer = []
		for layer_idx in range(nlayers):
			gate, src = self._resolve_gate_for_layer(layer_idx, module_map)
			self._gate_by_layer.append(gate)
			self._gate_src_by_layer.append(src)
			if self._debug_diag and (self._debug_layer < 0 or self._debug_layer == layer_idx):
				self._debug_print(
					f"install layer={layer_idx} name={self._ordered_layer_names[layer_idx]} gate_src={src}"
				)

		for layer_idx, router in enumerate(self._routers):
			orig = router.select_experts
			lname = self._ordered_layer_names[layer_idx]
			self._orig_select[lname] = orig
			threshold = self._threshold_by_layer[layer_idx]
			ctrl = self

			def wrapped_select_experts(
				*args,
				_orig=orig,
				_layer_idx=layer_idx,
				_lname=lname,
				_threshold=threshold,
				**kwargs,
			):
				topk_weights, topk_ids = _orig(*args, **kwargs)
				ctrl._step += 1
				hidden_states = ctrl._extract_hidden_states(args, kwargs)
				flat_hidden_states = ctrl._flatten_hidden_states(hidden_states)
				single_token_route = (
					isinstance(topk_weights, torch.Tensor)
					and topk_weights.ndim == 2
					and topk_weights.shape[0] == 1
				)
				single_token_hidden = (
					flat_hidden_states is not None and flat_hidden_states.shape[0] == 1
				)
				gate_hidden = (
					flat_hidden_states[:1]
					if flat_hidden_states is not None and flat_hidden_states.shape[0] >= 1
					else None
				)

				if ctrl._debug_should_log(_layer_idx):
					w_shape = tuple(topk_weights.shape) if isinstance(topk_weights, torch.Tensor) else None
					id_shape = tuple(topk_ids.shape) if isinstance(topk_ids, torch.Tensor) else None
					if isinstance(topk_weights, torch.Tensor) and topk_weights.ndim == 2 and topk_weights.shape[0] >= 1:
						row0 = topk_weights[0].detach().float().cpu()
						w_min = float(row0.min().item())
						w_max = float(row0.max().item())
						w_mean = float(row0.mean().item())
						w_vals = [float(x) for x in row0.tolist()]
					else:
						w_min = float("nan")
						w_max = float("nan")
						w_mean = float("nan")
						w_vals = []
					if isinstance(topk_ids, torch.Tensor) and topk_ids.ndim == 2 and topk_ids.shape[0] >= 1:
						id_vals = [int(x) for x in topk_ids[0].detach().cpu().tolist()]
					else:
						id_vals = []
					ctrl._debug_print(
						f"step={ctrl._step} layer={_layer_idx} topk: "
						f"weights_shape={w_shape} ids_shape={id_shape} "
						f"w_row0={w_vals} ids_row0={id_vals} "
						f"w_row0_min={w_min:.6f} w_row0_max={w_max:.6f} w_row0_mean={w_mean:.6f} "
						f"single_route={single_token_route} single_hidden={single_token_hidden} "
						f"gate_hidden_shape={tuple(gate_hidden.shape) if gate_hidden is not None else None}"
					)

				if (
					ctrl._enable_adap_gate
					and isinstance(topk_weights, torch.Tensor)
					and isinstance(topk_ids, torch.Tensor)
					and topk_weights.ndim == 2
					and topk_weights.shape[0] >= 1
					and topk_weights.shape[1] >= 2
					and (single_token_route or single_token_hidden)
					and _threshold > 0
				):
					# Match official logic: single-token path checks token-0 second weight.
					if bool((topk_weights[0, 1] < _threshold).item()):
						topk_weights = topk_weights.clone()
						topk_ids = topk_ids.clone()
						# Keep shape stable while enforcing top-1 fallback semantics.
						topk_ids[0, 1:] = topk_ids[0, :1]
						topk_weights[0, 1:] = 0
						denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-8)
						topk_weights = topk_weights / denom
						ctrl._top1_fallback_hits[_lname] += 1

				if isinstance(topk_ids, torch.Tensor) and topk_ids.numel() > 0:
					for eid in topk_ids.reshape(-1).tolist():
						ctrl._shadow_touch(_layer_idx, int(eid))

				if gate_hidden is not None:
					ctrl._predict_and_prefetch_next_layers(_layer_idx, gate_hidden)

				ctrl._log_stats()
				return topk_weights, topk_ids

			router.select_experts = wrapped_select_experts
			router._adapmoe_vllm_official_wrapped = True

		print(
			f"[AdapMoE-vLLM] Installed official-like controller: layers={nlayers}, "
			f"adap_gate={self._enable_adap_gate}, total_slots={self._total_slots}",
			file=sys.stderr,
			flush=True,
		)


_state: VLLMAdapMoEOfficialController | None = None


def activate_vllm_adapmoe_official(model: Any) -> VLLMAdapMoEOfficialController | None:
	global _state
	if model is None:
		return None
	ctrl = VLLMAdapMoEOfficialController(model)
	ctrl.install()
	_state = ctrl
	return ctrl
