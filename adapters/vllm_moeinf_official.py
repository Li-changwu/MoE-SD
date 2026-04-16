"""Pure vLLM MoE-Infinity (official-like) controller.

This controller intentionally does NOT depend on ELMM.
It mirrors official MoE-Infinity core algorithmic logic on top of vLLM routing:
1) expert activation tracing
2) similar-trace prediction with layer decay
3) activation-aware prefetch candidate generation
4) priority-based cache policy (shadow cache)

Because vLLM native runtime does not expose expert-level swap APIs, this module
maintains a shadow cache for faithful policy reproduction and instrumentation.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
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


class VLLMMoEInfinityOfficialController:
    def __init__(self, model: Any):
        self._model = model
        self._capacity = int(os.environ.get("MOE_INFINITY_TRACE_CAPACITY", "256"))
        self._prefetch_topk = int(os.environ.get("MOE_INFINITY_PREFETCH_TOPK", "4"))
        self._prefetch_per_layer = int(os.environ.get("MOE_INFINITY_PREFETCH_PER_LAYER", "1"))
        self._prefetch_max_layers = int(os.environ.get("MOE_INFINITY_PREFETCH_MAX_LAYERS", "2"))
        self._prefetch_score_threshold = float(
            os.environ.get("MOE_INFINITY_PREFETCH_SCORE_THRESHOLD", "1e-3")
        )
        self._shadow_cache_slots = int(
            os.environ.get("MOE_INFINITY_SHADOW_CACHE_SLOTS", "24")
        )
        self._log_interval = int(os.environ.get("MOE_INFINITY_LOG_INTERVAL", "100"))

        self._ordered_layer_names: list[str] = []
        self._routers: list[Any] = []
        self._orig_select: dict[str, Any] = {}

        self._num_layers = 0
        self._num_experts = 0
        self._step = 0

        self._trace_collection: list[_TraceEntry] = []
        self._current_trace: torch.Tensor | None = None
        self._pred_matrix: torch.Tensor | None = None

        self._layer_decay = lambda x, l, L: -1.0 / (L + 1) * (x - l) + 1.0

        # Shadow-cache state (policy reproduction only).
        self._shadow_cache: dict[int, set[int]] = defaultdict(set)
        self._expert_freq: Counter[tuple[int, int]] = Counter()
        self._shadow_hits = 0
        self._shadow_misses = 0
        self._shadow_evictions = 0

    def _extract_hidden_states(self, args: tuple, kwargs: dict) -> torch.Tensor | None:
        if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            return kwargs["hidden_states"]
        if args and isinstance(args[0], torch.Tensor):
            return args[0]
        return None

    def _begin_new_sequence_if_needed(self, layer_idx: int, hidden_states: torch.Tensor | None):
        if layer_idx != 0 or hidden_states is None:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[0] <= 1:
            return
        if self._current_trace is not None and float(self._current_trace.sum().item()) > 0:
            self._push_current_trace()
            self._current_trace.zero_()

    def _push_current_trace(self):
        if self._current_trace is None:
            return
        entry = _TraceEntry(matrix=self._current_trace.clone(), access=1)
        if len(self._trace_collection) < self._capacity:
            self._trace_collection.append(entry)
            return
        min_i = 0
        min_v = self._trace_collection[0].access
        for i, e in enumerate(self._trace_collection):
            if e.access < min_v:
                min_i = i
                min_v = e.access
        self._trace_collection[min_i] = entry

    def _predict_future_matrix(self, layer_idx: int) -> torch.Tensor:
        if self._current_trace is None or not self._trace_collection:
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

    def _official_priority_score(self, layer_id: int) -> torch.Tensor:
        # Reproduce official multiplicative form:
        # total_score = topo_expert_score * decoder_matrix * frequency_score
        assert self._current_trace is not None
        assert self._pred_matrix is not None

        freq = torch.zeros_like(self._current_trace)
        for (l, e), v in self._expert_freq.items():
            if 0 <= l < self._num_layers and 0 <= e < self._num_experts:
                freq[l, e] = float(v)
        if float(freq.sum().item()) <= 0:
            freq.fill_(1.0)
        freq = freq / torch.clamp(freq.sum(), min=1.0) + 1e-6

        topo = torch.zeros_like(self._current_trace)
        for l in range(self._num_layers):
            if l <= layer_id:
                topo[l, :] = 1.0
            else:
                topo[l, :] = self._layer_decay(l, layer_id, self._num_layers)
        topo = topo / torch.clamp(topo.sum(), min=1.0) + 1e-6

        decoder = self._pred_matrix.clone()
        if float(decoder.sum().item()) <= 0:
            decoder.fill_(1.0)
        for l in range(self._num_layers):
            s = float(decoder[l].sum().item())
            if s <= 0:
                decoder[l].fill_(1.0)
            decoder[l] = decoder[l] / torch.clamp(decoder[l].sum(), min=1.0)
        decoder = decoder / torch.clamp(decoder.sum(), min=1.0) + 1e-6

        return topo * decoder * freq

    def _prefetch_candidates(self, layer_idx: int) -> dict[int, list[int]]:
        if self._pred_matrix is None:
            return {}
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
        candidates = candidates[: self._prefetch_topk]
        by_layer: dict[int, list[int]] = defaultdict(list)
        for _v, l, e in candidates:
            by_layer[l].append(e)
        return by_layer

    def _shadow_cache_update(self, layer_id: int, current_eids: set[int]):
        cache = self._shadow_cache[layer_id]
        for eid in current_eids:
            if eid in cache:
                self._shadow_hits += 1
            else:
                self._shadow_misses += 1

        # Protect current step experts and predicted near-future experts.
        protected = set(current_eids)
        for _, eids in self._prefetch_candidates(layer_id).items():
            protected.update(eids)

        # Insert current experts.
        for eid in current_eids:
            cache.add(eid)

        # Evict until capacity satisfied, following priority ascending.
        if len(cache) <= self._shadow_cache_slots:
            return
        score = self._official_priority_score(layer_id)
        while len(cache) > self._shadow_cache_slots:
            victims = [
                eid
                for eid in cache
                if eid not in protected and 0 <= eid < self._num_experts
            ]
            if not victims:
                victims = list(cache)
            victim = min(victims, key=lambda e: float(score[layer_id, e].item()))
            cache.remove(victim)
            self._shadow_evictions += 1

    def _log_stats(self):
        if self._log_interval <= 0 or self._step % self._log_interval != 0:
            return
        total = self._shadow_hits + self._shadow_misses
        hr = (self._shadow_hits / total) if total > 0 else 0.0
        print(
            f"[MoEInf-vLLM] step={self._step} shadow_hit_rate={hr:.3f} "
            f"hits={self._shadow_hits} misses={self._shadow_misses} "
            f"evictions={self._shadow_evictions}",
            file=sys.stderr,
            flush=True,
        )

    def install(self):
        # Collect MoE routers directly from vLLM model modules.
        for name, module in self._model.named_modules():
            router = getattr(module, "router", None)
            if router is None or not hasattr(router, "select_experts"):
                continue
            if getattr(router, "_moeinf_vllm_official_wrapped", False):
                continue
            self._ordered_layer_names.append(name)
            self._routers.append(router)

        self._num_layers = len(self._routers)
        if self._num_layers == 0:
            print("[MoEInf-vLLM] no router found; skip install", file=sys.stderr, flush=True)
            return

        # Infer num_experts from first router call later if unavailable.
        self._num_experts = int(os.environ.get("MOE_INFINITY_NUM_EXPERTS", "0"))
        if self._num_experts <= 0:
            self._num_experts = 256

        self._current_trace = torch.zeros(self._num_layers, self._num_experts, dtype=torch.float32)
        self._pred_matrix = torch.zeros_like(self._current_trace)

        for layer_idx, router in enumerate(self._routers):
            orig = router.select_experts
            lname = self._ordered_layer_names[layer_idx]
            self._orig_select[lname] = orig
            ctrl = self

            def wrapped_select_experts(*args, **kwargs):
                hs = ctrl._extract_hidden_states(args, kwargs)
                ctrl._begin_new_sequence_if_needed(layer_idx, hs)

                topk_weights, topk_ids = orig(*args, **kwargs)
                if isinstance(topk_ids, torch.Tensor) and topk_ids.numel() > 0:
                    with torch.no_grad():
                        local_max = int(topk_ids.max().item()) + 1
                        if local_max > ctrl._num_experts:
                            # Grow trace matrices if model experts exceed default.
                            pad = local_max - ctrl._num_experts
                            assert ctrl._current_trace is not None
                            assert ctrl._pred_matrix is not None
                            ctrl._current_trace = torch.nn.functional.pad(ctrl._current_trace, (0, pad))
                            ctrl._pred_matrix = torch.nn.functional.pad(ctrl._pred_matrix, (0, pad))
                            ctrl._num_experts = local_max

                        flat = topk_ids.reshape(-1).cpu()
                        counts = torch.bincount(flat, minlength=ctrl._num_experts).float()
                        assert ctrl._current_trace is not None
                        ctrl._current_trace[layer_idx] += counts
                        ctrl._pred_matrix = ctrl._predict_future_matrix(layer_idx)

                        uniq = set(int(x) for x in flat.unique().tolist())
                        for eid in uniq:
                            ctrl._expert_freq[(layer_idx, eid)] += 1
                        ctrl._step += 1
                        ctrl._shadow_cache_update(layer_idx, uniq)
                        ctrl._log_stats()
                return topk_weights, topk_ids

            router.select_experts = wrapped_select_experts
            router._moeinf_vllm_official_wrapped = True

        print(
            f"[MoEInf-vLLM] Installed official-like controller: layers={self._num_layers}, "
            f"trace_capacity={self._capacity}, shadow_slots={self._shadow_cache_slots}",
            file=sys.stderr,
            flush=True,
        )


_state: VLLMMoEInfinityOfficialController | None = None


def activate_vllm_moeinf_official(model: Any) -> VLLMMoEInfinityOfficialController | None:
    global _state
    if model is None:
        return None
    ctrl = VLLMMoEInfinityOfficialController(model)
    ctrl.install()
    _state = ctrl
    return ctrl
