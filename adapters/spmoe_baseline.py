"""
SP-MoE Baseline Reproduction
==============================
Reproduces the core ideas from:
  "SP-MoE: Speculative Decoding and Prefetching for Accelerating
   MoE-based Model Inference" (Chen et al., arXiv 2510.10302)

Adapted to BriskMoE's ELMM infrastructure for fair comparison.

SP-MoE key components:
  1. Cross-model gating predictor: uses draft hidden states + target gate
     weights to predict which experts verification will need.
  2. Cutoff layer policy: only prefetch layers 0..L, determined by an
     analytical latency model bounding I/O within draft duration.
  3. Pipelined prefetcher: async worker thread with batched I/O.
  4. LRU cache + scratchpad execution (no Pool-Direct, no TASER).

Adaptation notes:
  - EAGLE3 is not a full replica of the target model (unlike Mistral->Mixtral).
    We use the target model's own gate weights applied to the last target
    hidden states (available from the previous verification pass) as a proxy,
    matching SP-MoE's gating-based prediction.
  - Cutoff layer L is computed per SP-MoE's analytical model:
    max L s.t. L * k * t_IO <= T_draft  AND  L * k * M_expert <= M_free.

Env vars:
  SPMOE_ENABLE=1        -- activate SP-MoE predictor + cutoff + pipeline
  SPMOE_CUTOFF=-1       -- auto-compute cutoff layer (or set integer)
  SPMOE_TOP_K=8         -- experts to prefetch per layer
  SPMOE_DRAFT_MS=30     -- estimated draft duration in ms
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SPMoEConfig:
    """Configuration for SP-MoE baseline reproduction."""
    top_k: int = 8
    cutoff_layer: int = -1          # -1 = auto
    draft_duration_ms: float = 30.0
    t_io_per_expert_ms: float = 0.79  # PCIe 4.0 x16, profiled
    expert_size_bytes: int = 0        # 0 = auto-detect from model
    prefetch_memory_bytes: int = 0    # 0 = use cache budget
    batched_io: bool = True
    use_worker_thread: bool = True


# ---------------------------------------------------------------------------
# Gating Predictor (SP-MoE Section 3.2)
# ---------------------------------------------------------------------------

class SPMoEGatingPredictor:
    """
    Cross-model gating predictor.

    Uses the target model's gate weights to predict expert activations from
    hidden states available during the draft phase.  Since EAGLE3 is a
    single-layer lightweight model (not a full transformer replica), we use
    the target model's last hidden states from the previous verification pass.
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self._gate_weights: dict[str, torch.Tensor] = {}
        self._layer_order: list[str] = []
        self._num_experts_per_layer: dict[str, int] = {}

    def register_layer(self, layer_name: str, gate_weight: torch.Tensor,
                       num_experts: int):
        """Register a target layer's gate weights for prediction."""
        self._gate_weights[layer_name] = gate_weight.to(
            device=self.device, dtype=torch.float32
        )
        self._num_experts_per_layer[layer_name] = num_experts
        if layer_name not in self._layer_order:
            self._layer_order.append(layer_name)

    def predict(self, hidden_states: torch.Tensor, layer_name: str,
                top_k: int = 8) -> list[int]:
        """Predict top-k experts for a target layer using gating scores."""
        gate_w = self._gate_weights.get(layer_name)
        if gate_w is None:
            return []
        with torch.no_grad():
            h = hidden_states.float()
            logits = torch.matmul(h, gate_w.t())  # [tokens, experts]
            expert_scores = logits.max(dim=0).values
            k = min(top_k, expert_scores.shape[0])
            _, top_indices = expert_scores.topk(k)
            return top_indices.tolist()

    def predict_all_layers(self, hidden_states: torch.Tensor,
                           top_k: int = 8, cutoff: int = -1
                           ) -> dict[str, list[int]]:
        """Predict experts for all layers up to cutoff."""
        predictions: dict[str, list[int]] = {}
        for i, layer_name in enumerate(self._layer_order):
            if 0 <= cutoff <= i:
                break
            predicted = self.predict(hidden_states, layer_name, top_k)
            if predicted:
                predictions[layer_name] = predicted
        return predictions


# ---------------------------------------------------------------------------
# Cutoff Layer Computation (SP-MoE Section 3.2)
# ---------------------------------------------------------------------------

def compute_cutoff_layer(
    num_layers: int,
    top_k: int,
    t_io_ms: float,
    draft_duration_ms: float,
    expert_size_bytes: int,
    available_memory_bytes: int,
) -> int:
    """
    Compute optimal cutoff layer L for SP-MoE prefetching.
    Constraints: L*k*M_expert <= M_avail  AND  L*k*t_IO <= T_draft
    """
    if top_k <= 0 or t_io_ms <= 0 or expert_size_bytes <= 0:
        return num_layers
    max_l_memory = int(available_memory_bytes / (top_k * expert_size_bytes))
    max_l_time = int(draft_duration_ms / (top_k * t_io_ms))
    cutoff = min(max_l_memory, max_l_time, num_layers)
    return max(cutoff, 1)


# ---------------------------------------------------------------------------
# Pipelined Prefetcher with Worker Thread (SP-MoE Section 3.3)
# ---------------------------------------------------------------------------

@dataclass
class PrefetchTask:
    layer_name: str
    expert_id: int
    priority: float = 0.0


class SPMoEPrefetcher:
    """
    Async prefetch worker with batched I/O.

    A dedicated worker thread processes prefetch tasks from a priority queue,
    delegating to ELMM's existing prefetch_experts() API for H2D transfers
    on a dedicated CUDA stream.
    """

    def __init__(self, elmm_manager: Any, config: SPMoEConfig,
                 device: torch.device = torch.device("cuda")):
        self._elmm = elmm_manager
        self._config = config
        self._device = device
        self._task_queue: deque[PrefetchTask] = deque()
        self._queue_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._total_prefetched = 0
        self._total_cache_hits = 0
        self._total_batches = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="spmoe-prefetcher"
        )
        self._worker.start()

    def stop(self):
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def submit_tasks(self, tasks: list[PrefetchTask]):
        with self._queue_lock:
            tasks.sort(key=lambda t: t.priority, reverse=True)
            self._task_queue.extend(tasks)

    def clear(self):
        with self._queue_lock:
            self._task_queue.clear()

    def _worker_loop(self):
        batch_size = 4 if self._config.batched_io else 1
        while self._running:
            batch: list[PrefetchTask] = []
            with self._queue_lock:
                for _ in range(batch_size):
                    if self._task_queue:
                        batch.append(self._task_queue.popleft())
                    else:
                        break
            if not batch:
                time.sleep(0.0001)
                continue
            self._execute_batch(batch)
            self._total_batches += 1

    def _execute_batch(self, batch: list[PrefetchTask]):
        """Execute batch using ELMM's prefetch_experts() API."""
        elmm = self._elmm
        layer_tasks: dict[str, list[int]] = {}
        for task in batch:
            cache = elmm._layer_caches.get(task.layer_name)
            if cache is None:
                continue
            if cache.contains(task.expert_id):
                self._total_cache_hits += 1
                continue
            layer_tasks.setdefault(task.layer_name, []).append(task.expert_id)

        for layer_name, expert_ids in layer_tasks.items():
            elmm.prefetch_experts(layer_name, expert_ids)
            self._total_prefetched += len(expert_ids)

    def get_stats(self) -> dict:
        return {
            "total_prefetched": self._total_prefetched,
            "total_cache_hits": self._total_cache_hits,
            "total_batches": self._total_batches,
            "queue_size": len(self._task_queue),
        }


# ---------------------------------------------------------------------------
# SP-MoE Draft Hook (replaces DraftPrefetchHook)
# ---------------------------------------------------------------------------

class SPMoEDraftHook:
    """
    Hooks into EAGLE3's propose() to trigger SP-MoE draft-stage prefetching.

    On each draft completion:
      1. Retrieve target model's last hidden states
      2. Run gating predictor for layers 0..cutoff
      3. Submit prefetch tasks to the async worker
    """

    def __init__(self, elmm_manager: Any, predictor: SPMoEGatingPredictor,
                 prefetcher: SPMoEPrefetcher, config: SPMoEConfig):
        self._elmm = elmm_manager
        self._predictor = predictor
        self._prefetcher = prefetcher
        self._config = config
        self._cutoff = config.cutoff_layer
        self._installed = False
        self._original_propose = None
        self._draft_rounds = 0
        self._last_hidden_states: Optional[torch.Tensor] = None

    def install(self) -> bool:
        """Monkey-patch EagleProposer.propose to trigger SP-MoE prefetch."""
        try:
            from vllm.v1.spec_decode.eagle import EagleProposer
        except ImportError:
            print("[SP-MoE] Cannot import EagleProposer, skipping",
                  file=sys.stderr, flush=True)
            return False

        self._original_propose = EagleProposer.propose
        hook = self

        def patched_propose(self_proposer, *args, **kwargs):
            result = hook._original_propose(self_proposer, *args, **kwargs)
            hook._on_draft_complete(self_proposer)
            return result

        EagleProposer.propose = patched_propose
        self._installed = True
        print(f"[SP-MoE] Installed draft hook (cutoff={self._cutoff})",
              file=sys.stderr, flush=True)
        return True

    def set_hidden_states(self, hidden_states: torch.Tensor):
        """Called from the target model forward pass to cache hidden states."""
        self._last_hidden_states = hidden_states

    def _on_draft_complete(self, proposer: Any):
        """Called after each EAGLE3 draft proposal."""
        self._draft_rounds += 1
        if self._last_hidden_states is None:
            return

        predictions = self._predictor.predict_all_layers(
            self._last_hidden_states,
            top_k=self._config.top_k,
            cutoff=self._cutoff,
        )
        if not predictions:
            return

        tasks: list[PrefetchTask] = []
        for i, (layer_name, expert_ids) in enumerate(predictions.items()):
            cache = self._elmm._layer_caches.get(layer_name)
            if cache is None:
                continue
            urgency = 1.0 / (i + 1)
            for eid in expert_ids:
                if not cache.contains(eid):
                    tasks.append(PrefetchTask(
                        layer_name=layer_name,
                        expert_id=eid,
                        priority=urgency,
                    ))
        if tasks:
            self._prefetcher.submit_tasks(tasks)

    def uninstall(self):
        if self._installed and self._original_propose is not None:
            try:
                from vllm.v1.spec_decode.eagle import EagleProposer
                EagleProposer.propose = self._original_propose
                self._installed = False
            except ImportError:
                pass

    def get_stats(self) -> dict:
        return {
            "installed": self._installed,
            "draft_rounds": self._draft_rounds,
            "cutoff": self._cutoff,
        }


# ---------------------------------------------------------------------------
# SP-MoE Activation
# ---------------------------------------------------------------------------

_spmoe_state: Optional[dict] = None


def get_spmoe_state() -> Optional[dict]:
    return _spmoe_state


def activate_spmoe(elmm_manager: Any) -> Optional[dict]:
    """
    Activate SP-MoE baseline on top of the ELMM infrastructure.

    Steps:
      1. Extract gate weights from all offloaded MoE layers
      2. Auto-detect expert size + compute cutoff layer
      3. Initialize gating predictor and async prefetcher
      4. Install draft hook on EAGLE3
      5. Patch ELMM forward to capture hidden states for prediction

    The caller must ensure BriskMoE optimizations are DISABLED:
        ELMM_POOL_DIRECT=0 ELMM_STALE_REMAP=0
        ELMM_ORACLE_PREFETCH=0 BRISKMOE_DIPP=0
    """
    global _spmoe_state

    if not elmm_manager._installed:
        print("[SP-MoE] ELMM not installed, cannot activate", file=sys.stderr)
        return None

    config = SPMoEConfig(
        top_k=int(os.environ.get("SPMOE_TOP_K", "8")),
        cutoff_layer=int(os.environ.get("SPMOE_CUTOFF", "-1")),
        draft_duration_ms=float(os.environ.get("SPMOE_DRAFT_MS", "30")),
        t_io_per_expert_ms=float(os.environ.get("SPMOE_T_IO_MS", "0.79")),
        expert_size_bytes=int(os.environ.get("SPMOE_EXPERT_SIZE", "0")),
        batched_io=os.environ.get("SPMOE_BATCHED_IO", "1") == "1",
        use_worker_thread=os.environ.get("SPMOE_WORKER_THREAD", "1") == "1",
    )
    device = torch.device("cuda")

    # --- Step 1: Extract gate weights from offloaded MoE modules ---
    predictor = SPMoEGatingPredictor(device=device)

    for layer_name, module in elmm_manager._patched_modules.items():
        gate_weight = None
        # vLLM FusedMoE stores gate as module.gate (ReplicatedLinear)
        if hasattr(module, 'gate') and module.gate is not None:
            if hasattr(module.gate, 'weight'):
                gate_weight = module.gate.weight.data
            elif hasattr(module.gate, 'linear'):
                gate_weight = module.gate.linear.weight.data
        if gate_weight is None:
            print(f"[SP-MoE] WARNING: No gate for {layer_name}",
                  file=sys.stderr)
            continue

        meta = elmm_manager._layer_meta.get(layer_name, {})
        num_experts = meta.get("num_experts", gate_weight.shape[0])
        predictor.register_layer(layer_name, gate_weight, num_experts)

    num_layers = len(predictor._layer_order)
    if num_layers == 0:
        print("[SP-MoE] No layers registered, aborting", file=sys.stderr)
        return None
    print(f"[SP-MoE] Registered gate weights for {num_layers} layers",
          file=sys.stderr, flush=True)

    # --- Step 2: Auto-detect expert size ---
    if config.expert_size_bytes == 0:
        first_meta = next(iter(elmm_manager._layer_meta.values()))
        config.expert_size_bytes = first_meta["expert_size"]
        print(f"[SP-MoE] expert_size={config.expert_size_bytes} bytes",
              file=sys.stderr, flush=True)

    # --- Step 3: Compute cutoff layer ---
    if config.cutoff_layer < 0:
        available_mem = config.prefetch_memory_bytes
        if available_mem == 0:
            available_mem = elmm_manager.config.gpu_cache_budget_bytes
        config.cutoff_layer = compute_cutoff_layer(
            num_layers=num_layers,
            top_k=config.top_k,
            t_io_ms=config.t_io_per_expert_ms,
            draft_duration_ms=config.draft_duration_ms,
            expert_size_bytes=config.expert_size_bytes,
            available_memory_bytes=max(available_mem, config.expert_size_bytes),
        )
    print(f"[SP-MoE] Cutoff={config.cutoff_layer}/{num_layers} "
          f"(top_k={config.top_k}, draft={config.draft_duration_ms}ms, "
          f"t_io={config.t_io_per_expert_ms}ms/expert)",
          file=sys.stderr, flush=True)

    # --- Step 4: Initialize prefetcher ---
    prefetcher = SPMoEPrefetcher(elmm_manager, config, device)
    if config.use_worker_thread:
        prefetcher.start()
        print("[SP-MoE] Worker prefetch thread started",
              file=sys.stderr, flush=True)

    # --- Step 5: Install draft hook ---
    hook = SPMoEDraftHook(elmm_manager, predictor, prefetcher, config)
    if not hook.install():
        prefetcher.stop()
        print("[SP-MoE] Failed to install draft hook", file=sys.stderr)
        return None

    # --- Step 6: Capture hidden states for prediction ---
    # Wrap _elmm_forward_impl to grab hidden_states from first MoE layer.
    if predictor._layer_order:
        first_layer = predictor._layer_order[0]
        _orig_forward = elmm_manager._elmm_forward_impl

        def _capturing_forward(self_mgr, layer_name, hidden_states,
                               router_logits):
            if layer_name == first_layer:
                hook.set_hidden_states(hidden_states.detach())
            return _orig_forward(layer_name, hidden_states, router_logits)

        elmm_manager._elmm_forward_impl = types.MethodType(
            _capturing_forward, elmm_manager
        )
        print(f"[SP-MoE] Hidden state capture on {first_layer}",
              file=sys.stderr, flush=True)

    result = {
        "predictor": predictor,
        "prefetcher": prefetcher,
        "hook": hook,
        "config": config,
    }
    _spmoe_state = result
    print("[SP-MoE] Baseline activated successfully",
          file=sys.stderr, flush=True)
    return result
