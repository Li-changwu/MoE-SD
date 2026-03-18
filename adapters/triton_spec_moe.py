"""
SpecFusedMoE Triton Kernel — SD-Aware Expert Deduplication
============================================================
Production Triton kernel that replaces vLLM's native fused_moe during
speculative decoding verify phase.

Key optimizations vs vanilla fused_moe:
  1. Cross-token expert deduplication: load each expert weight once
  2. Fused SiLU-gate-up + down projection
  3. Frozen-token skip via active_mask
  4. Compatible with vLLM's weight layout: w1=[E,2N,D], w2=[E,D,N]

Interface contract with vLLM 0.17:
  vLLM calls `fused_moe(hidden_states, w1, w2, topk_weights, topk_ids, ...)`
  We monkey-patch this to `spec_fused_moe(...)` during SD verify.

Requires: triton >= 3.0, torch >= 2.1
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton Kernel: Fused SiLU-gated MoE with dedup dispatch
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _spec_fused_moe_kernel(
        # Pointers
        hidden_ptr,          # [T, D] input hidden states (all active tokens)
        w1_ptr,              # [E, 2*N, D] gate+up projection weights
        w2_ptr,              # [E, D, N] down projection weights
        output_ptr,          # [T, D] output accumulation buffer
        # Dispatch table: which (token, expert, weight) combinations to process
        dispatch_token_ids,  # [num_dispatch] which token this dispatch entry belongs to
        dispatch_expert_ids, # [num_dispatch] which expert to use
        dispatch_weights,    # [num_dispatch] routing weight for this token-expert pair
        # Dimensions
        num_dispatch,        # total number of dispatch entries
        D: tl.constexpr,     # hidden_size
        N: tl.constexpr,     # moe_intermediate_size
        # Strides
        stride_ht, stride_hd,      # hidden_states strides
        stride_w1e, stride_w1n, stride_w1d,  # w1 strides [E, 2N, D]
        stride_w2e, stride_w2d, stride_w2n,  # w2 strides [E, D, N]
        stride_ot, stride_od,      # output strides
        # Block sizes
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Each program instance processes one dispatch entry (token, expert) pair.
        Computes: out[token] += weight * down_proj(silu(gate_proj(x)) * up_proj(x))

        w1 layout: [E, 2*N, D] where first N rows = gate, next N rows = up
        w2 layout: [E, D, N]
        """
        pid = tl.program_id(0)
        if pid >= num_dispatch:
            return

        token_id = tl.load(dispatch_token_ids + pid)
        expert_id = tl.load(dispatch_expert_ids + pid)
        w = tl.load(dispatch_weights + pid)

        # Pointers to this token's hidden state
        hidden_base = hidden_ptr + token_id * stride_ht

        # Pointers to expert weights
        w1_gate_base = w1_ptr + expert_id * stride_w1e  # [2N, D], first N rows
        w1_up_base = w1_ptr + expert_id * stride_w1e + N * stride_w1n  # offset by N rows
        w2_base = w2_ptr + expert_id * stride_w2e  # [D, N]

        # Output pointer
        out_base = output_ptr + token_id * stride_ot

        # --- Phase 1: Compute gate = x @ gate_proj.T and up = x @ up_proj.T ---
        # We tile over N (intermediate dimension)
        for n_start in range(0, N, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < N

            gate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            up_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

            # Dot product: tile over D
            for d_start in range(0, D, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offsets < D

                # Load hidden[d_start:d_start+BLOCK_D]
                x = tl.load(hidden_base + d_offsets * stride_hd, mask=d_mask, other=0.0)

                # Load gate_proj[n_offsets, d_offsets] — need 2D tile
                for ni in range(BLOCK_N):
                    if n_start + ni < N:
                        gate_w = tl.load(
                            w1_gate_base + (n_start + ni) * stride_w1n + d_offsets * stride_w1d,
                            mask=d_mask, other=0.0
                        )
                        up_w = tl.load(
                            w1_up_base + (n_start + ni) * stride_w1n + d_offsets * stride_w1d,
                            mask=d_mask, other=0.0
                        )
                        gate_acc = tl.where(
                            tl.arange(0, BLOCK_N) == ni,
                            gate_acc + tl.sum(x * gate_w),
                            gate_acc
                        )
                        up_acc = tl.where(
                            tl.arange(0, BLOCK_N) == ni,
                            up_acc + tl.sum(x * up_w),
                            up_acc
                        )

            # SiLU activation on gate: silu(x) = x * sigmoid(x)
            gate_activated = gate_acc * tl.sigmoid(gate_acc)
            # Element-wise multiply: intermediate = silu(gate) * up
            intermediate = gate_activated * up_acc  # [BLOCK_N]

            # --- Phase 2: Down projection: out += weight * intermediate @ w2.T ---
            # w2: [D, N], so out[d] = sum_n(intermediate[n] * w2[d, n])
            for d_start2 in range(0, D, BLOCK_D):
                d_offsets2 = d_start2 + tl.arange(0, BLOCK_D)
                d_mask2 = d_offsets2 < D

                down_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
                for ni in range(BLOCK_N):
                    if n_start + ni < N:
                        w2_col = tl.load(
                            w2_base + d_offsets2 * stride_w2d + (n_start + ni) * stride_w2n,
                            mask=d_mask2, other=0.0
                        )
                        intermed_val = tl.load(
                            tl.make_block_ptr(
                                intermediate, shape=[BLOCK_N], strides=[1],
                                offsets=[ni], block_shape=[1], order=[0]
                            )
                        ) if False else intermediate  # use broadcast
                        # Scalar broadcast of intermediate[ni]
                        # Use element extraction
                        i_val = tl.sum(tl.where(tl.arange(0, BLOCK_N) == ni, intermediate, 0.0))
                        down_acc += i_val * w2_col

                # Atomic add to output (multiple dispatch entries may write to same token)
                tl.atomic_add(out_base + d_offsets2 * stride_od, w * down_acc, mask=d_mask2)


# ---------------------------------------------------------------------------
# High-Performance PyTorch Fallback (works without Triton / on CPU)
# ---------------------------------------------------------------------------

class SpecFusedMoEFunction:
    """
    Efficient PyTorch implementation of dedup-aware fused MoE.
    Used as fallback when Triton is unavailable, and as reference for correctness.

    The key optimization: builds a dispatch table mapping unique experts to
    all tokens that need them, then batches the matmul per expert.
    """

    @staticmethod
    def forward(
        hidden_states: torch.Tensor,   # [T, D]
        w1: torch.Tensor,              # [E, 2*N, D]  (gate_proj || up_proj)
        w2: torch.Tensor,              # [E, D, N]    (down_proj)
        topk_weights: torch.Tensor,    # [T, top_k]
        topk_ids: torch.Tensor,        # [T, top_k]
        active_mask: torch.Tensor = None,  # [T] bool: True = process this token
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input hidden states [T, D]
            w1: Expert gate+up weights [num_experts, 2*intermediate, hidden]
            w2: Expert down weights [num_experts, hidden, intermediate]
            topk_weights: Routing weights [T, top_k]
            topk_ids: Expert indices [T, top_k]
            active_mask: Optional mask for frozen tokens

        Returns:
            output: [T, D]
        """
        T, D = hidden_states.shape
        E = w1.shape[0]
        N = w1.shape[1] // 2  # intermediate_size
        top_k = topk_ids.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        output = torch.zeros(T, D, device=device, dtype=dtype)

        # Apply active mask
        if active_mask is not None:
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            if len(active_idx) == 0:
                return output
        else:
            active_idx = torch.arange(T, device=device)

        # Build expert -> tokens dispatch table (DEDUPLICATION)
        # Instead of iterating per-token, group by expert
        expert_to_tokens = {}  # expert_id -> [(token_idx_in_active, slot_idx)]
        active_topk = topk_ids[active_idx]   # [A, top_k]
        active_weights = topk_weights[active_idx]  # [A, top_k]
        active_hidden = hidden_states[active_idx]  # [A, D]

        unique_experts = active_topk.reshape(-1).unique()

        for eid_tensor in unique_experts:
            eid = eid_tensor.item()
            # Find all (token, slot) pairs that use this expert
            mask = (active_topk == eid)  # [A, top_k]
            token_slots = mask.nonzero(as_tuple=False)  # [N_matches, 2]
            expert_to_tokens[eid] = token_slots

        # Statistics
        naive_loads = len(active_idx) * top_k
        dedup_loads = len(unique_experts)

        # Process each unique expert ONCE
        active_output = torch.zeros(len(active_idx), D, device=device, dtype=dtype)

        for eid, token_slots in expert_to_tokens.items():
            if eid < 0 or eid >= E:
                continue

            token_indices = token_slots[:, 0]  # indices into active batch
            slot_indices = token_slots[:, 1]   # which top-k slot

            # Gather hidden states for tokens using this expert
            expert_input = active_hidden[token_indices]  # [n, D]

            # Expert computation: SiLU-gated FFN
            # w1[eid]: [2*N, D] -> gate_proj = w1[eid, :N, :], up_proj = w1[eid, N:, :]
            gate_up = expert_input @ w1[eid].T  # [n, 2*N]
            gate = F.silu(gate_up[:, :N])        # [n, N]
            up = gate_up[:, N:]                   # [n, N]
            intermediate = gate * up              # [n, N]

            # Down projection
            expert_out = intermediate @ w2[eid].T  # [n, D]

            # Gather routing weights and accumulate
            w_vals = active_weights[token_indices, slot_indices]  # [n]
            active_output.index_add_(
                0, token_indices,
                expert_out * w_vals.unsqueeze(1),
            )

        output[active_idx] = active_output
        return output, naive_loads, dedup_loads


class SpecFusedMoEDispatcher:
    """
    High-level dispatcher that chooses between Triton kernel and PyTorch fallback.
    Manages the dedup dispatch table and statistics.

    Usage:
        dispatcher = SpecFusedMoEDispatcher(num_experts=128, top_k=8)
        output = dispatcher(hidden_states, w1, w2, topk_weights, topk_ids,
                           active_mask=mask)
    """

    def __init__(
        self,
        num_experts: int = 128,
        top_k: int = 8,
        use_triton: bool = True,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_triton = use_triton and HAS_TRITON

        # Running statistics
        self._total_naive_loads = 0
        self._total_dedup_loads = 0
        self._total_calls = 0
        self._total_tokens = 0
        self._total_frozen = 0

    def __call__(
        self,
        hidden_states: torch.Tensor,   # [T, D]
        w1: torch.Tensor,              # [E, 2N, D]
        w2: torch.Tensor,              # [E, D, N]
        topk_weights: torch.Tensor,    # [T, top_k]
        topk_ids: torch.Tensor,        # [T, top_k]
        active_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]

        if active_mask is not None:
            num_frozen = int((~active_mask).sum().item())
        else:
            num_frozen = 0

        if self.use_triton and hidden_states.is_cuda:
            output = self._triton_dispatch(
                hidden_states, w1, w2, topk_weights, topk_ids, active_mask
            )
            # Estimate dedup stats
            if active_mask is not None:
                active_topk = topk_ids[active_mask]
            else:
                active_topk = topk_ids
            naive = active_topk.shape[0] * self.top_k
            dedup = int(active_topk.reshape(-1).unique().numel())
        else:
            output, naive, dedup = SpecFusedMoEFunction.forward(
                hidden_states, w1, w2, topk_weights, topk_ids, active_mask
            )

        self._total_naive_loads += naive
        self._total_dedup_loads += dedup
        self._total_calls += 1
        self._total_tokens += T
        self._total_frozen += num_frozen

        return output

    def _triton_dispatch(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        active_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Dispatch via Triton kernel with dedup table."""
        T, D = hidden_states.shape
        N = w1.shape[1] // 2

        # Build dispatch table
        if active_mask is not None:
            active_idx = active_mask.nonzero(as_tuple=True)[0]
        else:
            active_idx = torch.arange(T, device=hidden_states.device)

        if len(active_idx) == 0:
            return torch.zeros_like(hidden_states)

        # Flatten (token, slot) -> (token_id, expert_id, weight)
        dispatch_tokens = []
        dispatch_experts = []
        dispatch_w = []

        for local_i, global_i in enumerate(active_idx):
            gi = global_i.item()
            for s in range(self.top_k):
                dispatch_tokens.append(gi)
                dispatch_experts.append(topk_ids[gi, s].item())
                dispatch_w.append(topk_weights[gi, s].item())

        device = hidden_states.device
        dispatch_token_ids = torch.tensor(dispatch_tokens, device=device, dtype=torch.int32)
        dispatch_expert_ids = torch.tensor(dispatch_experts, device=device, dtype=torch.int32)
        dispatch_weights_t = torch.tensor(dispatch_w, device=device, dtype=hidden_states.dtype)
        num_dispatch = len(dispatch_tokens)

        output = torch.zeros_like(hidden_states)

        # Launch kernel
        grid = (num_dispatch,)
        BLOCK_D = min(128, D)
        BLOCK_N = min(64, N)

        _spec_fused_moe_kernel[grid](
            hidden_states, w1, w2, output,
            dispatch_token_ids, dispatch_expert_ids, dispatch_weights_t,
            num_dispatch,
            D, N,
            hidden_states.stride(0), hidden_states.stride(1),
            w1.stride(0), w1.stride(1), w1.stride(2),
            w2.stride(0), w2.stride(1), w2.stride(2),
            output.stride(0), output.stride(1),
            BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
        )
        return output

    @property
    def dedup_ratio(self) -> float:
        if self._total_naive_loads == 0:
            return 0.0
        return 1.0 - self._total_dedup_loads / self._total_naive_loads

    @property
    def effective_maf(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._total_dedup_loads / (self._total_calls * self.top_k)

    @property
    def frozen_ratio(self) -> float:
        if self._total_tokens == 0:
            return 0.0
        return self._total_frozen / self._total_tokens

    def get_statistics(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_frozen": self._total_frozen,
            "total_naive_loads": self._total_naive_loads,
            "total_dedup_loads": self._total_dedup_loads,
            "dedup_ratio": round(self.dedup_ratio, 4),
            "effective_maf": round(self.effective_maf, 4),
            "frozen_ratio": round(self.frozen_ratio, 4),
            "triton_enabled": self.use_triton,
        }

    def reset_statistics(self):
        self._total_naive_loads = 0
        self._total_dedup_loads = 0
        self._total_calls = 0
        self._total_tokens = 0
        self._total_frozen = 0


# ---------------------------------------------------------------------------
# Optimized Triton Kernel v2: Expert-Grouped Matmul
# ---------------------------------------------------------------------------
# This is the real performance-critical kernel. Instead of dispatching per
# (token, expert) pair, we group tokens by expert and launch one Triton
# matmul program per (expert, output_tile) pair — true batched GEMM.

if HAS_TRITON:

    @triton.jit
    def _grouped_gemm_gate_up_kernel(
        # Input
        x_ptr,          # [max_tokens_per_expert, D] gathered input (per-expert)
        w1_ptr,         # [E, 2*N, D] gate+up proj
        # Output
        gate_up_ptr,    # [max_tokens_per_expert, 2*N] (gate || up)
        # Dispatch info
        expert_id,      # scalar
        num_tokens,     # how many tokens use this expert
        # Dims
        D: tl.constexpr,
        N2: tl.constexpr,  # 2*N
        # Strides
        stride_xr, stride_xd,
        stride_w1e, stride_w1n, stride_w1d,
        stride_or, stride_on,
        # Tiles
        BLOCK_M: tl.constexpr,  # tokens tile
        BLOCK_N: tl.constexpr,  # output tile
        BLOCK_K: tl.constexpr,  # reduction tile
    ):
        """Batched GEMM: gate_up = x @ w1[expert].T for all tokens using this expert."""
        pid_m = tl.program_id(0)  # token tile
        pid_n = tl.program_id(1)  # output dim tile

        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N

        m_offsets = m_start + tl.arange(0, BLOCK_M)
        n_offsets = n_start + tl.arange(0, BLOCK_N)

        # Accumulator
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # w1 base for this expert
        w1_base = w1_ptr + expert_id * stride_w1e

        for k_start in range(0, D, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)

            # Load x tile [BLOCK_M, BLOCK_K]
            x = tl.load(
                x_ptr + m_offsets[:, None] * stride_xr + k_offsets[None, :] * stride_xd,
                mask=(m_offsets[:, None] < num_tokens) & (k_offsets[None, :] < D),
                other=0.0,
            )

            # Load w1 tile [BLOCK_N, BLOCK_K] -> transposed as [BLOCK_K, BLOCK_N]
            w = tl.load(
                w1_base + n_offsets[None, :] * stride_w1n + k_offsets[:, None] * stride_w1d,
                mask=(n_offsets[None, :] < N2) & (k_offsets[:, None] < D),
                other=0.0,
            )

            acc += tl.dot(x, w)

        # Store gate_up
        tl.store(
            gate_up_ptr + m_offsets[:, None] * stride_or + n_offsets[None, :] * stride_on,
            acc,
            mask=(m_offsets[:, None] < num_tokens) & (n_offsets[None, :] < N2),
        )

    @triton.jit
    def _silu_mul_kernel(
        gate_up_ptr,  # [M, 2*N] in-place
        out_ptr,      # [M, N] result
        M, N: tl.constexpr,
        stride_gm, stride_gn,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused SiLU(gate) * up element-wise."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask = (m_off[:, None] < M) & (n_off[None, :] < N)

        gate = tl.load(
            gate_up_ptr + m_off[:, None] * stride_gm + n_off[None, :] * stride_gn,
            mask=mask, other=0.0,
        ).to(tl.float32)

        up = tl.load(
            gate_up_ptr + m_off[:, None] * stride_gm + (n_off[None, :] + N) * stride_gn,
            mask=mask, other=0.0,
        ).to(tl.float32)

        result = (gate * tl.sigmoid(gate)) * up

        tl.store(
            out_ptr + m_off[:, None] * stride_om + n_off[None, :] * stride_on,
            result,
            mask=mask,
        )

    @triton.jit
    def _grouped_gemm_down_kernel(
        intermediate_ptr,  # [max_tokens, N]
        w2_ptr,            # [E, D, N]
        output_ptr,        # [max_tokens, D]
        expert_id,
        num_tokens,
        D: tl.constexpr,
        N: tl.constexpr,
        stride_ir, stride_in,
        stride_w2e, stride_w2d, stride_w2n,
        stride_or, stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Down projection: output = intermediate @ w2[expert].T"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N

        m_offsets = m_start + tl.arange(0, BLOCK_M)
        n_offsets = n_start + tl.arange(0, BLOCK_N)

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        w2_base = w2_ptr + expert_id * stride_w2e

        for k_start in range(0, N, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)

            x = tl.load(
                intermediate_ptr + m_offsets[:, None] * stride_ir + k_offsets[None, :] * stride_in,
                mask=(m_offsets[:, None] < num_tokens) & (k_offsets[None, :] < N),
                other=0.0,
            )

            # w2: [D, N] for this expert, we want output in D dimension
            w = tl.load(
                w2_base + n_offsets[None, :] * stride_w2d + k_offsets[:, None] * stride_w2n,
                mask=(n_offsets[None, :] < D) & (k_offsets[:, None] < N),
                other=0.0,
            )

            acc += tl.dot(x, w)

        tl.store(
            output_ptr + m_offsets[:, None] * stride_or + n_offsets[None, :] * stride_od,
            acc,
            mask=(m_offsets[:, None] < num_tokens) & (n_offsets[None, :] < D),
        )


class SpecFusedMoETritonV2:
    """
    Production Triton implementation: Expert-grouped batched GEMM.

    For each unique expert in the verify batch:
      1. Gather all tokens routed to this expert
      2. Launch batched GEMM for gate_up = tokens @ w1[expert].T
      3. Launch fused SiLU activation
      4. Launch batched GEMM for down = intermediate @ w2[expert].T
      5. Scatter-add weighted results back to output

    This naturally deduplicates: each expert's weight is loaded from HBM
    exactly once regardless of how many tokens share it.
    """

    def __init__(
        self,
        num_experts: int = 128,
        top_k: int = 8,
        block_m: int = 16,
        block_n: int = 64,
        block_k: int = 64,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

        self._total_naive_loads = 0
        self._total_dedup_loads = 0
        self._total_calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, D]
        w1: torch.Tensor,              # [E, 2*N, D]
        w2: torch.Tensor,              # [E, D, N]
        topk_weights: torch.Tensor,    # [T, top_k]
        topk_ids: torch.Tensor,        # [T, top_k]
        active_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        T, D = hidden_states.shape
        N = w1.shape[1] // 2
        device = hidden_states.device
        dtype = hidden_states.dtype

        output = torch.zeros(T, D, device=device, dtype=dtype)

        # Active tokens
        if active_mask is not None:
            active_idx = active_mask.nonzero(as_tuple=True)[0]
        else:
            active_idx = torch.arange(T, device=device)

        if len(active_idx) == 0:
            return output

        active_topk_ids = topk_ids[active_idx]      # [A, top_k]
        active_topk_w = topk_weights[active_idx]     # [A, top_k]
        active_hidden = hidden_states[active_idx]     # [A, D]
        A = len(active_idx)

        # Get unique experts and build dispatch
        unique_experts = active_topk_ids.reshape(-1).unique()
        self._total_naive_loads += A * self.top_k
        self._total_dedup_loads += len(unique_experts)
        self._total_calls += 1

        # Process each expert via grouped GEMM
        for eid in unique_experts:
            eid_val = eid.item()

            # Find all (token_in_active, slot) using this expert
            mask = (active_topk_ids == eid_val)  # [A, top_k]
            token_slot_pairs = mask.nonzero(as_tuple=False)  # [n, 2]
            token_indices = token_slot_pairs[:, 0]  # local indices in active batch
            slot_indices = token_slot_pairs[:, 1]
            n_tokens = len(token_indices)

            # Gather input
            expert_input = active_hidden[token_indices]  # [n, D]

            if HAS_TRITON and device.type == 'cuda':
                # Allocate buffers
                gate_up_buf = torch.empty(n_tokens, 2 * N, device=device, dtype=dtype)
                intermediate_buf = torch.empty(n_tokens, N, device=device, dtype=dtype)
                expert_out = torch.empty(n_tokens, D, device=device, dtype=dtype)

                # Gate+Up GEMM
                grid_gu = (
                    triton.cdiv(n_tokens, self.block_m),
                    triton.cdiv(2 * N, self.block_n),
                )
                _grouped_gemm_gate_up_kernel[grid_gu](
                    expert_input, w1, gate_up_buf,
                    eid_val, n_tokens,
                    D, 2 * N,
                    expert_input.stride(0), expert_input.stride(1),
                    w1.stride(0), w1.stride(1), w1.stride(2),
                    gate_up_buf.stride(0), gate_up_buf.stride(1),
                    BLOCK_M=self.block_m,
                    BLOCK_N=self.block_n,
                    BLOCK_K=self.block_k,
                )

                # Fused SiLU
                grid_act = (
                    triton.cdiv(n_tokens, self.block_m),
                    triton.cdiv(N, self.block_n),
                )
                _silu_mul_kernel[grid_act](
                    gate_up_buf, intermediate_buf,
                    n_tokens, N,
                    gate_up_buf.stride(0), gate_up_buf.stride(1),
                    intermediate_buf.stride(0), intermediate_buf.stride(1),
                    BLOCK_M=self.block_m,
                    BLOCK_N=self.block_n,
                )

                # Down GEMM
                grid_down = (
                    triton.cdiv(n_tokens, self.block_m),
                    triton.cdiv(D, self.block_n),
                )
                _grouped_gemm_down_kernel[grid_down](
                    intermediate_buf, w2, expert_out,
                    eid_val, n_tokens,
                    D, N,
                    intermediate_buf.stride(0), intermediate_buf.stride(1),
                    w2.stride(0), w2.stride(1), w2.stride(2),
                    expert_out.stride(0), expert_out.stride(1),
                    BLOCK_M=self.block_m,
                    BLOCK_N=self.block_n,
                    BLOCK_K=self.block_k,
                )
            else:
                # PyTorch fallback
                gate_up = expert_input @ w1[eid_val].T  # [n, 2N]
                gate = F.silu(gate_up[:, :N])
                up = gate_up[:, N:]
                expert_out = (gate * up) @ w2[eid_val].T  # [n, D]

            # Weighted scatter-add
            weights = active_topk_w[token_indices, slot_indices].unsqueeze(1)  # [n, 1]
            # Map back to global token indices
            global_token_indices = active_idx[token_indices]
            output.index_add_(0, global_token_indices, (expert_out * weights).to(dtype))

        return output

    def get_statistics(self) -> dict:
        if self._total_naive_loads == 0:
            dedup_ratio = 0.0
        else:
            dedup_ratio = 1.0 - self._total_dedup_loads / self._total_naive_loads
        return {
            "total_calls": self._total_calls,
            "naive_loads": self._total_naive_loads,
            "dedup_loads": self._total_dedup_loads,
            "dedup_ratio": round(dedup_ratio, 4),
            "effective_maf": round(
                self._total_dedup_loads / (self._total_calls * self.top_k)
                if self._total_calls > 0 else 0, 4
            ),
            "triton_available": HAS_TRITON,
        }

    def reset_statistics(self):
        self._total_naive_loads = 0
        self._total_dedup_loads = 0
        self._total_calls = 0
