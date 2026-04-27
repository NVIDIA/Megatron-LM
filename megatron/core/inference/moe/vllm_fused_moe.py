# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/vllm-project/vllm.
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
"""vLLM-style Triton fused MoE kernel (BF16) for Megatron inference."""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

from megatron.core.inference.moe.fused_moe import ActivationType

# ---------------------------------------------------------------------------
# Triton kernel – BF16 grouped GEMM with indirect token addressing
# ---------------------------------------------------------------------------


@triton.jit
def _fused_moe_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Flags / constexprs
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused MoE grouped GEMM with indirect token addressing.

    Each Triton program computes a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of
    output C.  Tokens are NOT physically permuted; instead the kernel reads
    from original positions via *sorted_token_ids* and selects expert
    weights via *expert_ids*.

    A: [num_tokens, K]          – input hidden states (unpermuted)
    B: [num_local_experts, N, K] – stacked expert weights
    C: [num_tokens * topk, N]   – output (one row per topk slot)

    sorted_token_ids maps each padded slot to a flat (token * topk + k)
    index.  ``offs_token // top_k`` recovers the original token index for
    reading A.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Non-local expert block – write zeros
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A reads use (token // top_k) so the same input row serves all topk slots
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(tl.bfloat16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Token-to-expert alignment (pure-torch, no C++ dependency)
# ---------------------------------------------------------------------------


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert and pad each group to *block_size*.

    Args:
        topk_ids: [M, topk] **local** expert indices (−1 for non-local).
        block_size: BLOCK_SIZE_M used by the Triton kernel.
        num_local_experts: number of experts on this rank.

    Returns:
        sorted_token_ids  – [total_padded] int32 flat (token·topk + k) index
                            per slot; padding slots = num_valid.
        expert_ids        – [total_padded // block_size] int32 expert per block.
        num_tokens_post_padded – [1] int32 scalar tensor.
    """
    M, topk = topk_ids.shape
    num_valid = M * topk
    device = topk_ids.device

    flat_ids = topk_ids.reshape(-1)
    flat_indices = torch.arange(num_valid, dtype=torch.int32, device=device)

    # Push non-local (−1) tokens to the end during sort
    sort_keys = flat_ids.clone()
    sort_keys[flat_ids < 0] = num_local_experts
    _, sort_order = sort_keys.sort(stable=True)

    sorted_flat_indices = flat_indices[sort_order]
    sorted_expert_ids = flat_ids[sort_order]

    # Tokens per local expert
    valid_mask = sorted_expert_ids >= 0
    num_valid_sorted = valid_mask.sum().item()
    tokens_per_expert = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    if num_valid_sorted > 0:
        valid_experts = sorted_expert_ids[:num_valid_sorted].long()
        tokens_per_expert.scatter_add_(
            0, valid_experts, torch.ones(num_valid_sorted, dtype=torch.int32, device=device)
        )

    padded_counts = ((tokens_per_expert + block_size - 1) // block_size) * block_size
    padded_offsets = torch.zeros(num_local_experts + 1, dtype=torch.int32, device=device)
    padded_offsets[1:] = torch.cumsum(padded_counts, 0)
    total_padded = padded_offsets[-1].item()

    unpadded_offsets = torch.zeros(num_local_experts + 1, dtype=torch.int32, device=device)
    unpadded_offsets[1:] = torch.cumsum(tokens_per_expert, 0)

    padded_sorted = torch.full((total_padded,), num_valid, dtype=torch.int32, device=device)
    expert_ids_out = torch.full(
        (total_padded // block_size,), -1, dtype=torch.int32, device=device
    )

    for e in range(num_local_experts):
        count = tokens_per_expert[e].item()
        padded = padded_counts[e].item()
        if padded == 0:
            continue
        src = unpadded_offsets[e].item()
        dst = padded_offsets[e].item()
        padded_sorted[dst : dst + count] = sorted_flat_indices[src : src + count]
        num_blocks = padded // block_size
        expert_ids_out[dst // block_size : dst // block_size + num_blocks] = e

    num_post = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return padded_sorted, expert_ids_out, num_post


# ---------------------------------------------------------------------------
# Kernel launcher
# ---------------------------------------------------------------------------

# Default tile sizes — reasonable starting point for H100 / BF16.
_DEFAULT_CONFIG = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8)


def _invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Optional[dict] = None,
):
    """Launch the Triton fused-MoE kernel for one GEMM pass."""
    if config is None:
        config = _DEFAULT_CONFIG

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    _fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        **config,
    )


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------


def _apply_activation(activation_type: ActivationType, output: torch.Tensor, input: torch.Tensor):
    """Element-wise activation, writing into *output*."""
    if activation_type == ActivationType.SQUARED_RELU:
        torch.nn.functional.relu(input, inplace=False, out=output)
        output.mul_(output)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


# ---------------------------------------------------------------------------
# Public API – drop-in replacement for mcore_fused_moe
# ---------------------------------------------------------------------------


def vllm_fused_moe(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    routing_map: Optional[torch.Tensor] = None,
    tokens_per_expert: Optional[torch.Tensor] = None,
    skip_permute: bool = False,
) -> torch.Tensor:
    """Fused MoE using the vLLM Triton grouped-GEMM kernel (BF16).

    Two modes — same interface as ``mcore_fused_moe``:

    * ``skip_permute=False``: tokens are **unpermuted**.  Requires
      *routing_map* ``[M, topk]``.  The kernel reads from original token
      positions via indirection — **no physical permutation needed**.

    * ``skip_permute=True``: tokens are **already permuted** by the
      dispatcher.  Requires *tokens_per_expert* ``[num_local_experts]``.
      A thin adapter builds the indirection table from the already-
      contiguous layout.

    Args:
        hidden_states: [num_tokens, hidden_size] BF16 input.
        probs: routing probabilities.
            [num_tokens, topk] when skip_permute=False,
            [num_tokens] when skip_permute=True.
        fc1_weight: [num_local_experts, fc1_out, hidden_size] BF16.
        fc2_weight: [num_local_experts, hidden_size, fc1_out] BF16.
        activation_type: ActivationType enum.
        num_local_experts: experts on this rank.
        local_expert_start: first global expert index on this rank.
        routing_map: [num_tokens, topk] global expert assignments.
        tokens_per_expert: [num_local_experts] int32 counts.
        skip_permute: whether tokens are already in expert order.

    Returns:
        [num_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"vllm_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )

    BLOCK_SIZE_M = _DEFAULT_CONFIG["BLOCK_SIZE_M"]

    if not skip_permute:
        # ── Mode 1: unpermuted tokens, use indirect-addressing kernel ──
        assert routing_map is not None
        M, topk = routing_map.shape

        # Map global expert IDs → local (−1 for non-local)
        local_topk_ids = routing_map.int() - local_expert_start
        non_local = (local_topk_ids < 0) | (local_topk_ids >= num_local_experts)
        local_topk_ids[non_local] = -1

        sorted_token_ids, expert_ids, num_post = _moe_align_block_size(
            local_topk_ids, BLOCK_SIZE_M, num_local_experts
        )
        num_valid = M * topk

        N = fc1_weight.size(1)  # fc1 output dim
        K = fc1_weight.size(2)  # hidden_size

        # Flatten topk_weights to [M * topk] in the same token-major order
        topk_weights_flat = probs.reshape(-1).contiguous()

        # FC1: [M, K] → [M*topk, N]
        intermediate1 = torch.zeros(M * topk, N, dtype=hidden_states.dtype, device=hidden_states.device)
        _invoke_fused_moe_kernel(
            hidden_states,
            fc1_weight,
            intermediate1,
            topk_weights_flat,
            sorted_token_ids,
            expert_ids,
            num_post,
            mul_routed_weight=False,
            top_k=topk,
        )

        # Activation
        activation_out_dim = N  # non-gated: same size
        intermediate2 = torch.empty(
            M * topk, activation_out_dim, dtype=hidden_states.dtype, device=hidden_states.device
        )
        _apply_activation(activation_type, intermediate2, intermediate1)

        # FC2: [M*topk, activation_out_dim] → [M*topk, K], with routing weights
        intermediate3 = torch.zeros(
            M * topk, K, dtype=hidden_states.dtype, device=hidden_states.device
        )
        _invoke_fused_moe_kernel(
            intermediate2,
            fc2_weight,
            intermediate3,
            topk_weights_flat,
            sorted_token_ids,
            expert_ids,
            num_post,
            mul_routed_weight=True,
            top_k=1,  # FC2 input already has one row per topk slot
        )

        # Reduce over topk: [M, topk, K] → [M, K]
        output = intermediate3.view(M, topk, K).float().sum(dim=1).to(hidden_states.dtype)
        return output

    else:
        # ── Mode 2: already-permuted tokens (eager / dispatcher path) ──
        assert tokens_per_expert is not None
        tokens_per_expert = tokens_per_expert.int().cuda()

        total_tokens = hidden_states.size(0)
        N = fc1_weight.size(1)
        K = fc1_weight.size(2)

        # Build trivial indirection: tokens are already in expert order
        sorted_token_ids = torch.arange(total_tokens, dtype=torch.int32, device=hidden_states.device)
        # Pad each expert group to BLOCK_SIZE_M
        padded_counts = ((tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * BLOCK_SIZE_M
        padded_offsets = torch.zeros(num_local_experts + 1, dtype=torch.int32, device=hidden_states.device)
        padded_offsets[1:] = torch.cumsum(padded_counts, 0)
        total_padded = padded_offsets[-1].item()

        padded_sorted = torch.full(
            (total_padded,), total_tokens, dtype=torch.int32, device=hidden_states.device
        )
        expert_ids_out = torch.full(
            (total_padded // BLOCK_SIZE_M,), -1, dtype=torch.int32, device=hidden_states.device
        )

        unpadded_offsets = torch.zeros(num_local_experts + 1, dtype=torch.int32, device=hidden_states.device)
        unpadded_offsets[1:] = torch.cumsum(tokens_per_expert, 0)

        for e in range(num_local_experts):
            count = tokens_per_expert[e].item()
            padded = padded_counts[e].item()
            if padded == 0:
                continue
            src = unpadded_offsets[e].item()
            dst = padded_offsets[e].item()
            padded_sorted[dst : dst + count] = sorted_token_ids[src : src + count]
            num_blocks = padded // BLOCK_SIZE_M
            expert_ids_out[dst // BLOCK_SIZE_M : dst // BLOCK_SIZE_M + num_blocks] = e

        num_post = torch.tensor([total_padded], dtype=torch.int32, device=hidden_states.device)

        # Probs — already gathered; expand to per-token for the kernel
        if probs.dim() > 1:
            probs_flat = probs.squeeze(-1)
        else:
            probs_flat = probs

        # FC1: [total_tokens, K] → [total_tokens, N]  (top_k=1)
        intermediate1 = torch.zeros(
            total_tokens, N, dtype=hidden_states.dtype, device=hidden_states.device
        )
        _invoke_fused_moe_kernel(
            hidden_states,
            fc1_weight,
            intermediate1,
            probs_flat,
            padded_sorted,
            expert_ids_out,
            num_post,
            mul_routed_weight=False,
            top_k=1,
        )

        # Activation
        intermediate2 = torch.empty_like(intermediate1)
        _apply_activation(activation_type, intermediate2, intermediate1)

        # FC2: [total_tokens, N] → [total_tokens, K], with probs
        output = torch.zeros(
            total_tokens, K, dtype=hidden_states.dtype, device=hidden_states.device
        )
        _invoke_fused_moe_kernel(
            intermediate2,
            fc2_weight,
            output,
            probs_flat,
            padded_sorted,
            expert_ids_out,
            num_post,
            mul_routed_weight=True,
            top_k=1,
        )

        return output
