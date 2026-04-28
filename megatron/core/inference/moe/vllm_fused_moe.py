# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/vllm-project/vllm.
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
"""vLLM-style Triton fused MoE kernel (BF16) for Megatron inference.

CUDA-graph compatible: all indirection table construction happens on-device
via Triton kernels with fixed-size buffers and valid_tokens gating.
"""

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
from megatron.core.inference.moe.permute import (
    compute_expert_offsets,
    compute_local_tokens_per_expert,
)

# ---------------------------------------------------------------------------
# Triton kernel – BF16 grouped GEMM with indirect token addressing
# ---------------------------------------------------------------------------


def _select_block_size_m(max_tokens: int) -> int:
    """Select BLOCK_SIZE_M based on the token buffer size.

    Smaller tiles reduce padding waste in the indirection table when each
    expert sees few tokens (decode). Larger tiles improve compute density
    for large batches (prefill). Minimum is 16 (tl.dot requirement on NVIDIA).
    """
    if max_tokens <= 32:
        return 16
    if max_tokens <= 96:
        return 32
    if max_tokens <= 512:
        return 64
    return 128


# BLOCK_SIZE_M is NOT in these configs — it is selected on the Python side by
# _select_block_size_m and passed as a caller-provided constexpr. Each unique
# BLOCK_SIZE_M value triggers independent autotuning over these configs.
_AUTOTUNE_CONFIGS = [
    # GROUP_SIZE_M=1: better when each expert has few tokens (decode, sparse activation).
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=8, num_stages=4),
    # GROUP_SIZE_M=8: better for large batches where experts see many tokens.
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=5),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['N', 'K'])
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
    FUSE_SQUARED_RELU: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused MoE grouped GEMM with indirect token addressing.

    A: [num_tokens, K]           – input hidden states (unpermuted)
    B: [num_local_experts, N, K] – stacked expert weights
    C: [num_tokens * topk, N]    – output (one row per topk slot)

    When FUSE_SQUARED_RELU is True, applies squared_relu to the GEMM
    output in-register before writing, avoiding a separate activation pass.
    """
    pid = tl.program_id(0)

    # Compute group mapping from EM (host-side upper bound passed as a kernel
    # arg) so the compiler can resolve pid_m/pid_n without waiting on a global
    # memory load.  The device-side num_tokens_post_padded is loaded afterwards
    # only for the per-block early-exit check.
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

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

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

    if FUSE_SQUARED_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
        accumulator *= accumulator

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(tl.bfloat16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Indirection table construction (CUDA-graph safe, fully on-device)
# ---------------------------------------------------------------------------


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _init_sorted_ids_kernel(
    sorted_token_ids_ptr,
    expert_ids_ptr,
    max_sorted,
    max_blocks,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Initialize sorted_token_ids to SENTINEL and expert_ids to -1."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_token_ids_ptr + offs, SENTINEL, mask=offs < max_sorted)
    tl.store(expert_ids_ptr + offs, -1, mask=offs < max_blocks)


@triton.jit
def _scatter_token_indices_kernel(
    routing_map_ptr,
    sorted_token_ids_ptr,
    counters_ptr,
    valid_tokens_ptr,
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    max_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter local-expert pair indices into the padded indirection table.

    Only local expert pairs are written; non-local pairs are skipped (the
    _moe_sum kernel handles them by checking the routing map directly).
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    valid_pairs = valid_tokens * topk
    if pid * BLOCK_SIZE >= valid_pairs:
        return
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < valid_pairs

    eids = tl.load(routing_map_ptr + offs, mask=mask, other=-1)
    lids = eids - local_expert_start
    is_local = (lids >= 0) & (lids < num_local_experts) & mask

    local_pos = tl.atomic_add(counters_ptr + lids, 1, mask=is_local)
    tl.store(sorted_token_ids_ptr + local_pos, offs, mask=is_local)


@triton.jit
def _fill_expert_block_ids_kernel(
    expert_ids_ptr,
    exclusive_offsets_ptr,
    inclusive_offsets_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fill expert_ids with expert index for each BLOCK_SIZE_M block.

    Grid: one CTA per expert (parallelised across experts).
    Inner loop uses vectorised stores of BLOCK elements at a time.
    """
    e = tl.program_id(0)
    start_block = tl.load(exclusive_offsets_ptr + e) // BLOCK_SIZE_M
    end_block = tl.load(inclusive_offsets_ptr + e) // BLOCK_SIZE_M
    num_blocks = end_block - start_block
    for off in tl.range(0, num_blocks, BLOCK):
        idxs = start_block + off + tl.arange(0, BLOCK)
        tl.store(expert_ids_ptr + idxs, e, mask=idxs < end_block)




def _moe_align_block_size_cuda_graphable(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build indirection tables for the vLLM kernel, fully on-device.

    Replaces the original _moe_align_block_size which used .item() calls
    and host-side loops. All buffers are allocated at fixed max sizes so
    the function is safe for CUDA graph capture.

    Args:
        routing_map: [max_tokens, topk] expert assignments.
        block_size: BLOCK_SIZE_M for the vLLM kernel.
        num_local_experts: experts on this rank.
        local_expert_start: first global expert index on this rank.
        valid_tokens: scalar int32 CUDA tensor.

    Returns:
        sorted_token_ids: [max_sorted] int32 indirection table.
        expert_ids: [max_blocks] int32 expert per block.
        num_tokens_post_padded: [1] int32 (local expert padded count).
    """
    max_tokens, topk = routing_map.shape
    device = routing_map.device

    max_sorted = max_tokens * topk + block_size * (num_local_experts + 1)
    max_blocks = _ceil_div(max_sorted, block_size)
    sentinel = max_tokens * topk

    sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)

    INIT_BLOCK = 1024
    init_grid = _ceil_div(max(max_sorted, max_blocks), INIT_BLOCK)
    _init_sorted_ids_kernel[(init_grid,)](
        sorted_token_ids,
        expert_ids,
        max_sorted,
        max_blocks,
        SENTINEL=sentinel,
        BLOCK=INIT_BLOCK,
    )

    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts, valid_tokens
    )
    exclusive_offsets, inclusive_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=block_size
    )

    _fill_expert_block_ids_kernel[(num_local_experts,)](
        expert_ids,
        exclusive_offsets,
        inclusive_offsets,
        BLOCK_SIZE_M=block_size,
        BLOCK=128,
    )

    max_pairs = max_tokens * topk
    SCATTER_BLOCK = 256
    scatter_grid = _ceil_div(max_pairs, SCATTER_BLOCK)
    _scatter_token_indices_kernel[(scatter_grid,)](
        routing_map,
        sorted_token_ids,
        exclusive_offsets,
        valid_tokens,
        topk,
        local_expert_start,
        num_local_experts,
        max_pairs,
        BLOCK_SIZE=SCATTER_BLOCK,
    )

    num_tokens_post_padded = inclusive_offsets[-1:]
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ---------------------------------------------------------------------------
# Kernel launcher
# ---------------------------------------------------------------------------

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
    block_size_m: int,
    fuse_squared_relu: bool = False,
    grid_em: Optional[int] = None,
):
    """Launch the Triton fused-MoE kernel for one GEMM pass.

    grid_em: if provided, replaces sorted_token_ids.size(0) as the EM
    dimension for the grid, avoiding launching CTAs for unused buffer rows.
    """
    M = A.size(0)
    num_tokens = M * top_k
    EM = grid_em if grid_em is not None else sorted_token_ids.size(0)

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
        FUSE_SQUARED_RELU=fuse_squared_relu,
        top_k=top_k,
        BLOCK_SIZE_M=block_size_m,
    )


# ---------------------------------------------------------------------------
# Fused topk reduction (replaces torch.sum + copy)
# ---------------------------------------------------------------------------


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    valid_tokens_ptr,
    routing_map_ptr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    K,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    """Reduce topk dimension with valid_tokens gating.

    input:  [max_tokens * topk, K] bf16
    output: [max_tokens, K] bf16

    For token t < valid_tokens: output[t] = sum of input[t*topk+k] over
    topk slots k where the expert is local.  Non-local slots are skipped
    (their values in `input` are undefined because FC2 only processes
    local-expert blocks).
    For token t >= valid_tokens: output[t] = 0.
    Accumulation in fp32 for numerical accuracy.
    """
    token_id = tl.program_id(0).to(tl.int64)
    valid_tokens = tl.load(valid_tokens_ptr)
    is_valid = token_id < valid_tokens

    for k_idx in range(NUM_K_BLOCKS):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        if is_valid:
            base = token_id * topk * K
            for t in range(topk):
                eid = tl.load(routing_map_ptr + token_id * topk + t)
                lid = eid - local_expert_start
                if lid >= 0 and lid < num_local_experts:
                    v = tl.load(input_ptr + base + t * K + offs_k, mask=k_mask, other=0.0)
                    acc += v.to(tl.float32)

        tl.store(output_ptr + token_id * K + offs_k, acc.to(tl.bfloat16), mask=k_mask)


def _moe_sum(
    input: torch.Tensor,
    max_tokens: int,
    topk: int,
    K: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused topk reduction: [max_tokens*topk, K] → [max_tokens, K].

    Writes directly to `out` if provided (e.g. the RSV symmetric memory tensor),
    avoiding a separate allocation + copy. Rows beyond valid_tokens are zeroed.
    Only accumulates contributions from local experts; non-local topk slots are
    skipped (their values in `input` are undefined).
    """
    if out is None:
        out = torch.empty(max_tokens, K, dtype=input.dtype, device=input.device)
    BLOCK_K = min(triton.next_power_of_2(K), 1024)
    NUM_K_BLOCKS = _ceil_div(K, BLOCK_K)
    _moe_sum_kernel[(max_tokens,)](
        input,
        out,
        valid_tokens,
        routing_map,
        local_expert_start,
        num_local_experts,
        K,
        topk=topk,
        BLOCK_K=BLOCK_K,
        NUM_K_BLOCKS=NUM_K_BLOCKS,
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vllm_fused_moe(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_tokens_hint: Optional[int] = None,
) -> torch.Tensor:
    """Fused MoE using the vLLM Triton grouped-GEMM kernel (BF16).

    CUDA-graph compatible: indirection tables are built entirely on-device
    using fixed-size buffers gated by valid_tokens.

    Args:
        hidden_states: [max_tokens, hidden_size] BF16 input. Only the first
            valid_tokens rows are valid; the rest are ignored.
        probs: [max_tokens, topk] fp32 routing probabilities.
        fc1_weight: [num_local_experts, fc1_out, hidden_size] BF16.
        fc2_weight: [num_local_experts, hidden_size, fc1_out] BF16.
        activation_type: ActivationType enum.
        num_local_experts: experts on this rank.
        local_expert_start: first global expert index on this rank.
        valid_tokens: scalar int32 CUDA tensor with number of valid tokens.
        routing_map: [max_tokens, topk] int expert assignments.
        out: optional [max_tokens, hidden_size] bf16 output buffer (e.g. RSV
            symmetric memory tensor). If None, a local buffer is allocated.
        num_tokens_hint: optional host-side int with the expected number of
            valid tokens (e.g. batch_size * ep_size). Used to select a better
            BLOCK_SIZE_M instead of using the worst-case buffer size.

    Returns:
        [max_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"vllm_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )

    max_tokens = hidden_states.size(0)
    topk = routing_map.shape[1]
    effective_tokens = num_tokens_hint if num_tokens_hint is not None else max_tokens
    block_size_m = _select_block_size_m(effective_tokens)

    sorted_token_ids, expert_ids, num_post_padded = (
        _moe_align_block_size_cuda_graphable(
            routing_map, block_size_m, num_local_experts, local_expert_start, valid_tokens,
        )
    )
    num_valid = max_tokens * topk

    N = fc1_weight.size(1)
    K = fc1_weight.size(2)

    topk_weights_flat = probs.reshape(-1).contiguous()

    # FC1 + activation: [max_tokens, K] → [max_tokens*topk, N]
    assert activation_type == ActivationType.SQUARED_RELU
    intermediate1 = torch.empty(
        num_valid, N, dtype=hidden_states.dtype, device=hidden_states.device
    )
    _invoke_fused_moe_kernel(
        hidden_states,
        fc1_weight,
        intermediate1,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_post_padded,
        mul_routed_weight=False,
        top_k=topk,
        block_size_m=block_size_m,
        fuse_squared_relu=True,
    )

    # FC2: [max_tokens*topk, N] → [max_tokens*topk, K], with routing weights.
    # Only local-expert blocks are processed; non-local positions are left
    # undefined and skipped by _moe_sum (which checks the routing map).
    intermediate3 = torch.empty(
        num_valid, K, dtype=hidden_states.dtype, device=hidden_states.device
    )
    _invoke_fused_moe_kernel(
        intermediate1,
        fc2_weight,
        intermediate3,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_post_padded,
        mul_routed_weight=True,
        top_k=1,
        block_size_m=block_size_m,
    )

    # Reduce over topk: [max_tokens*topk, K] → [max_tokens, K]
    # Fused kernel accumulates in fp32, writes directly to out (if provided),
    # zeros rows beyond valid_tokens, and skips non-local expert slots.
    return _moe_sum(
        intermediate3, max_tokens, topk, K, valid_tokens,
        routing_map, local_expert_start, num_local_experts, out=out,
    )
