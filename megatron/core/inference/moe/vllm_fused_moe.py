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
from megatron.core.inference.moe.permute import compute_expert_offsets, compute_local_tokens_per_expert

# ---------------------------------------------------------------------------
# Triton kernel – BF16 grouped GEMM with indirect token addressing
# ---------------------------------------------------------------------------

BLOCK_SIZE_M = 64

_AUTOTUNE_CONFIGS = [
    # GROUP_SIZE_M=1: better when each expert has few tokens (decode, sparse activation).
    # With few tokens per expert, adjacent M-blocks belong to different experts, so
    # grouping them for L2 weight-tile reuse is counterproductive.
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_warps=8, num_stages=3),
    # GROUP_SIZE_M=8: better for large batches where experts see many tokens and
    # adjacent M-blocks can share weight tiles in L2 cache.
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=4),
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
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

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
    nonlocal_counter_ptr,
    valid_tokens_ptr,
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    max_pairs,
    NUM_BLOCKS: tl.constexpr,
):
    """Scatter flat token indices into the padded indirection table.

    For each valid (token, topk) pair, atomically claim a position and write
    the flat index. Local expert pairs go to their expert's padded block;
    non-local expert pairs go to a trailing section (expert_ids stays -1,
    so the GEMM kernel writes zeros for those output positions).
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    valid_pairs = valid_tokens * topk
    if pid >= valid_pairs:
        return
    for pair in tl.range(pid, max_pairs, NUM_BLOCKS):
        tok = pair // topk
        if tok < valid_tokens:
            k = pair % topk
            eid = tl.load(routing_map_ptr + tok * topk + k)
            lid = eid - local_expert_start
            if lid >= 0 and lid < num_local_experts:
                pos = tl.atomic_add(counters_ptr + lid, 1)
                tl.store(sorted_token_ids_ptr + pos, pair)
            else:
                pos = tl.atomic_add(nonlocal_counter_ptr, 1)
                tl.store(sorted_token_ids_ptr + pos, pair)


@triton.jit
def _fill_expert_block_ids_kernel(
    expert_ids_ptr,
    exclusive_offsets_ptr,
    inclusive_offsets_ptr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Fill expert_ids with expert index for each BLOCK_SIZE_M block."""
    for e in range(NUM_LOCAL_EXPERTS):
        start_block = tl.load(exclusive_offsets_ptr + e) // BLOCK_SIZE_M
        end_block = tl.load(inclusive_offsets_ptr + e) // BLOCK_SIZE_M
        for b in tl.range(start_block, end_block):
            tl.store(expert_ids_ptr + b, e)


def _moe_align_block_size_cuda_graphable(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        num_tokens_post_padded_local: [1] int32 (local expert padded count).
        num_tokens_post_padded_all: [1] int32 (including non-local expert slots).
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

    _fill_expert_block_ids_kernel[(1,)](
        expert_ids,
        exclusive_offsets,
        inclusive_offsets,
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE_M=block_size,
    )

    counters = exclusive_offsets.clone()
    nonlocal_counter = inclusive_offsets[-1:].clone()
    max_pairs = max_tokens * topk
    NUM_BLOCKS = min(max_pairs, 512)
    _scatter_token_indices_kernel[(NUM_BLOCKS,)](
        routing_map,
        sorted_token_ids,
        counters,
        nonlocal_counter,
        valid_tokens,
        topk,
        local_expert_start,
        num_local_experts,
        max_pairs,
        NUM_BLOCKS=NUM_BLOCKS,
    )

    num_tokens_post_padded_local = inclusive_offsets[-1:]
    num_tokens_post_padded_all = torch.full(
        (1,), max_sorted, dtype=torch.int32, device=device
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded_local, num_tokens_post_padded_all


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
    fuse_squared_relu: bool = False,
):
    """Launch the Triton fused-MoE kernel for one GEMM pass."""
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
        FUSE_SQUARED_RELU=fuse_squared_relu,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Fused topk reduction (replaces torch.sum + copy)
# ---------------------------------------------------------------------------


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    valid_tokens_ptr,
    K,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    """Reduce topk dimension with valid_tokens gating.

    input:  [max_tokens * topk, K] bf16
    output: [max_tokens, K] bf16

    For token t < valid_tokens: output[t] = sum(input[t*topk : (t+1)*topk]).
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
                v = tl.load(input_ptr + base + t * K + offs_k, mask=k_mask, other=0.0)
                acc += v.to(tl.float32)

        tl.store(output_ptr + token_id * K + offs_k, acc.to(tl.bfloat16), mask=k_mask)


def _moe_sum(
    input: torch.Tensor,
    max_tokens: int,
    topk: int,
    K: int,
    valid_tokens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused topk reduction: [max_tokens*topk, K] → [max_tokens, K].

    Writes directly to `out` if provided (e.g. the RSV symmetric memory tensor),
    avoiding a separate allocation + copy. Rows beyond valid_tokens are zeroed.
    """
    if out is None:
        out = torch.empty(max_tokens, K, dtype=input.dtype, device=input.device)
    BLOCK_K = min(triton.next_power_of_2(K), 1024)
    NUM_K_BLOCKS = _ceil_div(K, BLOCK_K)
    _moe_sum_kernel[(max_tokens,)](
        input,
        out,
        valid_tokens,
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

    Returns:
        [max_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"vllm_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )

    max_tokens = hidden_states.size(0)
    topk = routing_map.shape[1]

    sorted_token_ids, expert_ids, num_post_local, num_post_all = (
        _moe_align_block_size_cuda_graphable(
            routing_map, BLOCK_SIZE_M, num_local_experts, local_expert_start, valid_tokens
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
        num_post_local,
        mul_routed_weight=False,
        top_k=topk,
        fuse_squared_relu=True,
    )

    # FC2: [max_tokens*topk, N] → [max_tokens*topk, K], with routing weights
    # num_post_all covers the full indirection table including non-local expert
    # pairs (expert_ids=-1), so the kernel writes zeros to those output positions.
    # This eliminates the need for torch.zeros on intermediate3.
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
        num_post_all,
        mul_routed_weight=True,
        top_k=1,
    )

    # Reduce over topk: [max_tokens*topk, K] → [max_tokens, K]
    # Fused kernel accumulates in fp32, writes directly to out (if provided),
    # and zeros rows beyond valid_tokens.
    return _moe_sum(intermediate3, max_tokens, topk, K, valid_tokens, out=out)
