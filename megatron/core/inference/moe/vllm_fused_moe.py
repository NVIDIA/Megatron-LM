# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/vllm-project/vllm.
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
"""vLLM-style Triton fused MoE kernel (BF16) for Megatron inference.

CUDA-graph compatible: all indirection table construction happens on-device
via Triton kernels with fixed-size buffers and valid_tokens gating.
"""

import os
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

from megatron.core.inference.moe.activations import bounded_silu_mul
from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.permute import (
    _get_num_sms,
    compute_expert_offsets,
    compute_local_tokens_per_expert,
)

# ---------------------------------------------------------------------------
# Triton kernel – BF16 grouped GEMM with indirect token addressing
# ---------------------------------------------------------------------------


def _get_default_config(M: int, E: int, top_k: int) -> dict:
    """Pick BLOCK_SIZE_*, GROUP_SIZE_M, num_warps, num_stages from M, E, top_k.

    Mirrors vLLM's ``get_default_config`` (bf16/fp16 branch) verbatim:
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py

    M here is the host-side token-count hint (``num_tokens_hint`` in
    ``vllm_fused_moe``), NOT ``hidden_states.size(0)``. The hint is the
    expected per-step token count; the worst-case buffer size would over-tune
    for prefill on every decode step.

    Two intuitions drive the choices:
      1. Small M is memory-bound (favor tall/narrow tiles, more pipeline
         stages); large M is compute-bound (favor short/wide tiles, more warps).
      2. Padding tax dominates at small M — the indirection table pads M-tiles
         per expert, so small M-tiles minimize wasted rows.
    """
    # BLOCK_SIZE_M: shrink at small M to limit per-expert padding waste.
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128

    # BLOCK_SIZE_N: small M is memory-bound on weights, narrow N keeps weight
    # traffic in check; large M has enough FMAs per weight load for wider N.
    block_n = 64 if M <= 64 else 128

    # BLOCK_SIZE_K: small M needs depth in K to keep tensor cores fed; large M
    # already has enough M*N work, so shorter K reduces accumulator stall.
    block_k = 128 if M <= 64 else 64

    # GROUP_SIZE_M: tile-grouping for L2 reuse on weight tiles. Only profitable
    # when each expert sees enough adjacent M-tiles.
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1

    # num_warps: small M doesn't justify register pressure of more warps;
    # large M is compute-bound and feeds an MMA pipeline that wants more.
    num_warps = 4 if M <= 128 else 8

    # num_stages: extra prefetch only pays off when memory-bound (very small M).
    num_stages = 4 if M <= 32 else 3

    return {
        'BLOCK_SIZE_M': block_m,
        'BLOCK_SIZE_N': block_n,
        'BLOCK_SIZE_K': block_k,
        'GROUP_SIZE_M': group_m,
        'num_warps': num_warps,
        'num_stages': num_stages,
    }


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
    FUSE_SWIGLU: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused MoE grouped GEMM with indirect token addressing.

    Body mirrors vLLM's `fused_moe_kernel` verbatim except for the
    `FUSE_SQUARED_RELU` branch (Megatron applies relu+square in fp32 on
    the accumulator before the bf16 cast — strictly more accurate than
    upstream's separate post-FC1 activation kernel).

    Grid is sized host-side from `num_tokens_hint` (the typical-case token
    count), not the worst-case buffer length, so launch overhead at decode
    stays small. When the actual padded length exceeds the hinted grid
    size (rare prefill spikes), each CTA strides over multiple tiles via
    the outer `tl.range` loop.
    """
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    total_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_init = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for pid in tl.range(pid_init, total_tiles, grid_size):
        # GROUP_SIZE_M swizzle: pid → (pid_m, pid_n).  Mirrors upstream vLLM.
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Skip padding tiles whose expert slot was never assigned.  In
        # vLLM this also handles non-local experts via `write_zeros_to_output`;
        # our scatter excludes non-local pairs from `sorted_token_ids` entirely,
        # so `expert_id == -1` only fires on tail padding and we just skip.
        # (Triton's JIT does not support `continue`, so we gate the body.)
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts != -1:
            offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
            token_mask = offs_token < num_valid_tokens

            # `% N` keeps overflow lanes in-bounds; matching C-store mask drops
            # their contribution.  Saves a 2-D bounds check inside the K loop.
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

            a_ptrs = a_ptr + (
                offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )
            # SwiGLU epilogue fusion: the same [BLOCK_M, BLOCK_N] output tile of
            # the activated intermediate needs both the gate columns (offs_bn) and
            # the paired up columns (offs_bn + N), where N is the activated width.
            # A second accumulator reads the up half from the same B tensor.
            if FUSE_SWIGLU:
                b_up_ptrs = b_ptrs + N * stride_bn
                up_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                accumulator += tl.dot(a, b)
                if FUSE_SWIGLU:
                    b_up = tl.load(
                        b_up_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                    up_accumulator += tl.dot(a, b_up)
                    b_up_ptrs += BLOCK_SIZE_K * stride_bk
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            # Megatron-only: squared-relu fused on the fp32 accumulator before
            # the bf16 cast.  Upstream runs relu+square as a separate bf16 kernel.
            if FUSE_SQUARED_RELU:
                accumulator = tl.maximum(accumulator, 0.0)
                accumulator *= accumulator

            # SwiGLU fused on the fp32 accumulators: SiLU(gate) * up. The
            # arithmetic matches the standalone `bounded_silu_mul` kernel, but the
            # results are not bit-identical (measured max_abs 3.9e-5): that kernel
            # reads gate and up after they have been rounded to bf16, while these
            # accumulators have not been. The fused result is the less rounded of
            # the two. Removes that kernel and the 2N intermediate round-trip.
            if FUSE_SWIGLU:
                accumulator = accumulator * tl.sigmoid(accumulator) * up_accumulator

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
    block_start = pid * BLOCK
    if block_start < max_sorted or block_start < max_blocks:
        offs = block_start + tl.arange(0, BLOCK)
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


@triton.jit
def _prefix_fill_init_kernel(
    tokens_per_expert_ptr,  # [num_local_experts] raw token counts
    exclusive_offsets_ptr,  # [num_local_experts] out: exclusive prefix sum (scatter counters)
    inclusive_offsets_ptr,  # [num_local_experts] out: inclusive prefix sum
    expert_ids_ptr,  # [max_blocks] out: expert index per BLOCK_SIZE_M block
    sorted_token_ids_ptr,  # [max_sorted] out: SENTINEL-initialized indirection table
    num_local_experts,
    SENTINEL: tl.constexpr,
    ALIGNMENT: tl.constexpr,  # = BLOCK_SIZE_M
    NLE_POW2: tl.constexpr,  # next_power_of_2(num_local_experts) for tl.cumsum
    FILL_BLOCK: tl.constexpr,  # vectorized store width
):
    """Fused prefix-sum + expert_ids fill + sorted_token_ids SENTINEL init.

    Merges three kernels of the original indirection-table build
    (``_init_sorted_ids_kernel`` + ``_prefix_sum_kernel`` +
    ``_fill_expert_block_ids_kernel``) into one launch. Grid: one CTA per local
    expert. Every CTA recomputes the (tiny, num_local_experts-wide) aligned
    cumsum in registers, then CTA ``e`` writes only its own disjoint slices:

      - exclusive_offsets[e], inclusive_offsets[e]   (offset arrays)
      - expert_ids[excl_e//BM : incl_e//BM] = e      (block→expert map)
      - sorted_token_ids[excl_e : incl_e] = SENTINEL (padding for its block)

    The union of the per-expert [excl_e, incl_e) ranges is exactly
    [0, num_tokens_post_padded), which is the only region the grouped-GEMM
    reads (it strides tiles up to cdiv(num_tokens_post_padded, BM)); the buffer
    tail beyond that is never read, so it needs no initialization. The
    subsequent scatter overwrites the real token slots at the front of each
    expert's block, leaving the alignment padding at SENTINEL. Correctness is
    identical to the 3-kernel path; the only difference is fewer launches.
    """
    e = tl.program_id(0)
    r = tl.arange(0, NLE_POW2)
    mask = r < num_local_experts
    h = tl.load(tokens_per_expert_ptr + r, mask=mask, other=0)
    if ALIGNMENT > 1:
        h = tl.where(h > 0, ((h + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT, h)
    inc = tl.cumsum(h, axis=0)
    exc = inc - h
    # Select this CTA's own offsets by masked reduction over the lane axis.
    excl_e = tl.sum(tl.where(r == e, exc, 0))
    incl_e = tl.sum(tl.where(r == e, inc, 0))
    tl.store(exclusive_offsets_ptr + e, excl_e)
    tl.store(inclusive_offsets_ptr + e, incl_e)

    start_block = excl_e // ALIGNMENT
    end_block = incl_e // ALIGNMENT
    for off in tl.range(start_block, end_block, FILL_BLOCK):
        idxs = off + tl.arange(0, FILL_BLOCK)
        tl.store(expert_ids_ptr + idxs, e, mask=idxs < end_block)

    for off in tl.range(excl_e, incl_e, FILL_BLOCK):
        idxs = off + tl.arange(0, FILL_BLOCK)
        tl.store(sorted_token_ids_ptr + idxs, SENTINEL, mask=idxs < incl_e)


@triton.jit
def _count_prefix_fill_init_kernel(
    routing_map_ptr,  # [max_tokens, topk] expert assignments (global IDs)
    valid_tokens_ptr,  # scalar int32 CUDA tensor: valid tokens this iteration
    exclusive_offsets_ptr,  # [num_local_experts] out: exclusive prefix sum (scatter counters)
    inclusive_offsets_ptr,  # [num_local_experts] out: inclusive prefix sum
    expert_ids_ptr,  # [max_blocks] out: expert index per BLOCK_SIZE_M block
    sorted_token_ids_ptr,  # [max_sorted] out: SENTINEL-initialized indirection table
    num_local_experts,
    local_expert_start,
    topk: tl.constexpr,
    SENTINEL: tl.constexpr,
    ALIGNMENT: tl.constexpr,  # = BLOCK_SIZE_M
    NUM_BINS: tl.constexpr,  # next_power_of_2(num_local_experts + 1)
    COUNT_BLOCK: tl.constexpr,  # pairs histogrammed per iteration
    FILL_BLOCK: tl.constexpr,  # vectorized store width
):
    """``_prefix_fill_init_kernel`` with the per-expert token count folded in.

    ``_prefix_fill_init_kernel`` already has every CTA redundantly recompute the
    tiny ``num_local_experts``-wide cumsum in registers, and its only input is
    the count vector. That vector costs two extra launches to produce: a
    ``torch.zeros`` fill for the counters plus
    ``_count_local_tokens_kernel_persistent``. Recomputing the histogram
    redundantly in every CTA removes both launches.

    The count kernel is far more expensive than its work implies (measured 7.52
    us of device time in the BS256 decode graph, against a 0.72 us empty-kernel
    floor, to bucket 2048 int32s): with ``BLOCK_SIZE=1024`` and 2048 valid pairs
    only 2 of its 152 CTAs receive any work, and each issues 1024 global atomics
    contending on 32 counters. Here each CTA instead accumulates a private
    register histogram over the same pairs, so there are no global atomics and
    no cross-CTA ordering at all.

    Counts are integer-identical to the atomic path, so the whole indirection
    table -- and therefore the GEMM output -- is unchanged.

    Redundant work is the trade: every CTA reads all ``valid_pairs`` entries, so
    the read volume is ``num_local_experts x`` the atomic path's. At decode that
    is 32 x 8 KB from L2 and free; the caller restricts this variant to the
    decode regime for that reason.
    """
    e = tl.program_id(0)
    valid_pairs = tl.load(valid_tokens_ptr) * topk

    # Private per-CTA histogram of local expert IDs. Non-local and out-of-range
    # pairs are folded into the dropped bin `num_local_experts`, which NUM_BINS
    # is sized to hold and the mask below discards.
    acc = tl.zeros([NUM_BINS], dtype=tl.int32)
    for base in tl.range(0, valid_pairs, COUNT_BLOCK):
        offs = base + tl.arange(0, COUNT_BLOCK)
        m = offs < valid_pairs
        eids = tl.load(routing_map_ptr + offs, mask=m, other=-1)
        lids = eids - local_expert_start
        is_local = (lids >= 0) & (lids < num_local_experts) & m
        # routing_map is int64; tl.histogram wants int32 bin indices.
        bins = tl.where(is_local, lids, num_local_experts).to(tl.int32)
        acc += tl.histogram(bins, NUM_BINS)

    r = tl.arange(0, NUM_BINS)
    mask = r < num_local_experts
    h = tl.where(mask, acc, 0)
    if ALIGNMENT > 1:
        h = tl.where(h > 0, ((h + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT, h)
    inc = tl.cumsum(h, axis=0)
    exc = inc - h
    excl_e = tl.sum(tl.where(r == e, exc, 0))
    incl_e = tl.sum(tl.where(r == e, inc, 0))
    tl.store(exclusive_offsets_ptr + e, excl_e)
    tl.store(inclusive_offsets_ptr + e, incl_e)

    start_block = excl_e // ALIGNMENT
    end_block = incl_e // ALIGNMENT
    for off in tl.range(start_block, end_block, FILL_BLOCK):
        idxs = off + tl.arange(0, FILL_BLOCK)
        tl.store(expert_ids_ptr + idxs, e, mask=idxs < end_block)

    for off in tl.range(excl_e, incl_e, FILL_BLOCK):
        idxs = off + tl.arange(0, FILL_BLOCK)
        tl.store(sorted_token_ids_ptr + idxs, SENTINEL, mask=idxs < incl_e)


@triton.jit
def _align_single_kernel(
    routing_map_ptr,  # [max_tokens, topk] expert assignments (global IDs)
    valid_tokens_ptr,  # scalar int32 CUDA tensor: valid tokens this iteration
    exclusive_offsets_ptr,  # [num_local_experts] out: exclusive prefix sum
    inclusive_offsets_ptr,  # [num_local_experts] out: inclusive prefix sum
    expert_ids_ptr,  # [max_blocks] out: expert index per BLOCK_SIZE_M block
    sorted_token_ids_ptr,  # [max_sorted] out: indirection table
    num_local_experts,
    local_expert_start,
    topk: tl.constexpr,
    SENTINEL: tl.constexpr,
    ALIGNMENT: tl.constexpr,  # = BLOCK_SIZE_M
    NUM_BINS: tl.constexpr,  # next_power_of_2(num_local_experts + 1)
    COUNT_BLOCK: tl.constexpr,  # pairs processed per iteration
    FILL_BLOCK: tl.constexpr,  # vectorized store width
):
    """``_count_prefix_fill_init_kernel`` with the scatter folded in: the whole
    indirection-table build in one launch.

    CTA ``e`` owns local expert ``e`` and the disjoint output slice
    ``[excl_e, incl_e)``. It already streams every valid pair once to build its
    private histogram; a second streaming pass over the same (L2-resident, 16 KB
    at decode) pairs lets it place its own pairs itself. Within a pass the
    destination is ``excl_e + written_so_far + exclusive_cumsum(is_mine)``, so
    no global atomics and no cross-CTA ordering are involved, and the resulting
    table is deterministic (ascending pair index within each expert block)
    rather than atomically permuted.

    The second pass is why this is not free: it doubles the redundant read of
    the routing map. The pairs are tiny and cached, so the trade is one removed
    launch (``_scatter_token_indices_kernel``, measured 2.66 us + 0.55 us of
    dispatch gap per layer in the BS256 decode graph) against that extra pass.

    Row order within an expert block does not affect the result: each row of
    ``sorted_token_ids`` is a ``token * topk + slot`` pair index, and the FC2
    output is indexed by that same pair index, so any permutation within a
    block produces identical values downstream.
    """
    e = tl.program_id(0)
    valid_pairs = tl.load(valid_tokens_ptr) * topk

    acc = tl.zeros([NUM_BINS], dtype=tl.int32)
    for base in tl.range(0, valid_pairs, COUNT_BLOCK):
        offs = base + tl.arange(0, COUNT_BLOCK)
        m = offs < valid_pairs
        eids = tl.load(routing_map_ptr + offs, mask=m, other=-1)
        lids = eids - local_expert_start
        is_local = (lids >= 0) & (lids < num_local_experts) & m
        bins = tl.where(is_local, lids, num_local_experts).to(tl.int32)
        acc += tl.histogram(bins, NUM_BINS)

    r = tl.arange(0, NUM_BINS)
    mask = r < num_local_experts
    h = tl.where(mask, acc, 0)
    if ALIGNMENT > 1:
        h = tl.where(h > 0, ((h + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT, h)
    inc = tl.cumsum(h, axis=0)
    exc = inc - h
    excl_e = tl.sum(tl.where(r == e, exc, 0))
    incl_e = tl.sum(tl.where(r == e, inc, 0))
    tl.store(exclusive_offsets_ptr + e, excl_e)
    tl.store(inclusive_offsets_ptr + e, incl_e)

    start_block = excl_e // ALIGNMENT
    end_block = incl_e // ALIGNMENT
    for off in tl.range(start_block, end_block, FILL_BLOCK):
        idxs = off + tl.arange(0, FILL_BLOCK)
        tl.store(expert_ids_ptr + idxs, e, mask=idxs < end_block)

    # SENTINEL-fill this expert's slice first; the scatter below overwrites the
    # real pairs at the front of it and leaves the alignment padding sentinel.
    for off in tl.range(excl_e, incl_e, FILL_BLOCK):
        idxs = off + tl.arange(0, FILL_BLOCK)
        tl.store(sorted_token_ids_ptr + idxs, SENTINEL, mask=idxs < incl_e)

    pos = excl_e
    for base in tl.range(0, valid_pairs, COUNT_BLOCK):
        offs = base + tl.arange(0, COUNT_BLOCK)
        m = offs < valid_pairs
        eids = tl.load(routing_map_ptr + offs, mask=m, other=-1)
        is_mine = (eids - local_expert_start == e) & m
        sel = is_mine.to(tl.int32)
        rank = tl.cumsum(sel, axis=0) - sel
        tl.store(sorted_token_ids_ptr + pos + rank, offs, mask=is_mine)
        pos += tl.sum(sel)


# Fuse init + prefix-sum + expert-block fill (3 tiny serial kernels) into one
# launch in the decode indirection-table build. Env-toggleable for A/B; the
# fused path is numerically identical to the default (only launch count differs).
_USE_FUSED_ALIGN: bool = os.environ.get("MCORE_MOE_FUSED_ALIGN", "0") == "1"

# Additionally fold the per-expert token count (and the `torch.zeros` fill of its
# counter buffer) into that same launch, taking the decode indirection-table
# build from 4 kernels to 2. Requires MCORE_MOE_FUSED_ALIGN. Integer-exact.
_USE_FUSED_COUNT: bool = os.environ.get("MCORE_MOE_FUSED_COUNT", "0") == "1"

# Above this token hint the redundant per-CTA histogram stops being free, so hand
# back to the atomic count kernel, whose read volume does not scale with the
# expert count. 512 matches the decode regime `_get_decode_tuned_configs` targets.
_FUSED_COUNT_MAX_TOKENS = 512

# Additionally fold the scatter into that same launch, taking the decode
# indirection-table build from 2 kernels to 1. Requires MCORE_MOE_FUSED_COUNT.
# Produces the same table up to row order within an expert block, which the
# downstream pair-indexed GEMM is invariant to.
_USE_FUSED_SCATTER: bool = os.environ.get("MCORE_MOE_FUSED_SCATTER", "0") == "1"


def _moe_align_block_size_fused(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-kernel indirection-table build (count -> prefix_fill_init -> scatter).

    Drop-in replacement for ``_moe_align_block_size_cuda_graphable`` that folds
    the separate init + prefix-sum + fill kernels into ``_prefix_fill_init_kernel``.
    Returns the same (sorted_token_ids, expert_ids, num_tokens_post_padded).
    """
    max_tokens, topk = routing_map.shape
    device = routing_map.device

    max_sorted = max_tokens * topk + block_size * (num_local_experts + 1)
    max_blocks = _ceil_div(max_sorted, block_size)
    sentinel = max_tokens * topk

    sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)

    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts, valid_tokens, persistent=True
    )
    exclusive_offsets = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    inclusive_offsets = torch.empty(num_local_experts, dtype=torch.int32, device=device)

    _prefix_fill_init_kernel[(num_local_experts,)](
        tokens_per_expert,
        exclusive_offsets,
        inclusive_offsets,
        expert_ids,
        sorted_token_ids,
        num_local_experts,
        SENTINEL=sentinel,
        ALIGNMENT=block_size,
        NLE_POW2=triton.next_power_of_2(num_local_experts),
        FILL_BLOCK=128,
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


def _moe_align_block_size_count_fused(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2-kernel indirection-table build (count_prefix_fill_init -> scatter).

    Same contract and same outputs as ``_moe_align_block_size_fused``; it just
    folds the token count and its counter-buffer zero-fill into the first launch.
    """
    max_tokens, topk = routing_map.shape
    device = routing_map.device

    max_sorted = max_tokens * topk + block_size * (num_local_experts + 1)
    max_blocks = _ceil_div(max_sorted, block_size)
    sentinel = max_tokens * topk

    sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    exclusive_offsets = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    inclusive_offsets = torch.empty(num_local_experts, dtype=torch.int32, device=device)

    _count_prefix_fill_init_kernel[(num_local_experts,)](
        routing_map,
        valid_tokens,
        exclusive_offsets,
        inclusive_offsets,
        expert_ids,
        sorted_token_ids,
        num_local_experts,
        local_expert_start,
        topk=topk,
        SENTINEL=sentinel,
        ALIGNMENT=block_size,
        NUM_BINS=triton.next_power_of_2(num_local_experts + 1),
        COUNT_BLOCK=1024,
        FILL_BLOCK=128,
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


def _moe_align_block_size_single(
    routing_map: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1-kernel indirection-table build.

    Same contract and same outputs as ``_moe_align_block_size_count_fused``,
    with the scatter folded into the single launch as well. Row order within an
    expert block is ascending pair index instead of atomically permuted; the
    downstream GEMM indexes rows by pair id, so the result is unchanged.
    """
    max_tokens, topk = routing_map.shape
    device = routing_map.device

    max_sorted = max_tokens * topk + block_size * (num_local_experts + 1)
    max_blocks = _ceil_div(max_sorted, block_size)
    sentinel = max_tokens * topk

    sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    exclusive_offsets = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    inclusive_offsets = torch.empty(num_local_experts, dtype=torch.int32, device=device)

    _align_single_kernel[(num_local_experts,)](
        routing_map,
        valid_tokens,
        exclusive_offsets,
        inclusive_offsets,
        expert_ids,
        sorted_token_ids,
        num_local_experts,
        local_expert_start,
        topk=topk,
        SENTINEL=sentinel,
        ALIGNMENT=block_size,
        NUM_BINS=triton.next_power_of_2(num_local_experts + 1),
        COUNT_BLOCK=1024,
        FILL_BLOCK=128,
    )

    num_tokens_post_padded = inclusive_offsets[-1:]
    return sorted_token_ids, expert_ids, num_tokens_post_padded


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
        sorted_token_ids, expert_ids, max_sorted, max_blocks, SENTINEL=sentinel, BLOCK=INIT_BLOCK
    )

    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts, valid_tokens, persistent=True
    )
    exclusive_offsets, inclusive_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=block_size
    )

    _fill_expert_block_ids_kernel[(num_local_experts,)](
        expert_ids, exclusive_offsets, inclusive_offsets, BLOCK_SIZE_M=block_size, BLOCK=128
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
    config: dict,
    grid_size: int,
    fuse_squared_relu: bool = False,
    fuse_swiglu: bool = False,
):
    """Launch the Triton fused-MoE kernel for one GEMM pass.

    Body matches upstream vLLM `fused_moe_kernel` (1 CTA per (pid_m, pid_n)
    tile, raw pointer arithmetic with `% N` on the N axis), apart from the
    optional fused squared-relu activation in fp32.

    When ``fuse_swiglu`` is set, B is the [E, 2*Nf, K] gate|up FC1 weight and the
    kernel emits the activated [num_valid, Nf] intermediate directly (SiLU(gate) *
    up), so ``N`` is the activated width ``Nf = B.size(1) // 2`` and each program
    also reads the paired up columns at ``+N`` in B.

    `grid_size` is sized host-side from `num_tokens_hint` so launch overhead
    at decode is small.  When the actual padded length exceeds the hinted
    grid size, each CTA strides over additional tiles via the kernel's outer
    `tl.range`.  The full launch config (tile sizes, warps, stages) is picked
    host-side by ``_get_default_config`` from M = num_tokens_hint.
    """
    M = A.size(0)
    num_tokens = M * top_k
    # For fused SwiGLU, N is the activated width (half the 2*Nf FC1 output).
    n_dim = B.size(1) // 2 if fuse_swiglu else B.size(1)

    _fused_moe_kernel[(grid_size,)](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n_dim,
        B.size(2),
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
        FUSE_SWIGLU=fuse_swiglu,
        top_k=top_k,
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'],
        GROUP_SIZE_M=config['GROUP_SIZE_M'],
        num_warps=config['num_warps'],
        num_stages=config['num_stages'],
    )


# ---------------------------------------------------------------------------
# Fused topk reduction (replaces torch.sum + copy)
# ---------------------------------------------------------------------------


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    topk_weights_ptr,
    valid_tokens_ptr,
    routing_map_ptr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    K,
    topk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    """Reduce topk dimension with routing weight application.

    input:  [max_tokens * topk, K] bf16
    output: [max_tokens, K] — dtype matches the output buffer (fp32 or bf16)

    For token t < valid_tokens: output[t] = sum of input[t*topk+k] * prob[t*topk+k]
    over topk slots k where the expert is local.  Non-local slots are skipped
    (their values in `input` are undefined because FC2 only processes
    local-expert blocks).
    Rows for t >= valid_tokens are not written; downstream consumers
    (e.g. reduce-scatter-v) only read the first valid_tokens rows.
    Routing weight multiplication and accumulation in fp32 for numerical accuracy.

    Persistent grid: launches BLOCK_M CTAs that stride over valid_tokens.
    CUDA-graph safe (grid is static); the loop bound is loaded device-side.
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)

    for token_id in tl.range(pid, valid_tokens, BLOCK_M):
        token_id_i64 = token_id.to(tl.int64)
        base = token_id_i64 * topk * K

        # k_idx outer / topk inner keeps the live accumulator at one BLOCK_K tile.
        # Swapping (topk outer) would need NUM_K_BLOCKS persistent accumulators
        # (~NUM_K_BLOCKS * BLOCK_K * 4 B), which spills / cuts occupancy at large K.
        for k_idx in range(NUM_K_BLOCKS):
            offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            acc = tl.zeros([BLOCK_K], dtype=tl.float32)
            for t in range(topk):
                eid = tl.load(routing_map_ptr + token_id * topk + t)
                lid = eid - local_expert_start
                if lid >= 0 and lid < num_local_experts:
                    v = tl.load(input_ptr + base + t * K + offs_k, mask=k_mask, other=0.0)
                    w = tl.load(topk_weights_ptr + token_id * topk + t)
                    acc += v.to(tl.float32) * w

            tl.store(output_ptr + token_id_i64 * K + offs_k, acc, mask=k_mask)


def _moe_sum(
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    max_tokens: int,
    topk: int,
    K: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused topk reduction: [max_tokens*topk, K] bf16 → [max_tokens, K].

    Applies routing weights and reduces over topk in a single kernel.
    Accumulates in fp32. When `out` is None, allocates and returns an fp32
    buffer. When `out` is provided (e.g. the RSV symmetric memory tensor),
    writes directly into it — tl.store handles the cast to the buffer's dtype.
    Only writes the first valid_tokens rows; rows beyond are left untouched
    (downstream RSV reads only the valid range). Only accumulates contributions
    from local experts; non-local topk slots are skipped (their values in
    `input` are undefined).
    """
    if out is None:
        out = torch.empty(max_tokens, K, dtype=torch.float32, device=input.device)
    BLOCK_K = min(triton.next_power_of_2(K), 1024)
    NUM_K_BLOCKS = _ceil_div(K, BLOCK_K)
    BLOCK_M = _get_num_sms(input.device)
    _moe_sum_kernel[(BLOCK_M,)](
        input,
        out,
        topk_weights,
        valid_tokens,
        routing_map,
        local_expert_start,
        num_local_experts,
        K,
        topk=topk,
        BLOCK_M=BLOCK_M,
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
    fuse_fc1_activation: bool = False,
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
        out: optional [max_tokens, hidden_size] output buffer (e.g. the RSV
            symmetric memory tensor). If None, an fp32 buffer is allocated.
            When provided, tl.store casts to the buffer's dtype automatically.
        num_tokens_hint: optional host-side int with the expected number of
            valid tokens (e.g. batch_size * ep_size). Used to select a better
            BLOCK_SIZE_M instead of using the worst-case buffer size.
        fuse_fc1_activation: for SwiGLU, fuse SiLU(gate)*up into the FC1 GEMM
            epilogue instead of running the standalone ``bounded_silu_mul`` kernel
            over the 2N-wide intermediate. Ignored for other activations.

    Returns:
        [max_tokens, hidden_size] output (fp32 when out=None, else out's dtype).
        tl.store handles the implicit cast when out is a different dtype.
    """
    assert (
        hidden_states.dtype == torch.bfloat16
    ), f"vllm_fused_moe requires bf16 input, got {hidden_states.dtype}"

    max_tokens = hidden_states.size(0)
    topk = routing_map.shape[1]
    effective_tokens = num_tokens_hint if num_tokens_hint is not None else max_tokens

    # Mirror upstream vLLM: pick the full launch config (tile sizes, warps,
    # stages) host-side from the token-count hint, not from the worst-case
    # buffer size. Same config is used for both FC1 and FC2 (matches vLLM).
    config = _get_default_config(M=effective_tokens, E=num_local_experts, top_k=topk)

    if _USE_FUSED_ALIGN and _USE_FUSED_COUNT and effective_tokens <= _FUSED_COUNT_MAX_TOKENS:
        if _USE_FUSED_SCATTER:
            align_fn = _moe_align_block_size_single
        else:
            align_fn = _moe_align_block_size_count_fused
    elif _USE_FUSED_ALIGN:
        align_fn = _moe_align_block_size_fused
    else:
        align_fn = _moe_align_block_size_cuda_graphable
    sorted_token_ids, expert_ids, num_post_padded = align_fn(
        routing_map, config['BLOCK_SIZE_M'], num_local_experts, local_expert_start, valid_tokens
    )
    num_valid = max_tokens * topk

    N = fc1_weight.size(1)
    K = fc1_weight.size(2)

    assert activation_type in (ActivationType.SQUARED_RELU, ActivationType.SWIGLU)
    is_swiglu = activation_type == ActivationType.SWIGLU
    # When enabled for SwiGLU, FC1 fuses SiLU(gate)*up into its epilogue and
    # writes the activated [num_valid, N//2] intermediate directly, removing the
    # standalone bounded_silu_mul kernel and the 2N intermediate round-trip.
    use_fused_swiglu = is_swiglu and fuse_fc1_activation
    fc1_out_width = N // 2 if use_fused_swiglu else N

    # Grid sized for the typical-case token count (num_tokens_hint).  When the
    # actual num_tokens_post_padded exceeds this, the kernel's outer tl.range
    # makes each CTA stride over additional tiles — correct but with reduced
    # parallelism on rare prefill spikes.  EM hint = effective_tokens*topk +
    # BLOCK_SIZE_M*num_local_experts upper-bounds the per-expert padding.
    block_m = config['BLOCK_SIZE_M']
    em_hint = effective_tokens * topk + block_m * num_local_experts
    num_pid_m_hint = _ceil_div(em_hint, block_m)
    num_pid_n_fc1 = _ceil_div(fc1_out_width, config['BLOCK_SIZE_N'])
    num_pid_n_fc2 = _ceil_div(K, config['BLOCK_SIZE_N'])
    grid_size_fc1 = num_pid_m_hint * num_pid_n_fc1
    grid_size_fc2 = num_pid_m_hint * num_pid_n_fc2

    topk_weights_flat = probs.reshape(-1).contiguous()

    # FC1 + activation: [max_tokens, K] → [max_tokens*topk, fc1_out_width]
    # SQUARED_RELU fuses into the GEMM epilogue (elementwise). SwiGLU pairs column
    # c with c+N/2 across N-tiles; the unfused path writes the 2N intermediate and
    # applies gate/up separately below, while the fused path (use_fused_swiglu)
    # reads both halves per program and writes the N//2 activated output directly.
    intermediate1 = torch.empty(
        num_valid, fc1_out_width, dtype=hidden_states.dtype, device=hidden_states.device
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
        config=config,
        grid_size=grid_size_fc1,
        fuse_squared_relu=not is_swiglu,
        fuse_swiglu=use_fused_swiglu,
    )
    if is_swiglu and not use_fused_swiglu:
        # intermediate1 is [num_valid, 2N] (gate | up); reduce to [num_valid, N] via
        # SiLU(gate) * up over the valid_tokens*topk live rows only.
        n_rows = (valid_tokens * topk).to(torch.int32)
        intermediate1 = bounded_silu_mul(intermediate1, n_rows)

    # FC2: [max_tokens*topk, N] → [max_tokens*topk, K], without routing weights.
    # Routing weights are applied in the reduction kernel to avoid an extra
    # bf16 truncation of prob-scaled values before the topk summation.
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
        mul_routed_weight=False,
        top_k=1,
        config=config,
        grid_size=grid_size_fc2,
    )

    # Reduce over topk: [max_tokens*topk, K] → [max_tokens, K]
    # Applies routing weights and accumulates in fp32, writes directly to
    # out (if provided), zeros rows beyond valid_tokens, and skips non-local
    # expert slots.
    return _moe_sum(
        intermediate3,
        probs,
        max_tokens,
        topk,
        K,
        valid_tokens,
        routing_map,
        local_expert_start,
        num_local_experts,
        out=out,
    )
