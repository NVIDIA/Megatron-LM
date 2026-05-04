# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton kernels for token permutation and unpermutation in fused MoE.

Includes:
- Token counting per expert
- Expert offset computation (aligned prefix sums)
- Permute tokens into expert-grouped order
- Unpermute expert outputs back to original token order
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


_NUM_SMS: Optional[int] = None


def _get_num_sms(device: torch.device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _count_local_tokens_kernel(
    routing_map_ptr,  # [max_tokens, topk] flattened expert assignments
    tokens_per_expert_ptr,  # [num_local_experts] output counters (zeroed by caller)
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of valid tokens this iteration
    topk,  # number of expert choices per token
    local_expert_start,  # first global expert index owned by this rank
    num_local_experts: tl.constexpr,  # number of experts on this rank
    BLOCK_SIZE: tl.constexpr,  # number of pairs processed per program
):
    """Count tokens routed to experts on this rank, ignoring tokens routed elsewhere.

    Each program processes BLOCK_SIZE (token, topk) pairs. Tokens assigned to
    experts outside [local_expert_start, local_expert_start + num_local_experts)
    or beyond valid_tokens are silently skipped.

    Grid is launched at max size (max_tokens * topk); valid_tokens gates which
    pairs are actually processed — required for CUDA graph compatibility.
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    valid_pairs = valid_tokens * topk
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < valid_pairs
    expert_ids = tl.load(routing_map_ptr + offsets, mask=mask, other=-1)
    local_ids = expert_ids - local_expert_start
    is_local = (local_ids >= 0) & (local_ids < num_local_experts) & mask
    tl.atomic_add(tokens_per_expert_ptr + local_ids, 1, mask=is_local)


@triton.jit
def _count_local_tokens_kernel_persistent(
    routing_map_ptr,  # [max_tokens, topk] flattened expert assignments
    tokens_per_expert_ptr,  # [num_local_experts] output counters (zeroed by caller)
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of valid tokens this iteration
    topk,  # number of expert choices per token
    local_expert_start,  # first global expert index owned by this rank
    num_local_experts: tl.constexpr,  # number of experts on this rank
    num_sms,  # number of SMs (grid size for persistent kernel)
    BLOCK_SIZE: tl.constexpr,  # number of pairs processed per iteration
):
    """Count tokens routed to local experts using a persistent grid.

    Launches num_sms CTAs. Each CTA loops over its share of BLOCK_SIZE-sized
    chunks, with total work determined device-side from valid_tokens.
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    valid_pairs = valid_tokens * topk

    total_blocks = tl.cdiv(valid_pairs, BLOCK_SIZE)
    blocks_per_cta = tl.cdiv(total_blocks, num_sms)
    block_start = pid * blocks_per_cta

    if block_start < total_blocks:
        block_end = tl.minimum(block_start + blocks_per_cta, total_blocks)

        for block_id in tl.range(block_start, block_end):
            offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < valid_pairs
            expert_ids = tl.load(routing_map_ptr + offsets, mask=mask, other=-1)
            local_ids = expert_ids - local_expert_start
            is_local = (local_ids >= 0) & (local_ids < num_local_experts) & mask
            tl.atomic_add(tokens_per_expert_ptr + local_ids, 1, mask=is_local)


def compute_local_tokens_per_expert(
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    valid_tokens: torch.Tensor,
    persistent: bool = False,
) -> torch.Tensor:
    """Count tokens routed to each local expert.

    Args:
        routing_map: [max_tokens, topk] expert assignments. Only the first
            valid_tokens rows are processed; the rest are ignored.
        local_expert_start: first global expert index on this rank.
        num_local_experts: number of experts on this rank.
        valid_tokens: scalar int32 CUDA tensor with the number of valid tokens
            this iteration. Fixed address; value updated each step before graph replay.
        persistent: use persistent-grid kernel variant (fewer CTAs, looped).
    """
    max_pairs = routing_map.numel()
    topk = routing_map.shape[1]
    tokens_per_expert = torch.zeros(num_local_experts, dtype=torch.int32, device=routing_map.device)
    BLOCK = 1024
    if persistent:
        num_sms = _get_num_sms(routing_map.device)
        _count_local_tokens_kernel_persistent[(num_sms,)](
            routing_map,
            tokens_per_expert,
            valid_tokens,
            topk,
            local_expert_start,
            num_local_experts,
            num_sms,
            BLOCK_SIZE=BLOCK,
        )
    else:
        _count_local_tokens_kernel[(_ceil_div(max_pairs, BLOCK),)](
            routing_map,
            tokens_per_expert,
            valid_tokens,
            topk,
            local_expert_start,
            num_local_experts,
            BLOCK_SIZE=BLOCK,
        )
    return tokens_per_expert


@triton.jit
def _prefix_sum_kernel(
    tokens_per_expert_ptr,  # [num_local_experts] raw token counts
    exclusive_offsets_ptr,  # [num_local_experts] output: exclusive prefix sum of aligned counts
    inclusive_offsets_ptr,  # [num_local_experts] output: inclusive prefix sum of aligned counts
    num_local_experts,  # number of experts on this rank
    alignment: tl.constexpr,  # per-expert alignment (counts rounded up to this multiple)
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(num_local_experts) for tl.cumsum
):
    """Exclusive and inclusive prefix sums of aligned token counts.

    Each expert's token count is rounded up to the nearest multiple of
    `alignment` (experts with 0 tokens stay at 0). The inclusive offsets
    are used as `offs` by grouped_mm / scaled_grouped_mm.
    """
    r = tl.arange(0, BLOCK_SIZE)
    mask = r < num_local_experts
    h = tl.load(tokens_per_expert_ptr + r, mask=mask, other=0)
    # Round up non-zero counts to alignment boundary
    if alignment > 1:
        h = tl.where(h > 0, ((h + alignment - 1) // alignment) * alignment, h)
    inc = tl.cumsum(h, axis=0)
    tl.store(exclusive_offsets_ptr + r, inc - h, mask=mask)
    tl.store(inclusive_offsets_ptr + r, inc, mask=mask)


@triton.jit
def _init_permutation_map_kernel(
    perm_map_ptr,
    n_used_ptr,  # pointer to inclusive_expert_offsets[-1]: total used rows this iteration
    BLOCK_SIZE: tl.constexpr,
):
    """Initialize permutation_map entries to -1 up to n_used rows.

    Grid is launched at max size; entries beyond n_used are left untouched —
    the activation and unpermute kernels are gated by the same n_used pointer
    so they never read those entries.
    """
    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_used
    tl.store(perm_map_ptr + offsets, tl.full([BLOCK_SIZE], -1, tl.int32), mask=mask)


def init_permutation_map(permutation_map: torch.Tensor, n_used: torch.Tensor) -> None:
    """Fill permutation_map[0:n_used] with -1.

    Args:
        permutation_map: [output_size] int32 buffer (pre-allocated at max size).
        n_used: scalar int32 CUDA tensor = inclusive_expert_offsets[-1].
    """
    output_size = permutation_map.shape[0]
    BLOCK_SIZE = 1024
    _init_permutation_map_kernel[(_ceil_div(output_size, BLOCK_SIZE),)](
        permutation_map, n_used, BLOCK_SIZE=BLOCK_SIZE
    )


def compute_expert_offsets(tokens_per_expert: torch.Tensor, alignment: int = 1) -> tuple:
    """Compute exclusive and inclusive prefix sums of aligned token counts."""
    n = tokens_per_expert.shape[0]
    exclusive_cumsum = torch.empty_like(tokens_per_expert)
    inclusive_cumsum = torch.empty_like(tokens_per_expert)
    _prefix_sum_kernel[(1,)](
        tokens_per_expert,
        exclusive_cumsum,
        inclusive_cumsum,
        n,
        alignment,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return exclusive_cumsum, inclusive_cumsum


@triton.jit
def _permute_tokens_kernel(
    hidden_ptr,  # [max_tokens, hidden_dim] input hidden states
    probs_ptr,  # [max_tokens, topk] routing probabilities
    routing_map_ptr,  # [max_tokens, topk] expert assignments (global IDs)
    out_hidden_ptr,  # [output_size, hidden_dim] output: permuted hidden states
    out_probs_ptr,  # [output_size] output: permuted probabilities
    out_src_idx_ptr,  # [output_size] output: permutation_map (original token index, -1 for padding)
    counters_ptr,  # [num_local_experts] exclusive offsets, atomically incremented
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of valid tokens this iteration
    hidden_dim,  # hidden dimension
    max_pairs,  # max_tokens * topk (fixed for CG)
    topk: tl.constexpr,  # number of expert choices per token
    local_expert_start,  # first global expert index on this rank
    num_local_experts: tl.constexpr,  # number of experts on this rank
    BLOCK_H: tl.constexpr,  # tile size for copying hidden_dim
    NUM_BLOCKS: tl.constexpr,  # grid size (fixed for CG)
):
    """Permute tokens into expert-grouped order.

    Grid: fixed NUM_BLOCKS CTAs, each iterating over multiple (token, topk) pairs.
    valid_tokens gates which pairs are actually processed — required for CUDA graph
    compatibility since the grid size never changes across steps.
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
            # Skip tokens routed to non-local experts
            if lid >= 0 and lid < num_local_experts:
                # Atomically claim a position within this expert's aligned block
                pos = tl.atomic_add(counters_ptr + lid, 1)
                # Copy hidden state row
                for h in tl.range(0, hidden_dim, BLOCK_H):
                    o = h + tl.arange(0, BLOCK_H)
                    m = o < hidden_dim
                    tl.store(
                        out_hidden_ptr + pos * hidden_dim + o,
                        tl.load(hidden_ptr + tok * hidden_dim + o, mask=m),
                        mask=m,
                    )
                tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * topk + k))
                # Record source token index for unpermute
                tl.store(out_src_idx_ptr + pos, tok)


def permute_tokens(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    valid_tokens: torch.Tensor,
    alignment: int = 1,
) -> tuple:
    """Permute tokens into expert-grouped order.

    Computes token counts, aligned expert offsets, output sizing, and
    permutation in a single call.

    Args:
        hidden_states: [max_tokens, hidden_size] input. Only the first valid_tokens
            rows are valid; the rest are ignored.
        probs: [max_tokens, topk] routing probabilities.
        routing_map: [max_tokens, topk] expert assignments.
        local_expert_start: first global expert index on this rank.
        num_local_experts: number of experts on this rank.
        valid_tokens: scalar int32 CUDA tensor with the number of valid tokens this
            iteration. Fixed address; value updated each step before graph replay.
        alignment: per-expert token alignment (default 1).

    Returns:
        (permuted_hidden, permuted_probs, permutation_map, inclusive_offsets)
        - permuted_hidden: [output_size, hidden_size]
        - permuted_probs: [output_size]
        - permutation_map: [output_size] int32, maps each permuted row back to
          its original token index. Used by unpermute_tokens to scatter expert
          outputs back and by activation kernels to skip padding rows (-1).
        - inclusive_offsets: [num_local_experts] int32 cumulative offsets for grouped_mm
    """
    max_tokens, hidden_dim = hidden_states.shape
    topk = probs.shape[1]

    # Count how many (token, topk) pairs are routed to each local expert.
    # Non-local experts and rows beyond valid_tokens are ignored.
    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts, valid_tokens
    )

    # exclusive_expert_offsets[i] = start of expert i's block in the padded output.
    #   Used as the initial counter for atomic position assignment in the permute kernel.
    # inclusive_expert_offsets[i] = end of expert i's block (= start of expert i+1).
    #   Passed as `offs` to grouped_mm / scaled_grouped_mm to delimit expert boundaries.
    exclusive_expert_offsets, inclusive_expert_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=alignment
    )
    # Output sized at max to keep allocations fixed across steps (CUDA graph compatible).
    output_size = max_tokens * min(topk, num_local_experts) + alignment * num_local_experts

    permuted_hidden = torch.empty(
        output_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device
    )
    permuted_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    permutation_map = torch.empty(output_size, dtype=torch.int32, device=probs.device)
    # Only initialize [0, n_used) to -1; activation and unpermute kernels are gated
    # by the same inclusive_expert_offsets[-1] pointer so they never read beyond n_used.
    init_permutation_map(permutation_map, inclusive_expert_offsets[-1:])
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    max_pairs = max_tokens * topk
    NUM_BLOCKS = min(max_pairs, 512)
    _permute_tokens_kernel[(NUM_BLOCKS,)](
        hidden_states,
        probs,
        routing_map,
        permuted_hidden,
        permuted_probs,
        permutation_map,
        exclusive_expert_offsets,
        valid_tokens,
        hidden_dim,
        max_pairs,
        topk,
        local_expert_start,
        num_local_experts,
        BLOCK_H=BLOCK_H,
        NUM_BLOCKS=NUM_BLOCKS,
    )
    return permuted_hidden, permuted_probs, permutation_map, inclusive_expert_offsets


@triton.jit
def _zero_output_rows_kernel(
    output_ptr,  # [num_tokens, hidden_dim] fp32 buffer to partially zero
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of rows to zero
    hidden_dim,  # hidden dimension
    num_tokens,  # max token count (fixed for CG)
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # grid size (fixed for CG)
):
    """Zero rows [0, valid_tokens) of the fp32 output buffer.

    Grid: fixed NUM_BLOCKS CTAs, each iterating over multiple rows.
    valid_tokens gates which rows are zeroed — required for CUDA graph compatibility.
    """
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    if pid >= valid_tokens:
        return
    zero = tl.zeros([BLOCK_H], dtype=tl.float32)
    for row in tl.range(pid, num_tokens, NUM_BLOCKS):
        if row < valid_tokens:
            for h in tl.range(0, hidden_dim, BLOCK_H):
                o = h + tl.arange(0, BLOCK_H)
                m = o < hidden_dim
                tl.store(output_ptr + row * hidden_dim + o, zero, mask=m)


@triton.jit
def _unpermute_tokens_kernel(
    expert_out_ptr,  # [output_size, hidden_dim] expert outputs in permuted order
    probs_ptr,  # [output_size] fp32 routing probabilities (permuted)
    src_idx_ptr,  # [output_size] permutation_map: original token index, or -1 for padding
    output_ptr,  # [max_tokens, hidden_dim] fp32 output buffer (zeroed by caller)
    n_used_ptr,  # pointer to inclusive_expert_offsets[-1]: number of used rows this iteration
    hidden_dim,  # hidden dimension
    max_rows,  # output_size (fixed for CG)
    BLOCK_H: tl.constexpr,  # tile size for processing hidden_dim
    NUM_BLOCKS: tl.constexpr,  # grid size (fixed for CG)
):
    """Scatter weighted expert outputs back to original token positions.

    Grid: fixed NUM_BLOCKS CTAs, each iterating over multiple rows.
    Rows beyond n_used and alignment-padding rows (src_idx == -1) are skipped.
    Multiple topk selections for the same token are accumulated via atomic adds.
    All arithmetic is in fp32 to avoid precision loss.
    """
    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    if pid >= n_used:
        return
    for row in tl.range(pid, max_rows, NUM_BLOCKS):
        if row < n_used:
            source_idx = tl.load(src_idx_ptr + row)
            # Skip alignment-padding rows within the used range
            if source_idx >= 0:
                prob = tl.load(probs_ptr + row)  # fp32
                for h in tl.range(0, hidden_dim, BLOCK_H):
                    offsets = h + tl.arange(0, BLOCK_H)
                    m = offsets < hidden_dim
                    # Upcast bf16 expert output to fp32 before multiply + accumulate
                    v = tl.load(expert_out_ptr + row * hidden_dim + offsets, mask=m).to(tl.float32)
                    tl.atomic_add(output_ptr + source_idx * hidden_dim + offsets, v * prob, mask=m)


def unpermute_tokens(
    expert_output: torch.Tensor,
    permuted_probs: torch.Tensor,
    permutation_map: torch.Tensor,
    num_tokens: int,
    n_used: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Unpermute expert outputs back to original token order.

    Accumulates in fp32 to avoid precision loss from multiple topk atomic adds.
    Returns fp32 output.

    Args:
        expert_output: [output_size, hidden_dim] expert outputs in permuted order.
        permuted_probs: [output_size] fp32 routing probabilities.
        permutation_map: [output_size] int32, original token index or -1 for padding.
        num_tokens: max token count (output buffer height); always fixed for CG.
        n_used: scalar int32 CUDA tensor = inclusive_expert_offsets[-1]. Rows
            beyond this are skipped without reading permutation_map.
        valid_tokens: scalar int32 CUDA tensor = number of valid input tokens.
            Only rows [0, valid_tokens) are zeroed; all atomic_adds target
            source_idx < valid_tokens so rows beyond are never written.
        out: optional pre-allocated [num_tokens, hidden_dim] fp32 output buffer.
            Pass a symmetric memory tensor to scatter directly into it, avoiding
            a separate copy before RSV. If None, a local buffer is allocated.
    """
    assert (
        permuted_probs.dtype == torch.float32
    ), f"permuted_probs must be fp32, got {permuted_probs.dtype}"
    output_size, hidden_dim = expert_output.shape
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    if out is None:
        out = torch.empty(num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device)
    NUM_BLOCKS_ZERO = min(num_tokens, 512)
    _zero_output_rows_kernel[(NUM_BLOCKS_ZERO,)](
        out, valid_tokens, hidden_dim, num_tokens, BLOCK_H=BLOCK_H, NUM_BLOCKS=NUM_BLOCKS_ZERO
    )
    NUM_BLOCKS = min(output_size, 512)
    _unpermute_tokens_kernel[(NUM_BLOCKS,)](
        expert_output,
        permuted_probs,
        permutation_map,
        out,
        n_used,
        hidden_dim,
        output_size,
        BLOCK_H=BLOCK_H,
        NUM_BLOCKS=NUM_BLOCKS,
    )
    return out


@triton.jit
def _permute_quantize_mxfp8_kernel(
    hidden_ptr,
    probs_ptr,
    routing_map_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    out_probs_ptr,
    out_src_idx_ptr,
    counters_ptr,
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of valid tokens this iteration
    K,
    n_col_blocks,
    max_pairs,  # max_tokens * topk (fixed for CG)
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # grid size (fixed for CG)
):
    """Fused permute + MXFP8 quantize + swizzle in one kernel.

    Grid: fixed NUM_BLOCKS CTAs, each iterating over multiple (token, topk) pairs.
    valid_tokens gates which pairs are actually processed — required for CUDA graph
    compatibility since the grid size never changes across steps.
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

                # Load full row from source token
                offs = tl.arange(0, BLOCK_K)
                mask = offs < K
                x = tl.load(hidden_ptr + tok * K + offs, mask=mask, other=0.0).to(tl.float32)

                # Per-group-of-32 quantization
                x_grouped = tl.reshape(x, [BLOCK_GROUPS, 32])
                abs_grouped = tl.abs(x_grouped)
                max_vals = tl.max(abs_grouped, axis=1)

                dequant_scale = max_vals / 448.0
                dequant_exp = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
                dequant_rounded = dequant_exp.to(tl.float32, bitcast=True)
                quant_scale = tl.where(dequant_rounded == 0, 0.0, 1.0 / dequant_rounded)

                quantized = x_grouped * quant_scale[:, None]
                quantized_flat = tl.reshape(quantized, [BLOCK_K])
                out_fp8 = quantized_flat.to(tl.float8e4nv)

                # Store FP8 data at permuted position
                tl.store(out_fp8_ptr + pos * K + offs, out_fp8, mask=mask)

                # Store swizzled scales at permuted position
                scale_exp = (dequant_exp >> 23).to(tl.uint8)
                col_offs = tl.arange(0, BLOCK_GROUPS)
                col_mask = col_offs < REAL_GROUPS

                macro_row_block = pos // 128
                macro_col_block = col_offs // 4
                local_row = pos % 128
                local_col = col_offs % 4
                group = local_row // 32
                sub_row = local_row % 32
                tile_idx = macro_row_block * n_col_blocks + macro_col_block
                swizzled_offs = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

                tl.store(out_scale_ptr + swizzled_offs, scale_exp, mask=col_mask)

                # Store prob and source index
                tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * topk + k))
                tl.store(out_src_idx_ptr + pos, tok)


def permute_and_quantize_mxfp8(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    valid_tokens: torch.Tensor,
    alignment: int = 128,
) -> tuple:
    """Fused permute + MXFP8 quantize + swizzle.

    Self-contained API matching permute_tokens: computes token counts, aligned
    expert offsets, output sizing, permutation, and MXFP8 quantization in a
    single kernel launch.

    Args:
        hidden_states: [max_tokens, hidden_size] BF16 input. Only the first
            valid_tokens rows are valid; the rest are ignored.
        probs: [max_tokens, topk] routing probabilities.
        routing_map: [max_tokens, topk] expert assignments.
        local_expert_start: first global expert index on this rank.
        num_local_experts: number of experts on this rank.
        valid_tokens: scalar int32 CUDA tensor with the number of valid tokens this
            iteration. Fixed address; value updated each step before graph replay.
        alignment: per-expert token alignment (default 128, required for MXFP8 swizzle).

    Returns:
        (permuted_mxfp8, permuted_probs, permutation_map, inclusive_offsets)
        - permuted_mxfp8: MXFP8Tensor with .data [output_size, K] and .scale (swizzled)
        - permuted_probs: [output_size] routing probs
        - permutation_map: [output_size] int32, original token index or -1 for padding
        - inclusive_offsets: [num_local_experts] int32 cumulative offsets for scaled_grouped_mm
    """
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    max_tokens, K = hidden_states.shape
    topk = probs.shape[1]
    assert K % 32 == 0

    # Count how many (token, topk) pairs are routed to each local expert.
    # Rows beyond valid_tokens are ignored.
    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts, valid_tokens
    )

    # exclusive_expert_offsets[i] = start of expert i's block in the padded output.
    # inclusive_expert_offsets[i] = end of expert i's block (= start of expert i+1).
    exclusive_expert_offsets, inclusive_expert_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=alignment
    )
    # Output sized at max to keep allocations fixed across steps (CUDA graph compatible).
    output_size = max_tokens * min(topk, num_local_experts) + alignment * num_local_experts

    scale_cols = K // 32
    n_row_blocks = _ceil_div(output_size, 128)
    n_col_blocks = _ceil_div(scale_cols, 4)
    total_scale_bytes = n_row_blocks * n_col_blocks * 512

    out_fp8 = torch.empty(output_size, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=hidden_states.device)
    permuted_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    permutation_map = torch.empty(output_size, dtype=torch.int32, device=probs.device)
    init_permutation_map(permutation_map, inclusive_expert_offsets[-1:])

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // 32
    max_pairs = max_tokens * topk
    NUM_BLOCKS = min(max_pairs, 512)
    _permute_quantize_mxfp8_kernel[(NUM_BLOCKS,)](
        hidden_states,
        probs,
        routing_map,
        out_fp8,
        out_scale,
        permuted_probs,
        permutation_map,
        exclusive_expert_offsets,
        valid_tokens,
        K,
        n_col_blocks,
        max_pairs,
        topk,
        local_expert_start,
        num_local_experts,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
        NUM_BLOCKS=NUM_BLOCKS,
    )

    permuted_mxfp8 = MXFP8Tensor(
        data=out_fp8, scale=out_scale.view(torch.float8_e8m0fnu), backend="triton"
    )
    return permuted_mxfp8, permuted_probs, permutation_map, inclusive_expert_offsets
