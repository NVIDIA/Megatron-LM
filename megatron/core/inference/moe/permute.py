# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton kernels for token permutation and unpermutation in fused MoE.

Includes:
- Token counting per expert
- Expert offset computation (aligned prefix sums)
- Permute tokens into expert-grouped order
- Unpermute expert outputs back to original token order
"""

from unittest.mock import MagicMock

import torch
from packaging import version

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


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _count_local_tokens_kernel(
    routing_map_ptr,            # [num_tokens * topk] flattened expert assignments
    tokens_per_expert_ptr,      # [num_local_experts] output counters (zeroed by caller)
    total_pairs,                # num_tokens * topk — total (token, topk) pairs
    local_expert_start,         # first global expert index owned by this rank
    num_local_experts: tl.constexpr,  # number of experts on this rank
    BLOCK_SIZE: tl.constexpr,         # number of pairs processed per program
):
    """Count tokens routed to experts on this rank, ignoring tokens routed elsewhere.

    Each program processes BLOCK_SIZE (token, topk) pairs. Tokens assigned to
    experts outside [local_expert_start, local_expert_start + num_local_experts)
    are silently skipped.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_pairs
    expert_ids = tl.load(routing_map_ptr + offsets, mask=mask, other=-1)
    # Map global expert IDs to local indices; non-local experts become negative
    local_ids = expert_ids - local_expert_start
    is_local = (local_ids >= 0) & (local_ids < num_local_experts) & mask
    tl.atomic_add(tokens_per_expert_ptr + local_ids, 1, mask=is_local)


def compute_local_tokens_per_expert(
    routing_map: torch.Tensor, local_expert_start: int, num_local_experts: int,
) -> torch.Tensor:
    """Count tokens routed to each local expert."""
    total_pairs = routing_map.numel()
    tokens_per_expert = torch.zeros(
        num_local_experts, dtype=torch.int32, device=routing_map.device,
    )
    BLOCK = 256
    _count_local_tokens_kernel[(_ceil_div(total_pairs, BLOCK),)](
        routing_map, tokens_per_expert, total_pairs,
        local_expert_start, num_local_experts, BLOCK_SIZE=BLOCK,
    )
    return tokens_per_expert


@triton.jit
def _prefix_sum_kernel(
    tokens_per_expert_ptr,      # [num_local_experts] raw token counts
    exclusive_offsets_ptr,      # [num_local_experts] output: exclusive prefix sum of aligned counts
    inclusive_offsets_ptr,       # [num_local_experts] output: inclusive prefix sum of aligned counts
    num_local_experts,          # number of experts on this rank
    alignment: tl.constexpr,    # per-expert alignment (counts rounded up to this multiple)
    BLOCK_SIZE: tl.constexpr,   # next_power_of_2(num_local_experts) for tl.cumsum
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


def compute_expert_offsets(
    tokens_per_expert: torch.Tensor, alignment: int = 1,
) -> tuple:
    """Compute exclusive and inclusive prefix sums of aligned token counts."""
    n = tokens_per_expert.shape[0]
    exclusive_cumsum = torch.empty_like(tokens_per_expert)
    inclusive_cumsum = torch.empty_like(tokens_per_expert)
    _prefix_sum_kernel[(1,)](
        tokens_per_expert, exclusive_cumsum, inclusive_cumsum, n, alignment,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return exclusive_cumsum, inclusive_cumsum


@triton.jit
def _permute_tokens_kernel(
    hidden_ptr,         # [num_tokens, hidden_dim] input hidden states
    probs_ptr,          # [num_tokens, topk] routing probabilities
    routing_map_ptr,    # [num_tokens, topk] expert assignments (global IDs)
    out_hidden_ptr,     # [output_size, hidden_dim] output: permuted hidden states
    out_probs_ptr,      # [output_size] output: permuted probabilities
    out_src_idx_ptr,    # [output_size] output: permutation_map (original token index, -1 for padding)
    counters_ptr,       # [num_local_experts] exclusive offsets, atomically incremented to assign positions
    num_tokens,         # number of input tokens
    hidden_dim,         # hidden dimension
    topk: tl.constexpr,              # number of expert choices per token
    local_expert_start,               # first global expert index on this rank
    num_local_experts: tl.constexpr,  # number of experts on this rank
    BLOCK_H: tl.constexpr,           # tile size for copying hidden_dim
):
    """Permute tokens into expert-grouped order.

    Grid: one program per (token, topk) pair. Each program looks up the assigned
    expert, skips non-local experts, then atomically claims a position within
    that expert's block and copies the hidden state + prob + source index.
    """
    # Each program handles one (token, topk) pair
    pair = tl.program_id(0)
    tok = pair // topk
    k = pair % topk
    if tok >= num_tokens:
        return
    eid = tl.load(routing_map_ptr + tok * topk + k)
    lid = eid - local_expert_start
    # Skip tokens routed to non-local experts
    if lid < 0 or lid >= num_local_experts:
        return
    # Atomically claim a position within this expert's aligned block
    pos = tl.atomic_add(counters_ptr + lid, 1)
    # Copy hidden state row
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        tl.store(out_hidden_ptr + pos * hidden_dim + o,
                 tl.load(hidden_ptr + tok * hidden_dim + o, mask=m), mask=m)
    tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * topk + k))
    # Record source token index for unpermute
    tl.store(out_src_idx_ptr + pos, tok)


def permute_tokens(
    hidden_states: torch.Tensor, probs: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int, num_local_experts: int,
    alignment: int = 1,
) -> tuple:
    """Permute tokens into expert-grouped order.

    Computes token counts, aligned expert offsets, output sizing, and
    permutation in a single call.

    Args:
        hidden_states: [num_tokens, hidden_size] input.
        probs: [num_tokens, topk] routing probabilities.
        routing_map: [num_tokens, topk] expert assignments.
        local_expert_start: first global expert index on this rank.
        num_local_experts: number of experts on this rank.
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
    num_tokens, hidden_dim = hidden_states.shape
    topk = probs.shape[1]

    # Count how many (token, topk) pairs are routed to each local expert.
    # Non-local experts are ignored. Result is [num_local_experts] int32.
    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts,
    )

    # exclusive_expert_offsets[i] = start of expert i's block in the padded output.
    #   Used as the initial counter for atomic position assignment in the permute kernel.
    # inclusive_expert_offsets[i] = end of expert i's block (= start of expert i+1).
    #   Passed as `offs` to grouped_mm / scaled_grouped_mm to delimit expert boundaries.
    exclusive_expert_offsets, inclusive_expert_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=alignment,
    )
    output_size = (
        num_tokens * min(topk, num_local_experts)
        + alignment * num_local_experts
    )

    permuted_hidden = torch.empty(output_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    permuted_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    permutation_map = torch.full((output_size,), -1, dtype=torch.int32, device=probs.device)
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    _permute_tokens_kernel[(num_tokens * topk,)](
        hidden_states, probs, routing_map,
        permuted_hidden, permuted_probs, permutation_map,
        exclusive_expert_offsets, num_tokens, hidden_dim, topk,
        local_expert_start, num_local_experts, BLOCK_H=BLOCK_H,
    )
    return permuted_hidden, permuted_probs, permutation_map, inclusive_expert_offsets


@triton.jit
def _unpermute_tokens_kernel(
    expert_out_ptr,     # [output_size, hidden_dim] expert outputs in permuted order
    probs_ptr,          # [output_size] fp32 routing probabilities (permuted)
    src_idx_ptr,        # [output_size] permutation_map: original token index, or -1 for padding
    output_ptr,         # [num_tokens, hidden_dim] fp32 output buffer (zeroed by caller)
    hidden_dim,         # hidden dimension
    BLOCK_H: tl.constexpr,  # tile size for processing hidden_dim
):
    """Scatter weighted expert outputs back to original token positions.

    Grid: one program per row of expert_out. Padding rows (src_idx == -1) are
    skipped. Multiple topk selections for the same token are accumulated via
    atomic adds. All arithmetic is in fp32 to avoid precision loss.
    """
    row = tl.program_id(0)
    source_idx = tl.load(src_idx_ptr + row)
    # Skip padding rows
    if source_idx < 0:
        return
    prob = tl.load(probs_ptr + row)  # fp32
    for h in tl.range(0, hidden_dim, BLOCK_H):
        offsets = h + tl.arange(0, BLOCK_H)
        m = offsets < hidden_dim
        # Upcast bf16 expert output to fp32 before multiply + accumulate
        v = tl.load(expert_out_ptr + row * hidden_dim + offsets, mask=m).to(tl.float32)
        tl.atomic_add(output_ptr + source_idx * hidden_dim + offsets, v * prob, mask=m)

def unpermute_tokens(
    expert_output: torch.Tensor, permuted_probs: torch.Tensor,
    permutation_map: torch.Tensor, num_tokens: int,
) -> torch.Tensor:
    """Unpermute expert outputs back to original token order.

    Accumulates in fp32 to avoid precision loss from multiple topk atomic adds.
    Returns fp32 output.
    """
    assert permuted_probs.dtype == torch.float32, (
        f"permuted_probs must be fp32, got {permuted_probs.dtype}"
    )
    output_size, hidden_dim = expert_output.shape
    output = torch.zeros(num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device)
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    _unpermute_tokens_kernel[(output_size,)](
        expert_output, permuted_probs, permutation_map,
        output, hidden_dim, BLOCK_H=BLOCK_H,
    )
    return output


@triton.jit
def _permute_quantize_mxfp8_kernel(
    hidden_ptr, probs_ptr, routing_map_ptr,
    out_fp8_ptr, out_scale_ptr, out_probs_ptr, out_src_idx_ptr,
    counters_ptr,
    num_tokens, K,
    n_col_blocks,
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Fused permute + MXFP8 quantize + swizzle in one kernel.

    Grid: (num_tokens * topk,) — one program per (token, k) pair.
    Reads BF16 from source token, quantizes to FP8 e4m3, writes FP8 data +
    swizzled e8m0 scales to the permuted write position.
    """
    pair = tl.program_id(0)
    tok = pair // topk
    k = pair % topk
    if tok >= num_tokens:
        return
    eid = tl.load(routing_map_ptr + tok * topk + k)
    lid = eid - local_expert_start
    if lid < 0 or lid >= num_local_experts:
        return

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
    alignment: int = 128,
) -> tuple:
    """Fused permute + MXFP8 quantize + swizzle.

    Self-contained API matching permute_tokens: computes token counts, aligned
    expert offsets, output sizing, permutation, and MXFP8 quantization in a
    single kernel launch.

    Args:
        hidden_states: [num_tokens, hidden_size] BF16 input.
        probs: [num_tokens, topk] routing probabilities.
        routing_map: [num_tokens, topk] expert assignments.
        local_expert_start: first global expert index on this rank.
        num_local_experts: number of experts on this rank.
        alignment: per-expert token alignment (default 128, required for MXFP8 swizzle).

    Returns:
        (permuted_mxfp8, permuted_probs, permutation_map, inclusive_offsets)
        - permuted_mxfp8: MXFP8Tensor with .data [output_size, K] and .scale (swizzled)
        - permuted_probs: [output_size] routing probs
        - permutation_map: [output_size] int32, original token index or -1 for padding
        - inclusive_offsets: [num_local_experts] int32 cumulative offsets for scaled_grouped_mm
    """
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    num_tokens, K = hidden_states.shape
    topk = probs.shape[1]
    assert K % 32 == 0

    # Count how many (token, topk) pairs are routed to each local expert.
    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts,
    )

    # exclusive_expert_offsets[i] = start of expert i's block in the padded output.
    # inclusive_expert_offsets[i] = end of expert i's block (= start of expert i+1).
    exclusive_expert_offsets, inclusive_expert_offsets = compute_expert_offsets(
        tokens_per_expert, alignment=alignment,
    )
    output_size = (
        num_tokens * min(topk, num_local_experts)
        + alignment * num_local_experts
    )

    scale_cols = K // 32
    n_row_blocks = _ceil_div(output_size, 128)
    n_col_blocks = _ceil_div(scale_cols, 4)
    total_scale_bytes = n_row_blocks * n_col_blocks * 512

    out_fp8 = torch.empty(output_size, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=hidden_states.device)
    permuted_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    permutation_map = torch.full((output_size,), -1, dtype=torch.int32, device=probs.device)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // 32

    _permute_quantize_mxfp8_kernel[(num_tokens * topk,)](
        hidden_states, probs, routing_map,
        out_fp8, out_scale, permuted_probs, permutation_map,
        exclusive_expert_offsets,
        num_tokens, K, n_col_blocks,
        topk, local_expert_start, num_local_experts,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )

    permuted_mxfp8 = MXFP8Tensor(
        data=out_fp8,
        scale=out_scale.view(torch.float8_e8m0fnu),
        backend="triton",
    )
    return permuted_mxfp8, permuted_probs, permutation_map, inclusive_expert_offsets
