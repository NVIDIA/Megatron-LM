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

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


def _ceil_div(a, b):
    return (a + b - 1) // b


# =========================================================================== #
# Token count kernel
# =========================================================================== #
@triton.jit
def _count_local_tokens_kernel(
    routing_map_ptr, tokens_per_expert_ptr, total_pairs,
    local_expert_start, num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count tokens assigned to each local expert."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_pairs
    expert_ids = tl.load(routing_map_ptr + offsets, mask=mask, other=-1)
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


# =========================================================================== #
# Expert offset computation
# =========================================================================== #
@triton.jit
def _prefix_sum_kernel(
    tokens_per_expert_ptr, exclusive_offsets_ptr, inclusive_offsets_ptr,
    num_local_experts, alignment: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Exclusive and inclusive prefix sums of aligned token counts."""
    r = tl.arange(0, BLOCK_SIZE)
    mask = r < num_local_experts
    h = tl.load(tokens_per_expert_ptr + r, mask=mask, other=0)
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
    exc = torch.empty_like(tokens_per_expert)
    inc = torch.empty_like(tokens_per_expert)
    _prefix_sum_kernel[(1,)](
        tokens_per_expert, exc, inc, n, alignment,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return exc, inc


# =========================================================================== #
# Permute / unpermute kernels
# =========================================================================== #
@triton.jit
def _permute_tokens_kernel(
    hidden_ptr, probs_ptr, routing_map_ptr,
    out_hidden_ptr, out_probs_ptr, out_src_idx_ptr, counters_ptr,
    num_tokens, hidden_dim, topk: tl.constexpr,
    local_expert_start, num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Permute tokens into expert-grouped order."""
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
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        tl.store(out_hidden_ptr + pos * hidden_dim + o,
                 tl.load(hidden_ptr + tok * hidden_dim + o, mask=m), mask=m)
    tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * topk + k))
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

    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts,
    )
    padded_exc, padded_inc = compute_expert_offsets(
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
        padded_exc, num_tokens, hidden_dim, topk,
        local_expert_start, num_local_experts, BLOCK_H=BLOCK_H,
    )
    return permuted_hidden, permuted_probs, permutation_map, padded_inc


@triton.jit
def _unpermute_tokens_kernel(
    expert_out_ptr, probs_ptr, src_idx_ptr, output_ptr,
    hidden_dim, BLOCK_H: tl.constexpr,
):
    """Accumulate weighted expert outputs back to original token positions."""
    row = tl.program_id(0)
    tok = tl.load(src_idx_ptr + row)
    if tok < 0:
        return
    prob = tl.load(probs_ptr + row)
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        v = tl.load(expert_out_ptr + row * hidden_dim + o, mask=m)
        tl.atomic_add(output_ptr + tok * hidden_dim + o, v * prob, mask=m)


def unpermute_tokens(
    expert_output: torch.Tensor, permuted_probs: torch.Tensor,
    permutation_map: torch.Tensor, num_tokens: int,
) -> torch.Tensor:
    """Unpermute expert outputs back to original token order."""
    output_size, hidden_dim = expert_output.shape
    output = torch.zeros(num_tokens, hidden_dim, dtype=expert_output.dtype, device=expert_output.device)
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    _unpermute_tokens_kernel[(output_size,)](
        expert_output, permuted_probs, permutation_map,
        output, hidden_dim, BLOCK_H=BLOCK_H,
    )
    return output
