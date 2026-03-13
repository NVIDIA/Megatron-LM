# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pad / unpad utilities for already-permuted expert tokens.

When the token dispatcher has already permuted tokens into expert-grouped
order, these functions insert/remove alignment padding so that each expert's
token block satisfies the alignment requirements of grouped_mm /
scaled_grouped_mm.
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

from megatron.core.inference.moe.permute import compute_expert_offsets


# =========================================================================== #
# Pad kernel
# =========================================================================== #
@triton.jit
def _pad_tokens_kernel(
    src_ptr, dst_ptr, perm_map_ptr,
    tpe_ptr,  # tokens_per_expert [num_experts]
    hidden_dim, num_experts: tl.constexpr,
    alignment: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Copy one input row into the padded output buffer.

    Computes unpadded and padded cumulative offsets inline from
    tokens_per_expert, avoiding a separate cumsum kernel launch.
    """
    row = tl.program_id(0)

    # Walk tokens_per_expert to find which expert this row belongs to
    # and compute both unpadded and padded start offsets on the fly.
    unpadded_start = tl.zeros([], dtype=tl.int32)
    padded_start = tl.zeros([], dtype=tl.int32)
    expert_id = -1
    for e in tl.static_range(0, num_experts):
        count = tl.load(tpe_ptr + e).to(tl.int32)
        if expert_id < 0 and row < unpadded_start + count:
            expert_id = e
        if expert_id < 0:
            unpadded_start += count
            aligned = tl.where(count > 0, ((count + alignment - 1) // alignment) * alignment, tl.zeros([], dtype=tl.int32))
            padded_start += aligned

    if expert_id < 0:
        return

    local_idx = row - unpadded_start
    dst_row = padded_start + local_idx

    # Write permutation_map: padded row → original unpadded row
    tl.store(perm_map_ptr + dst_row, row)

    # Copy hidden state
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        tl.store(dst_ptr + dst_row * hidden_dim + o,
                 tl.load(src_ptr + row * hidden_dim + o, mask=m), mask=m)


def pad_to_alignment(
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    alignment: int,
) -> tuple:
    """Pad already-permuted tokens so each expert's block is aligned.

    Args:
        hidden_states: [total_tokens, hidden_size] already permuted by dispatcher.
        tokens_per_expert: [num_local_experts] int32 token counts.
        alignment: per-expert alignment.

    Returns:
        (padded_hidden, permutation_map, inclusive_offsets)
        - padded_hidden: [padded_total, hidden_size]
        - permutation_map: [padded_total] int32, original row index or -1 for padding.
        - inclusive_offsets: [num_local_experts] int32 cumulative aligned offsets for grouped_mm.
    """
    num_experts = tokens_per_expert.shape[0]
    total_tokens = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]

    # We still need padded_inc for the return value (used as offs by grouped_mm)
    _, padded_inc = compute_expert_offsets(tokens_per_expert, alignment=alignment)
    padded_total = int(padded_inc[-1].item())

    padded_hidden = torch.zeros(
        padded_total, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device,
    )
    permutation_map = torch.full(
        (padded_total,), -1, dtype=torch.int32, device=hidden_states.device,
    )

    if total_tokens > 0:
        BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
        _pad_tokens_kernel[(total_tokens,)](
            hidden_states, padded_hidden, permutation_map,
            tokens_per_expert,
            hidden_dim, num_experts, alignment, BLOCK_H=BLOCK_H,
        )

    return padded_hidden, permutation_map, padded_inc


# =========================================================================== #
# Unpad kernel
# =========================================================================== #
@triton.jit
def _unpad_tokens_kernel(
    src_ptr, dst_ptr, perm_map_ptr, probs_ptr,
    hidden_dim, has_probs: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Copy one real (non-padding) row from padded to unpadded layout.

    Optionally multiplies each row by its routing probability.
    """
    row = tl.program_id(0)
    dst_row = tl.load(perm_map_ptr + row)
    if dst_row < 0:
        return
    if has_probs:
        prob = tl.load(probs_ptr + dst_row)
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        v = tl.load(src_ptr + row * hidden_dim + o, mask=m)
        if has_probs:
            v = v * prob
        tl.store(dst_ptr + dst_row * hidden_dim + o, v, mask=m)


def unpad_from_alignment(
    padded_output: torch.Tensor,
    permutation_map: torch.Tensor,
    original_size: int,
    probs: torch.Tensor = None,
) -> torch.Tensor:
    """Remove alignment padding, scattering results back to original positions.

    Args:
        padded_output: [padded_total, hidden_size] output from expert computation.
        permutation_map: [padded_total] int32, original row index or -1 for padding.
        original_size: number of rows in the unpadded output.
        probs: optional [original_size] routing probabilities to multiply during unpad.

    Returns:
        [original_size, hidden_size] unpadded output.
    """
    hidden_dim = padded_output.shape[1]
    output = torch.zeros(
        original_size, hidden_dim, dtype=padded_output.dtype, device=padded_output.device,
    )
    has_probs = probs is not None
    if padded_output.shape[0] > 0:
        BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
        _unpad_tokens_kernel[(padded_output.shape[0],)](
            padded_output, output, permutation_map,
            probs if has_probs else padded_output,  # dummy pointer when no probs
            hidden_dim, has_probs, BLOCK_H=BLOCK_H,
        )
    return output
