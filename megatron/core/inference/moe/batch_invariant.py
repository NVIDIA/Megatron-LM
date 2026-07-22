# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant inference MoE helpers."""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    grouped_gemm_batch_invariant,
    grouped_gemm_batch_invariant_alignment,
    is_batch_invariant_mode_enabled,
)
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


def enabled() -> bool:
    """Return whether global batch-invariant mode is active."""
    return is_batch_invariant_mode_enabled()


def grouped_mm(x_bf16: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Batch-invariant BF16 grouped GEMM used by inference fused MoE."""
    return grouped_gemm_batch_invariant(
        x_bf16,
        weight,
        offs=offs.to(torch.int32),
        m_total=x_bf16.shape[0],
    )


def grouped_mm_alignment() -> int:
    """Per-expert row alignment required by the batch-invariant grouped GEMM."""
    return grouped_gemm_batch_invariant_alignment()


@triton.jit
def _unpermute_tokens_in_expert_order_kernel(
    expert_out_ptr,  # [output_size, hidden_dim] bf16 expert outputs
    probs_ptr,  # [output_size] fp32 routing probabilities
    inverse_map_ptr,  # [num_tokens, num_local_experts] permuted row or -1
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of valid tokens
    output_ptr,  # [num_tokens, hidden_dim] fp32 output buffer
    hidden_dim,
    num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Token-local batch-invariant unpermute.

    Each program owns one output token and one hidden tile. Contributions are
    accumulated in fp32 by increasing local expert id, avoiding atomic-add order.
    """
    tok = tl.program_id(0)
    block_h = tl.program_id(1)
    valid_tokens = tl.load(valid_tokens_ptr)
    offsets = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offsets < hidden_dim

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    if tok < valid_tokens:
        for lid in tl.range(0, num_local_experts):
            pos = tl.load(inverse_map_ptr + tok * num_local_experts + lid)
            if pos >= 0:
                prob = tl.load(probs_ptr + pos)
                vals = tl.load(expert_out_ptr + pos * hidden_dim + offsets, mask=mask_h).to(
                    tl.float32
                )
                acc += vals * prob
        tl.store(output_ptr + tok * hidden_dim + offsets, acc, mask=mask_h)


def unpermute_tokens_in_expert_order(
    expert_output: torch.Tensor,
    permuted_probs: torch.Tensor,
    inverse_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reduce local expert contributions token-by-token in fixed expert order."""
    _, hidden_dim = expert_output.shape
    num_tokens, num_local_experts = inverse_map.shape
    if out is None:
        out = torch.empty(
            num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device
        )

    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_H))
    _unpermute_tokens_in_expert_order_kernel[grid](
        expert_output,
        permuted_probs,
        inverse_map,
        valid_tokens,
        out,
        hidden_dim,
        num_local_experts,
        BLOCK_H=BLOCK_H,
    )
    return out
