# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

# Persistent grid: each CTA strides over the live rows, so idle CTAs cost nothing at decode loads.
_COMBINE_CTAS = 4096


@triton.jit
def _combine_valid_rows_kernel(
    src_ptr,  # [rows, hidden] bf16 kernel output; only tokens with a local expert were written
    routing_ptr,  # [rows, TOPK] int32 global expert ids
    out_ptr,  # [rows, hidden] fp32 output; rows [0, valid_tokens) are fully written
    valid_tokens_ptr,  # scalar int32 CUDA tensor
    hidden,
    local_expert_offset,
    num_local_experts,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # persistent grid size
):
    """Copy each valid token's row to fp32, or zero it when none of its experts is local."""
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    k = tl.arange(0, TOPK_PAD)
    for row in tl.range(pid, valid_tokens, NUM_BLOCKS):
        row64 = tl.cast(row, tl.int64)
        experts = tl.load(routing_ptr + row64 * TOPK + k, mask=k < TOPK, other=-1)
        experts = experts - local_expert_offset
        is_local = (experts >= 0) & (experts < num_local_experts)
        has_local = tl.sum(is_local.to(tl.int32), axis=0) > 0
        for h in tl.range(0, hidden, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            col_mask = offs < hidden
            values = tl.load(src_ptr + row64 * hidden + offs, mask=col_mask & has_local, other=0.0)
            tl.store(out_ptr + row64 * hidden + offs, values.to(tl.float32), mask=col_mask)


def combine_valid_rows(
    expert_output: torch.Tensor,
    routing_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: torch.Tensor,
    *,
    local_expert_offset: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Write rows [0, valid_tokens) of the bf16 `expert_output` into the fp32 `out` buffer.

    A token with no expert in [local_expert_offset, local_expert_offset + num_local_experts)
    gets a zero row, as the kernel's zero-filling finalize would have produced.
    Rows at or beyond valid_tokens are not written. Returns `out`.
    """
    if not HAVE_TRITON:
        raise RuntimeError("FlashInfer bf16 MoE on the NVLS dispatcher requires triton.")
    rows, hidden = expert_output.shape
    if routing_ids.ndim != 2 or routing_ids.shape[0] != rows or routing_ids.dtype != torch.int32:
        raise ValueError(
            f"routing_ids must be int32 [{rows}, top_k]; got {routing_ids.dtype} "
            f"{tuple(routing_ids.shape)}"
        )
    if out.ndim != 2 or out.shape[0] < rows or out.shape[1] != hidden or out.dtype != torch.float32:
        raise ValueError(
            f"out must be fp32 with at least {rows} rows of {hidden}; got {out.dtype} "
            f"{tuple(out.shape)}"
        )
    for name, tensor in (
        ("expert_output", expert_output),
        ("routing_ids", routing_ids),
        ("out", out),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    topk = routing_ids.shape[1]
    num_blocks = min(rows, _COMBINE_CTAS)
    _combine_valid_rows_kernel[(num_blocks,)](
        expert_output,
        routing_ids,
        out,
        valid_tokens,
        hidden,
        local_expert_offset,
        num_local_experts,
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        BLOCK_H=min(triton.next_power_of_2(hidden), 1024),
        NUM_BLOCKS=num_blocks,
    )
    return out
