# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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


@triton.jit
def _unpermute_add_kernel(
    permuted_ptr,
    map_ptr,
    num_dense_ptr,
    add_ptr,
    out_ptr,
    num_tokens,
    hidden_size,
    num_experts,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_ADD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program handles BLOCK_T tokens across the whole hidden dimension.
    pid = tl.program_id(0)
    num_dense = tl.load(num_dense_ptr)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    rows64 = rows.to(tl.int64)
    row_mask = rows < num_tokens
    cols = tl.arange(0, E_POW2)
    map_mask = row_mask[:, None] & (cols[None, :] < num_experts)
    src = tl.load(map_ptr + rows64[:, None] * num_experts + cols[None, :], mask=map_mask, other=-1)
    active = (src >= 0) & (rows[:, None] < num_dense) & map_mask
    # The k-th gather of a token reads the row of its k-th routed expert, in expert order.
    rank = tl.cumsum(active.to(tl.int32), axis=1) - 1
    for h0 in range(0, hidden_size, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        h_mask = offs < hidden_size
        out_mask = row_mask[:, None] & h_mask[None, :]
        if HAS_ADD:
            acc = tl.load(
                add_ptr + rows64[:, None] * hidden_size + offs[None, :], mask=out_mask, other=0.0
            ).to(tl.float32)
        else:
            acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            sel = active & (rank == k)
            src_k = tl.sum(tl.where(sel, src, 0), axis=1).to(tl.int64)
            has_k = tl.sum(sel.to(tl.int32), axis=1) > 0
            v = tl.load(
                permuted_ptr + src_k[:, None] * hidden_size + offs[None, :],
                mask=has_k[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += v.to(tl.float32)
        tl.store(
            out_ptr + rows64[:, None] * hidden_size + offs[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=out_mask,
        )


def fused_unpermute_add(
    permuted: torch.Tensor,
    dense_to_expert_map: torch.Tensor,
    num_dense_tokens: torch.Tensor,
    topk: int,
    add: Optional[torch.Tensor] = None,
    num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Sum the expert outputs of every token and optionally add a per-token row, in one pass.

    out[t] = add[t] + sum_e permuted[dense_to_expert_map[t, e]] over the entries >= 0, accumulated
    in fp32 and rounded once.

    Args:
        permuted (torch.Tensor): [num_permuted_tokens, hidden] expert outputs.
        dense_to_expert_map (torch.Tensor): [>= num_tokens, num_experts] int32 row of each
            (token, expert) pair in ``permuted``, -1 when the token is not routed to the expert.
        num_dense_tokens (torch.Tensor): int32 scalar on the device, the number of tokens that have
            routed rows; the rest get ``add`` only.
        topk (int): Maximum number of experts a token is routed to.
        add (torch.Tensor, optional): [num_tokens, hidden] rows added to the sum.
        num_tokens (int, optional): Number of output tokens. Defaults to ``add.shape[0]``.

    Returns:
        torch.Tensor: [num_tokens, hidden] in the dtype of ``add`` (or ``permuted``).
    """
    assert HAVE_TRITON, "fused_unpermute_add requires Triton"
    if num_tokens is None:
        assert add is not None, "num_tokens is required without add"
        num_tokens = add.shape[0]
    hidden_size = permuted.shape[-1]
    num_experts = dense_to_expert_map.shape[1]
    assert permuted.is_contiguous() and dense_to_expert_map.is_contiguous()
    assert dense_to_expert_map.dtype == torch.int32
    assert dense_to_expert_map.shape[0] >= num_tokens
    if add is not None:
        assert add.shape == (num_tokens, hidden_size) and add.is_contiguous()
    out = torch.empty(
        num_tokens,
        hidden_size,
        dtype=add.dtype if add is not None else permuted.dtype,
        device=permuted.device,
    )
    if num_tokens == 0:
        return out
    block_t = 16
    _unpermute_add_kernel[(triton.cdiv(num_tokens, block_t),)](
        permuted,
        dense_to_expert_map,
        num_dense_tokens,
        add if add is not None else permuted,
        out,
        num_tokens,
        hidden_size,
        num_experts,
        E_POW2=triton.next_power_of_2(num_experts),
        TOPK=min(topk, num_experts),
        HAS_ADD=add is not None,
        BLOCK_T=block_t,
        BLOCK_H=256,
        num_warps=4,
    )
    return out
