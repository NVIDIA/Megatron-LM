# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant MoE permutation helpers."""

from typing import Optional

import torch

from megatron.core.process_groups_config import resolve_process_groups
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    get_batch_invariant_collective,
)

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


def build_inverse_permutation_map(
    routing_map: torch.Tensor,
    flat_sorted: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_out_tokens: int,
) -> torch.Tensor:
    """Build token/top-k -> permuted-row and expert-id map for batch-invariant unpermute.

    The regular permutation map is row -> token. Batch-invariant unpermute needs
    the inverse ownership model so each output token can read its routed rows and
    add them in a fixed order.
    """
    num_tokens = routing_map.size(0)
    assert isinstance(
        num_out_tokens, int
    ), "batch-invariant graph unpermute requires static num_out_tokens"
    assert (
        num_out_tokens % num_tokens == 0
    ), "batch-invariant graph unpermute expects fixed top-k per token"

    topk = num_out_tokens // num_tokens
    row_ids = torch.arange(num_out_tokens, device=routing_map.device, dtype=torch.long)
    expert_ids = torch.div(flat_sorted, num_tokens, rounding_mode='floor').to(torch.long)
    token_ids = sorted_indices.to(torch.long)

    slots_by_token_expert = routing_map.bool().to(torch.long).cumsum(dim=1) - 1
    row_slots = slots_by_token_expert[token_ids, expert_ids]
    linear_slots = token_ids * topk + row_slots

    inverse_rows = torch.full((num_tokens, topk), -1, device=routing_map.device, dtype=torch.long)
    inverse_experts = torch.full(
        (num_tokens, topk), -1, device=routing_map.device, dtype=torch.long
    )
    inverse_rows.view(-1).scatter_(0, linear_slots, row_ids)
    inverse_experts.view(-1).scatter_(0, linear_slots, expert_ids)
    return torch.stack((inverse_rows, inverse_experts), dim=0)


@triton.jit
def _rank_tree_unpermute_kernel(
    rows_ptr,
    inverse_rows_ptr,
    inverse_experts_ptr,
    out_ptr,
    hidden,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    TOPK: tl.constexpr,
    CROSS_RANK_FP64: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per (token, hidden block): the rank-ordered combine tree of ``unpermute``."""
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    in_hidden = offsets < hidden
    if CROSS_RANK_FP64:
        acc = tl.zeros([BLOCK_H], dtype=tl.float64)
    else:
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for ep_rank in tl.static_range(EP_SIZE):
        rank_partial = tl.zeros([BLOCK_H], dtype=tl.float64)
        for k in tl.static_range(TOPK):
            row = tl.load(inverse_rows_ptr + token * TOPK + k)
            expert = tl.load(inverse_experts_ptr + token * TOPK + k)
            valid = (
                (row >= 0)
                & (expert >= ep_rank * EXPERTS_PER_RANK)
                & (expert < (ep_rank + 1) * EXPERTS_PER_RANK)
            )
            safe_row = tl.where(row >= 0, row, 0).to(tl.int64)
            values = tl.load(
                rows_ptr + safe_row * hidden + offsets, mask=in_hidden & valid, other=0.0
            )
            rank_partial += values.to(tl.float64)
        if CROSS_RANK_FP64:
            acc += rank_partial.to(tl.float32).to(tl.float64)
        else:
            acc += rank_partial.to(tl.float32)
    out = acc.to(tl.float32).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + token.to(tl.int64) * hidden + offsets, out, mask=in_hidden)


@triton.jit
def _rank_tree_unpermute_bwd_kernel(
    grad_out_ptr,
    grad_out_stride_token,
    grad_out_stride_hidden,
    inverse_rows_ptr,
    valid_ptr,
    grad_rows_ptr,
    hidden,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per (token, hidden block): scatter the token's gradient to its rows.

    Each permuted row belongs to one (token, slot), so the stores never collide. The grid
    depends only on tensor shapes, which keeps the backward CUDA-graph capturable.
    """
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    in_hidden = offsets < hidden
    grad = tl.load(
        grad_out_ptr
        + token.to(tl.int64) * grad_out_stride_token
        + offsets * grad_out_stride_hidden,
        mask=in_hidden,
        other=0.0,
    )
    # The loop accumulates into zeros, so -0.0 arrives as +0.0. Written as a select:
    # ``grad + 0.0`` may be folded away.
    grad = tl.where(grad == 0, 0.0, grad).to(grad_rows_ptr.dtype.element_ty)
    for k in tl.static_range(TOPK):
        row = tl.load(inverse_rows_ptr + token * TOPK + k)
        valid = tl.load(valid_ptr + token * TOPK + k) != 0
        safe_row = tl.where(valid, row, 0).to(tl.int64)
        tl.store(grad_rows_ptr + safe_row * hidden + offsets, grad, mask=in_hidden & valid)


class _RankTreeUnpermute(torch.autograd.Function):
    """Fused ``unpermute`` for ``probs=None``: same arithmetic tree as the loop, one launch."""

    @staticmethod
    def forward(ctx, permuted_tokens, inverse_rows, inverse_experts, ep_size, num_experts):
        # pylint: disable=missing-function-docstring
        num_tokens, topk = inverse_rows.shape
        hidden = permuted_tokens.shape[-1]
        rows = permuted_tokens.contiguous()
        inverse_rows = inverse_rows.contiguous()
        inverse_experts = inverse_experts.contiguous()
        out = torch.empty(num_tokens, hidden, dtype=rows.dtype, device=rows.device)
        if num_tokens > 0:
            block_h = min(triton.next_power_of_2(hidden), 512)
            _rank_tree_unpermute_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
                rows,
                inverse_rows,
                inverse_experts,
                out,
                hidden,
                EP_SIZE=ep_size,
                EXPERTS_PER_RANK=num_experts // ep_size,
                TOPK=topk,
                CROSS_RANK_FP64=get_batch_invariant_collective() == "multimem",
                BLOCK_H=block_h,
                num_warps=4,
            )
        valid = (inverse_rows >= 0) & (inverse_experts >= 0) & (inverse_experts < num_experts)
        ctx.save_for_backward(inverse_rows, valid.to(torch.int8))
        ctx.num_rows = rows.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        inverse_rows, valid = ctx.saved_tensors
        num_tokens, topk = inverse_rows.shape
        hidden = grad_output.shape[-1]
        grad_output = grad_output.reshape(num_tokens, hidden)
        # Rows no (token, slot) owns keep a zero gradient.
        grad_rows = grad_output.new_zeros(ctx.num_rows, hidden)
        if num_tokens > 0:
            block_h = min(triton.next_power_of_2(hidden), 512)
            _rank_tree_unpermute_bwd_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
                grad_output,
                grad_output.stride(0),
                grad_output.stride(1),
                inverse_rows,
                valid,
                grad_rows,
                hidden,
                TOPK=topk,
                BLOCK_H=block_h,
                num_warps=4,
            )
        return grad_rows, None, None, None, None


def _unpermute_loop(
    permuted_tokens: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor],
    num_experts: int,
    inverse_map: torch.Tensor,
    ep_size: int,
) -> torch.Tensor:
    """Reference form of ``unpermute``: a loop over EP ranks and top-k slots."""
    input_dtype = permuted_tokens.dtype
    # Mirror the inference engine's summation tree exactly:
    # - When probs is None the activations were probability-weighted upstream
    #   (gated/SwiGLU convention) and the engine's within-rank _moe_sum
    #   accumulates in fp64 with unit weights -> mirror in fp64, round each
    #   rank partial to fp32 (the engine stores fp32 partials into the RSV
    #   buffer before the collective).
    # - The cross-rank stage matches the configured collective: "multimem"
    #   returns the correctly-rounded exact fp32 sum of the partials (mirror =
    #   fp64 accumulate, single final rounding); "ordered" is an ascending
    #   rank-order fp32 chain.
    within_rank_fp64 = probs is None
    cross_rank_fp64 = get_batch_invariant_collective() == "multimem"
    acc_dtype = torch.float64 if cross_rank_fp64 else torch.float32
    output_tokens = torch.zeros(restore_shape, dtype=acc_dtype, device=permuted_tokens.device)
    experts_per_rank = num_experts // ep_size
    inverse_rows = inverse_map[0]
    inverse_experts = inverse_map[1]
    topk = inverse_rows.size(1)

    partial_dtype = torch.float64 if within_rank_fp64 else torch.float32
    for ep_rank in range(ep_size):
        rank_partial = torch.zeros(
            restore_shape, dtype=partial_dtype, device=permuted_tokens.device
        )
        start_expert = ep_rank * experts_per_rank
        end_expert = start_expert + experts_per_rank

        for k in range(topk):
            row_ids = inverse_rows[:, k]
            expert_ids = inverse_experts[:, k]
            valid_mask = (row_ids >= 0) & (expert_ids >= start_expert) & (expert_ids < end_expert)

            safe_rows = row_ids.clamp_min(0)
            chunk = permuted_tokens.index_select(0, safe_rows).to(partial_dtype)
            if probs is not None:
                safe_experts = expert_ids.clamp_min(0)
                chunk = chunk * probs.gather(1, safe_experts.unsqueeze(1)).to(partial_dtype)
            chunk.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
            rank_partial += chunk

        # The engine stores each rank's partial as fp32 (the RSV buffer dtype)
        # before the cross-rank reduction; mirror that rounding point.
        output_tokens += rank_partial.to(torch.float32).to(acc_dtype)

    return output_tokens.to(torch.float32).to(dtype=input_dtype)


def unpermute(
    permuted_tokens: torch.Tensor,
    restore_shape: torch.Size,
    *,
    probs: Optional[torch.Tensor],
    num_experts: int,
    inverse_map: torch.Tensor,
    ep_size: Optional[int] = None,
) -> torch.Tensor:
    """Batch-invariant MoE unpermute.

    Accumulation is token-owned. The AllToAll inverse map avoids data-dependent
    shapes and adds contributions by EP rank then top-k slot, matching the
    inference NVLS rank-ordered combine. Without ``probs`` the tree runs as one
    Triton kernel; with ``probs`` it runs as a loop over EP ranks and top-k slots.
    ``ep_size`` is the expert-parallel size of the caller's token dispatcher, which
    splits the experts into ``ep_size`` contiguous per-rank shards. Omitting it is
    deprecated: the size of the global expert-parallel group is used, with a
    ``ProcessGroupFallbackWarning``.
    """
    if ep_size is None:
        # Deprecated fallback for callers that do not pass their dispatcher's EP size.
        ep_group = resolve_process_groups(
            None,
            owner="unpermute",
            required=("ep",),
            argument="ep_size",
            deprecated_in="0.21",
            removed_in="0.23",
        ).ep
        if ep_group is None and torch.distributed.is_initialized():
            raise ValueError(
                "unpermute needs ep_size, and parallel_state has no expert-parallel group. "
                "Pass the expert-parallel size of the token dispatcher."
            )
        # A single process without torch.distributed has one EP rank.
        ep_size = 1 if ep_group is None else ep_group.size()
    assert num_experts % ep_size == 0, "batch-invariant MoE expects contiguous EP shards"
    if probs is None and HAVE_TRITON and permuted_tokens.is_cuda:
        out = _RankTreeUnpermute.apply(
            permuted_tokens, inverse_map[0], inverse_map[1], ep_size, num_experts
        )
        return out.view(restore_shape)
    return _unpermute_loop(permuted_tokens, restore_shape, probs, num_experts, inverse_map, ep_size)
