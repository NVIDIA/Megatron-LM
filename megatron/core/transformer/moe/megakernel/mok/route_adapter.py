# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Convert authoritative MCore routes into MOK scheduler inputs."""

from __future__ import annotations

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
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


@triton.jit
def _routing_map_to_indices_kernel(
    probs_ptr,
    routing_map_ptr,
    probs_indices_ptr,
    indices_ptr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    num_experts_block_size: tl.constexpr,
    topk_block_size: tl.constexpr,
):
    """Compact one dense routing-map row and gather its router probabilities."""
    token_idx = tl.program_id(0)
    expert_offsets = tl.arange(0, num_experts_block_size)
    expert_mask = expert_offsets < num_experts
    row_offset = token_idx * num_experts
    routed = tl.load(routing_map_ptr + row_offset + expert_offsets, mask=expert_mask, other=0) != 0
    route_positions = tl.cumsum(routed.to(tl.int32), axis=0) - 1
    topk_offsets = tl.arange(0, topk_block_size)
    topk_mask = topk_offsets < topk
    output_row_offset = token_idx * topk
    tl.store(indices_ptr + output_row_offset + topk_offsets, -1, mask=topk_mask)
    tl.store(probs_indices_ptr + output_row_offset + topk_offsets, 0.0, mask=topk_mask)
    tl.debug_barrier()
    valid_route = routed & expert_mask & (route_positions < topk)
    routed_probs = tl.load(probs_ptr + row_offset + expert_offsets, mask=valid_route, other=0.0).to(
        tl.float32
    )
    tl.store(indices_ptr + output_row_offset + route_positions, expert_offsets, mask=valid_route)
    tl.store(
        probs_indices_ptr + output_row_offset + route_positions, routed_probs, mask=valid_route
    )


@triton.jit
def _routing_map_to_indices_backward_kernel(
    grad_probs_indices_ptr,
    indices_ptr,
    grad_probs_ptr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    num_experts_block_size: tl.constexpr,
    topk_block_size: tl.constexpr,
):
    """Scatter compact router-weight gradients back into the dense probability row."""
    token_idx = tl.program_id(0)
    expert_offsets = tl.arange(0, num_experts_block_size)
    expert_mask = expert_offsets < num_experts
    grad_probs_row_offset = token_idx * num_experts
    tl.store(grad_probs_ptr + grad_probs_row_offset + expert_offsets, 0.0, mask=expert_mask)
    tl.debug_barrier()
    topk_offsets = tl.arange(0, topk_block_size)
    topk_mask = topk_offsets < topk
    compact_row_offset = token_idx * topk
    indices = tl.load(indices_ptr + compact_row_offset + topk_offsets, mask=topk_mask, other=-1)
    valid_route = topk_mask & (indices >= 0) & (indices < num_experts)
    grad_probs_indices = tl.load(
        grad_probs_indices_ptr + compact_row_offset + topk_offsets, mask=valid_route, other=0.0
    )
    tl.store(grad_probs_ptr + grad_probs_row_offset + indices, grad_probs_indices, mask=valid_route)


class RoutingMapToIndices(torch.autograd.Function):
    """Compact a dense routing map into fixed-width expert indices and probabilities.
    Unlike torch.topk(probs), this operation treats routing_map as the
    authoritative route set. Each row is compacted in increasing expert-index
    order. Rows with fewer than topk routes are padded with -1 indices
    and zero probabilities. The backward pass scatters router-weight gradients
    into the corresponding positions of the dense probability tensor.
    """

    @staticmethod
    def forward(ctx, probs, routing_map, topk):
        """Compact authoritative dense routes into fixed-width MOK inputs."""
        if not HAVE_TRITON:
            raise RuntimeError("routing_map_to_mok_inputs requires Triton")
        if probs.ndim != 2 or routing_map.ndim != 2:
            raise ValueError("probs and routing_map must both be two-dimensional")
        if probs.shape != routing_map.shape:
            raise ValueError("probs and routing_map must have the same shape")
        if not probs.is_cuda or not routing_map.is_cuda or probs.device != routing_map.device:
            raise ValueError("probs and routing_map must be CUDA tensors on the same device")
        if not probs.is_contiguous() or not routing_map.is_contiguous():
            raise ValueError("probs and routing_map must be contiguous")
        if probs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("probs must have dtype float16, bfloat16, or float32")
        if type(topk) is not int or topk <= 0 or topk > probs.shape[1]:
            raise ValueError("topk must be a positive integer no greater than num_experts")
        num_tokens, num_experts = probs.shape
        probs_indices = torch.empty((num_tokens, topk), dtype=torch.float32, device=probs.device)
        indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=probs.device)
        if num_tokens > 0:
            num_experts_block_size = triton.next_power_of_2(num_experts)
            topk_block_size = triton.next_power_of_2(topk)
            num_warps = 4 if num_experts_block_size <= 512 else 8
            _routing_map_to_indices_kernel[(num_tokens,)](
                probs,
                routing_map,
                probs_indices,
                indices,
                num_experts=num_experts,
                topk=topk,
                num_experts_block_size=num_experts_block_size,
                topk_block_size=topk_block_size,
                num_warps=num_warps,
            )
        ctx.save_for_backward(indices)
        ctx.num_experts = num_experts
        ctx.probs_dtype = probs.dtype
        ctx.mark_non_differentiable(indices)
        return probs_indices, indices

    @staticmethod
    def backward(ctx, grad_probs_indices, grad_indices):
        """Scatter compact routing-weight gradients back to the dense tensor."""
        del grad_indices
        (indices,) = ctx.saved_tensors
        num_tokens, topk = indices.shape
        grad_probs = torch.empty(
            (num_tokens, ctx.num_experts), dtype=ctx.probs_dtype, device=indices.device
        )
        if num_tokens > 0:
            num_experts_block_size = triton.next_power_of_2(ctx.num_experts)
            topk_block_size = triton.next_power_of_2(topk)
            num_warps = 4 if num_experts_block_size <= 512 else 8
            _routing_map_to_indices_backward_kernel[(num_tokens,)](
                grad_probs_indices.contiguous(),
                indices,
                grad_probs,
                num_experts=ctx.num_experts,
                topk=topk,
                num_experts_block_size=num_experts_block_size,
                topk_block_size=topk_block_size,
                num_warps=num_warps,
            )
        return grad_probs, None, None


# TODO: Remove this standalone conversion once the megakernel scheduling interface can either
# consume MCore's authoritative ``routing_map`` and ``probs`` directly or accept authoritative
# compact routes emitted by the router. That avoids an extra kernel launch and materializing
# intermediate ``[num_tokens, topk]`` tensors solely for the scheduler.
def routing_map_to_mok_inputs(
    probs: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact an authoritative routing map into fixed-width MOK inputs.
    Args:
        probs: Dense router probabilities with shape [num_tokens, num_experts].
        routing_map: Dense boolean/integer route map with the same shape as probs.
        topk: Maximum number of valid routes emitted per token.
    Returns:
        A pair (probs_indices, indices) with shapes [num_tokens, topk].
        probs_indices is float32 and indices is int32 with -1 padding.
    """
    return RoutingMapToIndices.apply(probs, routing_map, topk)
