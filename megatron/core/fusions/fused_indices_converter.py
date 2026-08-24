# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
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


# Assign a block to a row([1,topk]), generate a local routing map([1,num_of_local_experts])
@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  # bool
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    '''
    Triton kernel for converting indices to multihot representation.

    Input:
        indices: [num_of_tokens, topk]
        probs_in_indices: [num_of_tokens, topk]
    Output:
        multihot_indices: [num_of_tokens, num_of_local_experts]
        probs_in_multihot: [num_of_tokens, num_of_local_experts]

    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:

    Input Example:
        indices = [
                [0, 1],
                [1, 2]
            ]
        probs_in_indices = [
                [0.1, 0.2],
                [0.3, 0.4]
            ]
    Output Example:
        multihot_indices = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
        probs_in_multihot = [
                [0.1, 0.2, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.0]
            ]
    '''
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, topk] row from the indices buffer
    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    # Get the position of the each index in the indices_row, which is saved for backwards
    position_row = tl.where(indices_row != -1, topk_row, -1)
    # Mask of the valid indices
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    # Store to initialize
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()
    # Store the indices and probs_in_indices
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    # Store the position of the position_row for backwards
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)


# Assign a block to a row([1,topk]), generate a probs_indices([1,topk])
@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    '''
    Triton kernel for converting multihot representation to indices.

    Input:
        probs_in_multihot: [num_of_tokens, num_of_local_experts]
        position_map: [num_of_tokens, num_of_local_experts]
    Output:
        probs_indices: [num_of_tokens, topk]

    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:

    Input Example:
        probs_in_multihot = [
                [0.7, 0.8, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0]
            ]
        position_map = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
    Output Example:
        probs_indices = [
                [0.7, 0.8],
                [0.1, 0.9]
            ]
    '''
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, num_of_local_experts] row from the local routing map
    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask)

    # Get the original position of the valid value in the the indices
    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    # Store to initialize
    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()
    # Restore the indices and probs_indices
    tl.store(probs_indices_ptr + row_idx * topk + position_map_row, probs_in_multihot_row, mask)


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

    routed = tl.load(
        routing_map_ptr + row_offset + expert_offsets, mask=expert_mask, other=0
    ) != 0
    route_positions = tl.cumsum(routed.to(tl.int32), axis=0) - 1

    topk_offsets = tl.arange(0, topk_block_size)
    topk_mask = topk_offsets < topk
    output_row_offset = token_idx * topk
    tl.store(indices_ptr + output_row_offset + topk_offsets, -1, mask=topk_mask)
    tl.store(probs_indices_ptr + output_row_offset + topk_offsets, 0.0, mask=topk_mask)
    tl.debug_barrier()

    valid_route = routed & expert_mask & (route_positions < topk)
    routed_probs = tl.load(
        probs_ptr + row_offset + expert_offsets, mask=valid_route, other=0.0
    ).to(tl.float32)
    tl.store(
        indices_ptr + output_row_offset + route_positions,
        expert_offsets,
        mask=valid_route,
    )
    tl.store(
        probs_indices_ptr + output_row_offset + route_positions,
        routed_probs,
        mask=valid_route,
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
    indices = tl.load(
        indices_ptr + compact_row_offset + topk_offsets, mask=topk_mask, other=-1
    )
    valid_route = topk_mask & (indices >= 0) & (indices < num_experts)
    grad_probs_indices = tl.load(
        grad_probs_indices_ptr + compact_row_offset + topk_offsets,
        mask=valid_route,
        other=0.0,
    )
    tl.store(
        grad_probs_ptr + grad_probs_row_offset + indices,
        grad_probs_indices,
        mask=valid_route,
    )


class IndicesToMultihot(torch.autograd.Function):
    """Convert moe topk indices to multihot representation.

    This class implements a custom forward and backward propagation
    operation for efficiently converting indices to multihot
    representation.
    It is an experimental feature and may change in future versions.
    """

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        '''Forward function for IndicesToMultihot

        Convert indices to multihot representation.

        Args:
            indices: [num_of_tokens, topk]
            probs_indices: [num_of_tokens, topk]
            num_of_local_experts: int

        Returns:
            multihot_indices: [num_of_tokens, num_of_local_experts]
            probs_in_multihot: [num_of_tokens, num_of_local_experts]
        '''
        num_of_tokens = indices.shape[0]
        assert (
            indices.shape == probs_indices.shape
        ), "indices and probs_indices must have the same shape"
        topk = indices.shape[1]
        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda"
        )
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device="cuda"
        )
        position_map = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda"
        )
        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))
        grid = (num_of_tokens,)
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        '''Backward function for IndicesToMultihot

        Convert multihot probs representation to indices.
        indices is ignored in the backward function.

        Args:
            grad_multihot_indices: [num_of_tokens, num_of_local_experts]
            grad_probs_in_multihot: [num_of_tokens, num_of_local_experts]

        Returns:
            grad_probs_indices: [num_of_tokens, topk]
        '''
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        # Initialize the gradient of the indices and probs_indices
        grad_probs_indices = torch.empty(
            (num_of_tokens, topk), dtype=grad_probs_in_multihot.dtype, device="cuda"
        )
        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            # if the grad_probs_in_multihot is all-one/all-zero,
            # overlapping stride will cause error without contiguous()
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )
        return None, grad_probs_indices, None, None


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
        if not HAVE_TRITON:
            raise RuntimeError("fused_routing_map_to_indices requires Triton")
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
        probs_indices = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=probs.device
        )
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


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert moe topk indices to multihot representation.

    This function is an experimental feature and may change in future versions.
    """
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)


def fused_routing_map_to_indices(probs, routing_map, topk):
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
