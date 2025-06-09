# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch
import triton
import triton.language as tl

from megatron.core.utils import experimental_fn


@triton.jit
def _pad_routing_map_kernel(
    routing_map_ptr, output_ptr, num_tokens, pad_multiple: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    expert_idx = tl.program_id(axis=0)

    # Pointers for the current expert's row
    row_offset = expert_idx * num_tokens
    input_row_ptr = routing_map_ptr + row_offset
    output_row_ptr = output_ptr + row_offset

    # Token indices for this block
    token_indices = tl.arange(0, BLOCK_SIZE)
    token_mask = token_indices < num_tokens

    # Load the row for the current expert, masking out-of-bounds elements
    row = tl.load(input_row_ptr + token_indices, mask=token_mask, other=0)

    # 1. Calculate num_ones for the current expert
    # Ensure summation happens correctly even with masking
    # Convert boolean/int row to int if necessary before sum
    num_ones = tl.sum(row.to(tl.int32), axis=0)

    # 2. Calculate num_to_pad for the current expert
    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    # 3. Calculate zero ranks using cumsum (vectorized)
    is_zero = row == 0
    # Cast to int32 for cumsum
    zero_ranks = tl.cumsum(is_zero.to(tl.int32), axis=0)

    # 4. Create mask for elements to be flipped to 1
    # Only flip if the element is zero AND its rank is within the padding limit
    mask_to_flip = (zero_ranks <= num_to_pad) & is_zero

    # 5. Determine the output row values
    output_row = tl.where(mask_to_flip, 1, row)

    # 6. Store the result, masking out-of-bounds elements
    tl.store(output_row_ptr + token_indices, output_row, mask=token_mask)


@experimental_fn(introduced_with_version="0.13.0")
def fused_pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Fused version of pad_routing_map.
    Args:
        routing_map (torch.Tensor): A boolean or integer tensor of shape [num_tokens,
            num_experts] indicating which tokens are routed to which experts.
        pad_multiple (int): The multiple to pad each expert's token count to.

    Returns:
        torch.Tensor: The padded routing map of shape [num_tokens, num_experts].
    """
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    input_map = routing_map.transpose(0, 1).contiguous().int()  # [num_experts, num_tokens]

    output_map = torch.empty_like(input_map)

    # Kernel launch
    grid = (num_experts,)
    BLOCK_SIZE = triton.next_power_of_2(num_tokens)

    _pad_routing_map_kernel[grid](
        input_map, output_map, num_tokens, pad_multiple, BLOCK_SIZE=BLOCK_SIZE
    )

    return output_map.transpose(0, 1)  # [num_tokens, num_experts]
