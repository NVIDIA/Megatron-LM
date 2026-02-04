# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Triton kernels for MoE inference optimizations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def shift_and_mark_indices_kernel(
    topk_indices_ptr,  # Input: [num_tokens, topk]
    shifted_indices_ptr,  # Output: [num_tokens, topk]
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    local_start: tl.constexpr,  # First local expert index
    local_end: tl.constexpr,  # Last local expert index
    sentinel: tl.constexpr,  # num_local_experts (sentinel for invalid)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Shifts topk indices to local coordinate system and marks invalid indices.
    
    For each index:
    - If index in [local_start, local_end]: shift to 0-based (index - local_start)
    - Otherwise: mark as sentinel value
    """
    # Each program handles one block of elements
    pid = tl.program_id(0)
    
    # Calculate total elements
    num_elements = num_tokens * topk
    
    # Process BLOCK_SIZE elements per program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    
    # Load indices
    indices = tl.load(topk_indices_ptr + offset, mask=mask, other=0)
    
    # Check if index is in local range
    is_valid = (indices >= local_start) & (indices <= local_end)
    
    # Shift valid indices, mark invalid with sentinel
    shifted = tl.where(is_valid, indices - local_start, sentinel)
    
    # Store result
    tl.store(shifted_indices_ptr + offset, shifted, mask=mask)


def shift_topk_indices(
    topk_indices: torch.Tensor,
    local_start: int,
    local_end: int,
    num_local_experts: int,
) -> torch.Tensor:
    """
    Shift topk indices to local coordinate system using Triton kernel.
    
    Args:
        topk_indices: [num_tokens, topk] tensor of expert indices
        local_start: First local expert global index
        local_end: Last local expert global index
        num_local_experts: Number of local experts
        
    Returns:
        shifted_indices: [num_tokens, topk] with local indices or sentinel
    """
    num_tokens, topk = topk_indices.shape
    shifted_indices = torch.empty_like(topk_indices)
    
    num_elements = num_tokens * topk
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    shift_and_mark_indices_kernel[grid](
        topk_indices,
        shifted_indices,
        num_tokens=num_tokens,
        topk=topk,
        local_start=local_start,
        local_end=local_end,
        sentinel=num_local_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return shifted_indices


@triton.jit
def permute_and_count_kernel(
    # Input tensors
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    probs_ptr,  # [num_tokens, topk]
    expert_assignments_ptr,  # [num_tokens * topk] - local expert index per token-k pair
    permutation_ptr,  # [num_tokens * topk] - argsort result
    # Output tensors
    permuted_hidden_ptr,  # [max_out, hidden_dim]
    permuted_probs_ptr,  # [max_out]
    tokens_per_expert_ptr,  # [num_local_experts]
    # Scalars
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Permute hidden states and probs according to permutation, count tokens per expert.
    
    Each program handles one output position. Skips sentinel values (expert == num_local_experts).
    """
    # Each program handles one output position
    pid = tl.program_id(0)
    
    # Total elements to process
    num_elements = num_tokens * topk
    
    if pid >= num_elements:
        return
    
    # Load the permutation index - where to read from
    perm_idx = tl.load(permutation_ptr + pid)
    
    # Load the expert index for this position
    expert_idx = tl.load(expert_assignments_ptr + perm_idx)
    
    # Skip if this is a sentinel value
    if expert_idx >= num_local_experts:
        return
    
    # Compute source token and k indices
    # perm_idx tells us position in flattened [num_tokens * topk] array
    token_idx = perm_idx // topk
    k_idx = perm_idx % topk
    
    # Copy hidden state: load from [token_idx, :] and store to [pid, :]
    for d in range(0, hidden_dim, BLOCK_SIZE):
        offset = d + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_dim
        
        hidden_val = tl.load(
            hidden_states_ptr + token_idx * hidden_dim + offset,
            mask=mask,
            other=0.0
        )
        tl.store(
            permuted_hidden_ptr + pid * hidden_dim + offset,
            hidden_val,
            mask=mask
        )
    
    # Copy prob: load from [token_idx, k_idx]
    prob_val = tl.load(probs_ptr + token_idx * topk + k_idx)
    tl.store(permuted_probs_ptr + pid, prob_val)
    
    # Atomically increment tokens_per_expert[expert_idx]
    tl.atomic_add(tokens_per_expert_ptr + expert_idx, 1)


def permute_tokens_and_probs(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    expert_assignments: torch.Tensor,
    permutation: torch.Tensor,
    num_local_experts: int,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute hidden states and probs, count tokens per expert using Triton kernel.
    
    Args:
        hidden_states: [num_tokens, hidden_dim]
        probs: [num_tokens, topk]
        expert_assignments: [num_tokens * topk] local expert index per token-k pair
        permutation: [num_tokens * topk] argsort result
        num_local_experts: Number of local experts
        max_tokens: Maximum output size
        
    Returns:
        permuted_hidden: [max_tokens, hidden_dim]
        permuted_probs: [max_tokens]
        tokens_per_expert: [num_local_experts]
    """
    num_tokens, hidden_dim = hidden_states.shape
    topk = probs.size(1)
    
    # Allocate outputs
    permuted_hidden = torch.empty(
        (max_tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    permuted_probs = torch.empty(
        max_tokens,
        dtype=probs.dtype,
        device=probs.device
    )
    tokens_per_expert = torch.zeros(
        num_local_experts,
        dtype=torch.int32,
        device=hidden_states.device
    )
    
    # Launch kernel - one program per output position
    num_elements = num_tokens * topk
    
    # Adapt BLOCK_SIZE to hidden_dim for optimal memory access
    # Use next power of 2 for better vectorization
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    # Cap at reasonable maximum to avoid register pressure
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)
    
    grid = (num_elements,)
    
    permute_and_count_kernel[grid](
        hidden_states,
        probs,
        expert_assignments,
        permutation,
        permuted_hidden,
        permuted_probs,
        tokens_per_expert,
        num_tokens=num_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        num_local_experts=num_local_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return permuted_hidden, permuted_probs, tokens_per_expert


@triton.jit
def unpermute_and_combine_kernel(
    # Input tensors
    permuted_hidden_ptr,  # [max_out, hidden_dim] - expert outputs
    permutation_ptr,  # [num_tokens * topk] - argsort result (forward permutation)
    expert_assignments_ptr,  # [num_tokens * topk] - local expert index per token-k pair
    # Output tensor
    output_ptr,  # [num_tokens, hidden_dim] - unpermuted output
    # Scalars
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unpermute expert outputs back to original token positions.
    
    Each program handles one position in the permutation array:
    - Loads permutation[pid] to find source flat_pos (token_idx, k_idx)
    - Loads expert output from permuted arrays at position pid
    - Atomically accumulates output into output[token_idx]
    
    Note: Probability weighting is handled by the experts (via moe_apply_probs_on_input),
    so this kernel only does unpermutation and accumulation.
    """
    # Each program handles one permuted position
    pid = tl.program_id(0)
    
    num_elements = num_tokens * topk
    if pid >= num_elements:
        return
    
    # Load source position from permutation
    flat_pos = tl.load(permutation_ptr + pid)
    
    # Compute source token index
    token_idx = flat_pos // topk
    
    # Load expert index to check validity
    expert_idx = tl.load(expert_assignments_ptr + flat_pos)
    
    # Skip if sentinel (not a valid local expert)
    if expert_idx >= num_local_experts:
        return
    
    # Process each dimension chunk
    for d in range(0, hidden_dim, BLOCK_SIZE):
        offset = d + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_dim
        
        # Load expert output (already weighted by experts if configured)
        hidden_val = tl.load(
            permuted_hidden_ptr + pid * hidden_dim + offset,
            mask=mask,
            other=0.0
        )
        
        # Atomically accumulate into output[token_idx]
        tl.atomic_add(output_ptr + token_idx * hidden_dim + offset, hidden_val, mask=mask)


def unpermute_and_combine(
    permuted_hidden: torch.Tensor,
    expert_assignments: torch.Tensor,
    permutation: torch.Tensor,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
) -> torch.Tensor:
    """
    Unpermute expert outputs back to original token order.
    
    Args:
        permuted_hidden: [max_out, hidden_dim] expert outputs (already weighted by experts)
        expert_assignments: [num_tokens * topk] local expert index per token-k pair
        permutation: [num_tokens * topk] argsort result from dispatch
        num_tokens: Number of original tokens
        topk: Number of experts per token
        num_local_experts: Number of local experts
        
    Returns:
        output: [num_tokens, hidden_dim] unpermuted output
        
    Note: The expert outputs should already be weighted by routing probabilities
    if moe_apply_probs_on_input is enabled in the config.
    """
    hidden_dim = permuted_hidden.size(1)
    
    # Allocate output (zeroed for atomic accumulation)
    output = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=permuted_hidden.dtype,
        device=permuted_hidden.device
    )
    
    # Adapt BLOCK_SIZE to hidden_dim
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)
    
    # Launch kernel - one program per permuted position (same pattern as permute kernel)
    num_elements = num_tokens * topk
    grid = (num_elements,)
    
    unpermute_and_combine_kernel[grid](
        permuted_hidden,
        permutation,
        expert_assignments,
        output,
        num_tokens=num_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        num_local_experts=num_local_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def launch_fused_permute_and_probs(*args, **kwargs):
    """Placeholder for future fused permute kernel."""
    raise NotImplementedError("launch_fused_permute_and_probs not yet implemented")


def launch_unpermute_kernel(*args, **kwargs):
    """Placeholder for future unpermute kernel."""
    raise NotImplementedError("launch_unpermute_kernel not yet implemented")
