# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Triton kernels for CUDA-graph-compatible MoE token permutation and unpermutation.

These kernels enable the torch grouped GEMM path to work under CUDA graphs
by keeping all metadata (tokens_per_expert, permutation indices) GPU-resident.
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


# --------------------------------------------------------------------------- #
# Kernel: Count tokens per local expert
# --------------------------------------------------------------------------- #
@triton.jit
def _count_local_tokens_kernel(
    routing_map_ptr,  # [num_tokens, topk] - global expert IDs
    tokens_per_expert_ptr,  # [num_local_experts] output (must be zero-initialized)
    total_pairs,
    local_expert_start,
    num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count tokens assigned to each local expert, filtering out non-local experts.

    Each program handles BLOCK_SIZE (token, k) pairs from the routing_map.
    Pairs whose assigned expert is not on this rank are ignored. For local
    experts, atomically increments the corresponding tokens_per_expert counter.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_pairs

    expert_ids = tl.load(routing_map_ptr + offsets, mask=mask, other=-1)
    local_ids = expert_ids - local_expert_start
    is_local = (local_ids >= 0) & (local_ids < num_local_experts) & mask

    # Scatter atomic add: each element adds 1 to its expert's counter
    tl.atomic_add(tokens_per_expert_ptr + local_ids, 1, mask=is_local)


# --------------------------------------------------------------------------- #
# Python wrapper
# --------------------------------------------------------------------------- #
def compute_local_tokens_per_expert(
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Count tokens routed to each local expert, filtering out non-local assignments.

    Scans the routing_map for (token, k) pairs whose assigned expert lives on
    this rank (global ID in [local_expert_start, local_expert_start + num_local_experts)).
    Pairs routed to experts on other ranks are ignored.

    Args:
        routing_map (torch.Tensor): Expert assignments, shape [num_tokens, topk].
            Contains global expert IDs.
        local_expert_start (int): First global expert index on this rank.
        num_local_experts (int): Number of experts on this rank.

    Returns:
        torch.Tensor: tokens_per_expert, shape [num_local_experts], dtype int32.
            Count of (token, k) pairs assigned to each local expert.
    """
    total_pairs = routing_map.numel()

    tokens_per_expert = torch.zeros(
        num_local_experts, dtype=torch.int32, device=routing_map.device
    )

    HIST_BLOCK = 256
    hist_grid = ((total_pairs + HIST_BLOCK - 1) // HIST_BLOCK,)
    _count_local_tokens_kernel[hist_grid](
        routing_map,
        tokens_per_expert,
        total_pairs,
        local_expert_start,
        num_local_experts,
        BLOCK_SIZE=HIST_BLOCK,
    )

    return tokens_per_expert


# --------------------------------------------------------------------------- #
# Kernel: Exclusive prefix sum (single block)
# --------------------------------------------------------------------------- #
@triton.jit
def _prefix_sum_kernel(
    tokens_per_expert_ptr,  # [num_local_experts] input
    expert_offsets_ptr,  # [num_local_experts] output
    num_local_experts,
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(num_local_experts)
):
    """Compute exclusive prefix sum of tokens_per_expert.

    Runs as a single block. Reads tokens_per_expert, computes exclusive prefix
    sum via tl.cumsum, and writes expert_offsets.
    """
    expert_range = tl.arange(0, BLOCK_SIZE)
    mask = expert_range < num_local_experts
    histogram = tl.load(tokens_per_expert_ptr + expert_range, mask=mask, other=0)

    # Inclusive prefix sum, then shift to exclusive
    inclusive = tl.cumsum(histogram, axis=0)
    exclusive = inclusive - histogram

    tl.store(expert_offsets_ptr + expert_range, exclusive, mask=mask)


# --------------------------------------------------------------------------- #
# Python wrapper
# --------------------------------------------------------------------------- #
def compute_expert_offsets(
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Compute exclusive prefix sum of tokens_per_expert.

    Args:
        tokens_per_expert (torch.Tensor): Token counts per local expert,
            shape [num_local_experts], dtype int32.

    Returns:
        torch.Tensor: expert_offsets, shape [num_local_experts].
            Exclusive prefix sum: expert_offsets[i] is the start index of
            expert i's tokens in the permuted buffer. Passed to permute_tokens
            which mutates it in-place via atomic adds, turning the exclusive
            start offsets into inclusive end offsets (i.e. expert_offsets[i]
            becomes the end index of expert i's tokens after permutation).
    """
    num_local_experts = tokens_per_expert.shape[0]

    expert_offsets = torch.empty_like(tokens_per_expert)

    BLOCK_SIZE = triton.next_power_of_2(num_local_experts)
    _prefix_sum_kernel[(1,)](
        tokens_per_expert,
        expert_offsets,
        num_local_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_offsets


# --------------------------------------------------------------------------- #
# Kernel: Permute tokens by expert assignment
# --------------------------------------------------------------------------- #
@triton.jit
def _permute_tokens_kernel(
    # Input pointers
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    probs_ptr,  # [num_tokens, topk]
    routing_map_ptr,  # [num_tokens, topk]
    # Output pointers
    permuted_hidden_ptr,  # [output_size, hidden_dim]
    permuted_probs_ptr,  # [output_size]
    source_token_indices_ptr,  # [output_size]
    # Atomic counters (mutated in-place)
    atomic_counters_ptr,  # [num_local_experts]
    # Dimensions
    num_tokens,
    hidden_dim,
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,  # tile size for hidden_dim copy loop
):
    """Permute tokens into expert-grouped order.

    Each program handles one (token, k) pair. If the assigned expert is local,
    it atomically claims a write position and copies the token's hidden state,
    routing probability, and source token index to the output buffers.
    The hidden dimension is copied in tiles of BLOCK_H to support large hidden sizes.
    """
    pair_idx = tl.program_id(0)
    token_idx = pair_idx // topk
    k_idx = pair_idx % topk

    if token_idx >= num_tokens:
        return

    expert_id = tl.load(routing_map_ptr + token_idx * topk + k_idx)
    local_idx = expert_id - local_expert_start

    if local_idx < 0 or local_idx >= num_local_experts:
        return

    # Atomically claim a write position
    write_pos = tl.atomic_add(atomic_counters_ptr + local_idx, 1)

    # Copy hidden state row in tiles of BLOCK_H
    src_row_ptr = hidden_states_ptr + token_idx * hidden_dim
    dst_row_ptr = permuted_hidden_ptr + write_pos * hidden_dim
    for h_start in tl.range(0, hidden_dim, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_dim
        src_vals = tl.load(src_row_ptr + h_offsets, mask=h_mask)
        tl.store(dst_row_ptr + h_offsets, src_vals, mask=h_mask)

    # Copy the routing probability for this (token, k) pair
    prob_val = tl.load(probs_ptr + token_idx * topk + k_idx)
    tl.store(permuted_probs_ptr + write_pos, prob_val)

    # Record source token index for the unpermute kernel
    tl.store(source_token_indices_ptr + write_pos, token_idx)


# --------------------------------------------------------------------------- #
# Python wrapper
# --------------------------------------------------------------------------- #
def permute_tokens(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    expert_offsets: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
) -> tuple:
    """Permute tokens into expert-grouped order for the torch grouped GEMM.

    Scatters tokens into an output buffer where all tokens for expert 0 come
    first, then expert 1, etc. Uses expert_offsets (from compute_expert_offsets)
    to assign write positions.

    NOTE: expert_offsets is mutated in-place. On entry it contains exclusive
    start offsets (expert_offsets[i] = start index of expert i's region).
    The kernel atomically increments each entry as it places tokens, so on
    exit expert_offsets[i] = end index (inclusive) of expert i's region.
    The last entry equals the total number of routed tokens.

    Args:
        hidden_states (torch.Tensor): Input hidden states, shape [num_tokens, hidden_dim].
        probs (torch.Tensor): Routing probabilities, shape [num_tokens, topk].
        routing_map (torch.Tensor): Expert assignments, shape [num_tokens, topk].
            Contains global expert IDs.
        expert_offsets (torch.Tensor): Write position counters, shape [num_local_experts].
            Initialized to exclusive prefix sum by compute_expert_offsets.
            Mutated in-place to inclusive end offsets by the permute kernel.
        local_expert_start (int): First global expert index on this rank.
        num_local_experts (int): Number of experts on this rank.

    Returns:
        tuple: (permuted_hidden_states, permuted_probs, source_token_indices) where:
            - permuted_hidden_states: [output_size, hidden_dim] tokens grouped by expert.
            - permuted_probs: [output_size] scalar prob per permuted slot.
            - source_token_indices: [output_size] original token index per permuted slot.
            output_size = num_tokens * min(topk, num_local_experts).
            Slots beyond the actual routed token count contain uninitialized data.
    """
    num_tokens, hidden_dim = hidden_states.shape
    topk = probs.shape[1]
    output_size = num_tokens * min(topk, num_local_experts)

    # Allocate output buffers (statically sized for CUDA graph compatibility)
    permuted_hidden = torch.empty(
        output_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device
    )
    permuted_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    source_token_indices = torch.empty(output_size, dtype=torch.int32, device=probs.device)

    total_pairs = num_tokens * topk
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    # After this kernel, expert_offsets is mutated: exclusive start offsets
    # become inclusive end offsets (expert_offsets[-1] = total routed tokens).
    _permute_tokens_kernel[(total_pairs,)](
        hidden_states,
        probs,
        routing_map,
        permuted_hidden,
        permuted_probs,
        source_token_indices,
        expert_offsets,
        num_tokens,
        hidden_dim,
        topk,
        local_expert_start,
        num_local_experts,
        BLOCK_H=BLOCK_H,
    )

    return permuted_hidden, permuted_probs, source_token_indices


# --------------------------------------------------------------------------- #
# Kernel: Unpermute (accumulate expert outputs back to token positions)
# --------------------------------------------------------------------------- #
@triton.jit
def _unpermute_tokens_kernel(
    # Input pointers
    expert_output_ptr,  # [output_size, hidden_dim]
    permuted_probs_ptr,  # [output_size]
    source_token_indices_ptr,  # [output_size]
    # GPU-resident valid count (read from last atomic counter after permute)
    num_routed_slots_ptr,  # scalar tensor on GPU
    # Output pointer
    output_ptr,  # [num_tokens, hidden_dim] - must be zero-initialized
    # Dimensions
    hidden_dim,
    BLOCK_H: tl.constexpr,  # tile size for hidden_dim loop
):
    """Accumulate weighted expert outputs back into original token positions.

    Each program handles one row of the permuted expert output. It reads the
    source token index, multiplies the expert output by the routing probability,
    and atomically adds the result to the corresponding row in the output buffer.
    Multiple experts contributing to the same token are summed via atomic adds.
    Rows beyond num_routed_slots (read from GPU) are skipped.
    The hidden dimension is processed in tiles of BLOCK_H to support large hidden sizes.
    """
    row_idx = tl.program_id(0)

    # Read valid count from GPU — no host sync needed for CUDA graphability
    num_valid = tl.load(num_routed_slots_ptr)
    if row_idx >= num_valid:
        return

    token_idx = tl.load(source_token_indices_ptr + row_idx)
    prob = tl.load(permuted_probs_ptr + row_idx)

    src_row_ptr = expert_output_ptr + row_idx * hidden_dim
    dst_row_ptr = output_ptr + token_idx * hidden_dim
    for h_start in tl.range(0, hidden_dim, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_dim
        expert_vals = tl.load(src_row_ptr + h_offsets, mask=h_mask)
        scaled_vals = expert_vals * prob
        tl.atomic_add(dst_row_ptr + h_offsets, scaled_vals, mask=h_mask)


# --------------------------------------------------------------------------- #
# Python wrapper
# --------------------------------------------------------------------------- #
def unpermute_tokens(
    expert_output: torch.Tensor,
    permuted_probs: torch.Tensor,
    source_token_indices: torch.Tensor,
    num_tokens: int,
    num_routed_slots: torch.Tensor,
) -> torch.Tensor:
    """Unpermute expert outputs back to original token order.

    Accumulates weighted expert outputs into the original token positions.
    For each valid permuted row i, computes:
        output[source_token_indices[i], :] += expert_output[i, :] * permuted_probs[i]
    Multiple experts contributing to the same token are summed via atomic adds.
    Rows beyond num_routed_slots are skipped. num_routed_slots is read from GPU
    to avoid host synchronization, keeping the pipeline CUDA-graphable.

    Args:
        expert_output (torch.Tensor): Expert outputs, shape [output_size, hidden_dim].
            Only the first total_routed rows are valid.
        permuted_probs (torch.Tensor): Routing probs, shape [output_size].
            Only the first total_routed entries are valid.
        source_token_indices (torch.Tensor): Source token index for each permuted row,
            shape [output_size], dtype int32. Only the first total_routed are valid.
        num_tokens (int): Number of original tokens (for output allocation).
        num_routed_slots (torch.Tensor): 1-element GPU tensor containing the
            total number of routed tokens. Must stay on GPU (no host sync).

    Returns:
        torch.Tensor: Unpermuted output, shape [num_tokens, hidden_dim].
            Each row is the weighted sum of expert outputs for that token.
    """
    output_size, hidden_dim = expert_output.shape

    # Zero-initialized output (atomic adds accumulate into this)
    output = torch.zeros(
        num_tokens, hidden_dim, dtype=expert_output.dtype, device=expert_output.device
    )

    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    _unpermute_tokens_kernel[(output_size,)](
        expert_output,
        permuted_probs,
        source_token_indices,
        num_routed_slots,
        output,
        hidden_dim,
        BLOCK_H=BLOCK_H,
    )

    return output


