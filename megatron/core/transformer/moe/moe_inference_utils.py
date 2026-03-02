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
# Kernel: Exclusive prefix sum + atomic counters (single block)
# --------------------------------------------------------------------------- #
@triton.jit
def _prefix_sum_kernel(
    tokens_per_expert_ptr,  # [num_local_experts] input
    expert_offsets_ptr,  # [num_local_experts] output
    atomic_counters_ptr,  # [num_local_experts] output (copy of offsets)
    num_local_experts,
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(num_local_experts)
):
    """Compute exclusive prefix sum of tokens_per_expert.

    Runs as a single block. Reads tokens_per_expert, computes exclusive prefix
    sum via tl.cumsum, and writes expert_offsets and a copy as atomic_counters
    for use by the permute kernel.
    """
    expert_range = tl.arange(0, BLOCK_SIZE)
    mask = expert_range < num_local_experts
    histogram = tl.load(tokens_per_expert_ptr + expert_range, mask=mask, other=0)

    # Inclusive prefix sum, then shift to exclusive
    inclusive = tl.cumsum(histogram, axis=0)
    exclusive = inclusive - histogram

    tl.store(expert_offsets_ptr + expert_range, exclusive, mask=mask)
    tl.store(atomic_counters_ptr + expert_range, exclusive, mask=mask)


# --------------------------------------------------------------------------- #
# Python wrapper
# --------------------------------------------------------------------------- #
def compute_expert_offsets(
    tokens_per_expert: torch.Tensor,
) -> tuple:
    """Compute exclusive prefix sum of tokens_per_expert and a mutable copy for atomics.

    Args:
        tokens_per_expert (torch.Tensor): Token counts per local expert,
            shape [num_local_experts], dtype int32.

    Returns:
        tuple: (expert_offsets, atomic_counters) where:
            - expert_offsets: [num_local_experts] exclusive prefix sum (read-only).
            - atomic_counters: [num_local_experts] same values as expert_offsets,
              to be mutated by the permute kernel's atomic adds.
    """
    num_local_experts = tokens_per_expert.shape[0]

    expert_offsets = torch.empty_like(tokens_per_expert)
    atomic_counters = torch.empty_like(tokens_per_expert)

    BLOCK_SIZE = triton.next_power_of_2(num_local_experts)
    _prefix_sum_kernel[(1,)](
        tokens_per_expert,
        expert_offsets,
        atomic_counters,
        num_local_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_offsets, atomic_counters


