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


if __name__ == "__main__":
    torch.manual_seed(42)

    # --- Config ---
    num_tokens = 128
    topk = 8
    num_total_experts = 64
    num_local_experts = 8
    local_expert_start = 16  # this rank owns experts 16..23

    # --- Build a random routing_map with global expert IDs ---
    routing_map = torch.randint(
        0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    # --- Reference: count with PyTorch ---
    flat = routing_map.flatten()
    local_mask = (flat >= local_expert_start) & (flat < local_expert_start + num_local_experts)
    local_ids_ref = flat[local_mask] - local_expert_start
    ref = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
    ref.scatter_add_(0, local_ids_ref.long(), torch.ones_like(local_ids_ref, dtype=torch.int32))

    # --- Triton kernel ---
    result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)

    # --- Compare ---
    print(f"Reference: {ref.tolist()}")
    print(f"Triton:    {result.tolist()}")
    assert torch.equal(ref, result), f"MISMATCH!\n  ref={ref}\n  got={result}"
    print("PASSED - histogram matches reference")

    # --- Edge cases ---
    # All tokens routed to non-local experts
    routing_map_none = torch.zeros(
        num_tokens, topk, dtype=torch.int32, device="cuda"
    )  # expert 0, not in [16..23]
    result_none = compute_local_tokens_per_expert(routing_map_none, local_expert_start, num_local_experts)
    assert torch.equal(result_none, torch.zeros(num_local_experts, dtype=torch.int32, device="cuda"))
    print("PASSED - no local experts case")

    # All tokens routed to a single local expert
    routing_map_single = torch.full(
        (num_tokens, topk), local_expert_start + 3, dtype=torch.int32, device="cuda"
    )
    result_single = compute_local_tokens_per_expert(routing_map_single, local_expert_start, num_local_experts)
    expected_single = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
    expected_single[3] = num_tokens * topk
    assert torch.equal(result_single, expected_single)
    print("PASSED - single expert case")

    # Small: 1 token, topk=1
    routing_map_tiny = torch.tensor([[local_expert_start]], dtype=torch.int32, device="cuda")
    result_tiny = compute_local_tokens_per_expert(routing_map_tiny, local_expert_start, num_local_experts)
    expected_tiny = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
    expected_tiny[0] = 1
    assert torch.equal(result_tiny, expected_tiny)
    print("PASSED - single token case")

    print("\nAll tests passed.")
