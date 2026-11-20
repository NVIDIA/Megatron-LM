# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-expert token counts of a boolean routing map in one kernel.

``routing_map.sum(dim=0)`` on a ``[num_tokens, num_experts]`` bool map is a column reduction
that PyTorch runs as a multi-pass reduction (a cast to int64 plus a reduce). This kernel loads
row tiles of the map, sums each tile's columns in registers and adds the per-tile counts to the
output with integer atomics. Integer addition is associative, so the result is exact and does
not depend on the order in which the tiles land.
"""

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

# Upper bound on the number of map elements one program loads.
_MAX_TILE_ELEMENTS = 16384


@triton.jit
def _tokens_per_expert_kernel(
    map_ptr, out_ptr, num_rows, num_experts, BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr
):
    """Add the column sums of rows [pid * BLOCK_R, (pid + 1) * BLOCK_R) of the map to out."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_E)
    col_mask = cols < num_experts
    mask = (rows[:, None] < num_rows) & col_mask[None, :]
    offsets = rows[:, None].to(tl.int64) * num_experts + cols[None, :]
    values = tl.load(map_ptr + offsets, mask=mask, other=0)
    counts = tl.sum(values.to(tl.int32), axis=0)
    tl.atomic_add(out_ptr + cols, counts.to(tl.int64), mask=col_mask)


def fused_tokens_per_expert(routing_map: torch.Tensor) -> torch.Tensor:
    """Number of tokens routed to each expert, ``routing_map.sum(dim=0)``.

    Args:
        routing_map (torch.Tensor): ``[num_tokens, num_experts]`` routing map.

    Returns:
        torch.Tensor: ``[num_experts]`` int64 counts, equal to ``routing_map.sum(dim=0)``.
        Maps the kernel does not handle (not a contiguous 2D bool CUDA tensor, or Triton
        unavailable) take ``routing_map.sum(dim=0)``.
    """
    if (
        not HAVE_TRITON
        or not routing_map.is_cuda
        or routing_map.dtype != torch.bool
        or routing_map.dim() != 2
        or not routing_map.is_contiguous()
    ):
        return routing_map.sum(dim=0)
    num_rows, num_experts = routing_map.shape
    out = torch.zeros(num_experts, dtype=torch.int64, device=routing_map.device)
    if num_rows == 0 or num_experts == 0:
        return out
    block_e = triton.next_power_of_2(num_experts)
    block_r = max(1, min(64, _MAX_TILE_ELEMENTS // block_e))
    _tokens_per_expert_kernel[(triton.cdiv(num_rows, block_r),)](
        routing_map.view(torch.uint8), out, num_rows, num_experts, BLOCK_R=block_r, BLOCK_E=block_e
    )
    return out
