# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pinned host PLE row lookup adapted from ModelScope mcore-bridge 7baa28c."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _gather_rows(
        weight_ptr, ids_ptr, output_ptr, width, row_start, row_end, BLOCK_D: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        global_id = tl.load(ids_ptr + row)
        owned = (global_id >= row_start) & (global_id < row_end)
        local_id = tl.where(owned, global_id - row_start, 0)
        col = tl.arange(0, BLOCK_D)
        ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(ptr + local_id * width + col, mask=col < width, other=0.0)
        tl.store(output_ptr + row * width + col, tl.where(owned, values, 0.0), mask=col < width)


def gather_pinned_rows(table: torch.Tensor, ids: torch.Tensor, row_start: int, row_end: int):
    """Return a GPU tensor, or None when the supported Triton path is unavailable."""
    if triton is None or ids.device.type != "cuda" or table.dtype != torch.bfloat16:
        return None
    width = table.shape[1]
    output = torch.empty((*ids.shape, width), dtype=table.dtype, device=ids.device)
    flat = ids.contiguous().reshape(-1)
    if flat.numel():
        _gather_rows[(flat.numel(),)](
            table.data_ptr(),
            flat,
            output.view(-1, width),
            width,
            row_start,
            row_end,
            BLOCK_D=triton.next_power_of_2(width),
        )
    return output
