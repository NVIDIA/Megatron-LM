# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Row-order kernels for indexer top-k outputs.

Top-k selectors write the ids of a row in slots chosen by atomics, so the slot order is not
reproducible even when the selected set is. Sparse attention accumulates in slot order, which
makes a deterministic order a precondition for bitwise-reproducible layer outputs:

* :func:`sort_topk_rows_` sorts every row in place, ids ascending and invalid entries last;
* :func:`compact_valid_topk_` packs the valid ids of a raw selector output to the front of each
  row in their original order (a stable compaction, not a sort) and counts them.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments use the torch implementation.
    triton = None
    tl = None

__all__ = ["compact_valid_topk_", "sort_topk_rows_"]

_INT32_MAX = torch.iinfo(torch.int32).max
# The compaction kernel holds one output row per program in registers.
_COMPACT_KERNEL_MAX_WIDTH = 2048


def sort_topk_rows_(indices: Tensor, *, rows_per_chunk: int = 32768) -> Tensor:
    """Sort every row of an int32 top-k index matrix in place, invalid entries last.

    Negative entries are invalid: they are moved behind the valid ids and written back as -1.
    Valid ids must be below ``2**31 - 1``, the sort key used for invalid entries.

    ``torch.sort`` materializes an int64 permutation next to the sorted values, so the rows are
    sorted ``rows_per_chunk`` at a time, bounding the transient memory to about 13 bytes per
    element of one chunk. The result does not depend on ``rows_per_chunk``. If an error
    interrupts the call, the rows of the interrupted chunk are left unspecified.

    Args:
        indices: int32 ``[rows, k]`` top-k ids, sorted in place. Any strides are accepted.
        rows_per_chunk: Number of rows sorted per ``torch.sort`` call.

    Returns:
        ``indices``.

    Raises:
        ValueError: If ``indices`` is not a 2-D int32 tensor or ``rows_per_chunk`` is not a
            positive integer.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError(
            f"expected int32 [rows, k] indices, got {indices.dtype} {tuple(indices.shape)}"
        )
    if type(rows_per_chunk) is not int or rows_per_chunk <= 0:
        raise ValueError(f"rows_per_chunk must be a positive integer, got {rows_per_chunk!r}")
    for start in range(0, indices.shape[0], rows_per_chunk):
        block = indices[start : start + rows_per_chunk]
        block.masked_fill_(block < 0, _INT32_MAX)
        block.copy_(block.sort(dim=-1).values)
        block.masked_fill_(block == _INT32_MAX, -1)
    return indices


if triton is not None:

    @triton.jit
    def _compact_valid_topk_kernel(
        selected_ptr,
        ends_ptr,
        out_ptr,
        lengths_ptr,
        selected_row_stride,
        out_row_stride,
        INPUT_WIDTH: tl.constexpr,
        OUTPUT_WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_LENGTHS: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        selected = tl.load(
            selected_ptr + row.to(tl.int64) * selected_row_stride + columns,
            columns < INPUT_WIDTH,
            -1,
        )
        end = tl.load(ends_ptr + row)
        valid = (columns < INPUT_WIDTH) & (selected >= 0) & (selected < end)
        count = tl.sum(valid.to(tl.int32), axis=0)
        destinations = tl.cumsum(valid.to(tl.int32), axis=0) - 1
        # The two stores have disjoint destinations: the valid prefix [0, count) and the -1 tail
        # [count, OUTPUT_WIDTH). Filling the whole row before the scatter would need a barrier.
        out_row = out_ptr + row.to(tl.int64) * out_row_stride
        tl.store(out_row + columns, -1, (columns >= count) & (columns < OUTPUT_WIDTH))
        tl.store(out_row + destinations, selected, valid)
        if HAS_LENGTHS:
            tl.store(lengths_ptr + row, count)


def _compact_valid_topk_torch(
    selected: Tensor, ends: Tensor, out: Tensor, lengths: Tensor | None
) -> None:
    width = selected.shape[1]
    valid = (selected >= 0) & (selected < ends.unsqueeze(-1))
    counts = valid.sum(dim=-1, dtype=torch.int32)
    if width:
        # A stable sort on "is invalid" moves the valid entries to the front in their order.
        order = torch.sort((~valid).to(torch.uint8), dim=-1, stable=True).indices
        packed = torch.gather(selected, 1, order)
        slots = torch.arange(width, device=selected.device)
        out[:, :width].copy_(packed.masked_fill_(slots >= counts.unsqueeze(-1), -1))
    out[:, width:].fill_(-1)
    if lengths is not None:
        lengths.copy_(counts)


def compact_valid_topk_(
    selected: Tensor, ends: Tensor, out: Tensor, lengths: Tensor | None = None
) -> None:
    """Pack the valid ids of each row to the front of ``out`` and fill the rest with -1.

    Entry ``selected[r, j]`` is valid when ``0 <= selected[r, j] < ends[r]``. The valid entries of
    row ``r`` are written to ``out[r, :count]`` in their original order, ``out[r, count:]`` is
    set to -1, and ``lengths[r] = count`` when ``lengths`` is given. The ids are not sorted.

    CUDA tensors use one Triton program per row when ``out`` has at most 2048 columns,
    ``selected`` and ``out`` have unit stride in the last dimension and ``ends`` and ``lengths``
    are contiguous; other inputs use a torch implementation with the same result.

    Args:
        selected: int32 ``[rows, width]`` raw selector output; holes may hold -1 or out-of-range
            ids.
        ends: int32 ``[rows]`` exclusive upper bound of the valid ids of each row.
        out: int32 ``[rows, out_width]`` destination with ``out_width >= width``, written in place;
            it may be a row-strided view of a larger buffer but must not overlap ``selected``.
        lengths: Optional int32 ``[rows]`` destination of the valid counts.

    Raises:
        ValueError: If the dtypes, shapes or devices are inconsistent.
    """
    tensors = (selected, ends, out) if lengths is None else (selected, ends, out, lengths)
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise ValueError("selected, ends, out and lengths must be int32 tensors")
    if any(tensor.device != selected.device for tensor in tensors):
        raise ValueError("selected, ends, out and lengths must be on the same device")
    if selected.ndim != 2 or out.ndim != 2:
        raise ValueError(
            f"expected 2-D selected and out, got {tuple(selected.shape)} and {tuple(out.shape)}"
        )
    rows, width = selected.shape
    if out.shape[0] != rows or out.shape[1] < width:
        raise ValueError(
            f"out {tuple(out.shape)} must have {rows} rows and at least {width} columns"
        )
    if ends.shape != (rows,) or (lengths is not None and lengths.shape != (rows,)):
        raise ValueError(f"ends and lengths must have shape ({rows},)")
    if rows == 0:
        return
    use_kernel = (
        triton is not None
        and selected.is_cuda
        and 0 < out.shape[1] <= _COMPACT_KERNEL_MAX_WIDTH
        and selected.stride(1) == 1
        and out.stride(1) == 1
        and ends.is_contiguous()
        and (lengths is None or lengths.is_contiguous())
    )
    if not use_kernel:
        _compact_valid_topk_torch(selected, ends, out, lengths)
        return
    with torch.cuda.device(selected.device):
        _compact_valid_topk_kernel[(rows,)](
            selected,
            ends,
            out,
            lengths if lengths is not None else ends,
            selected.stride(0),
            out.stride(0),
            INPUT_WIDTH=width,
            OUTPUT_WIDTH=out.shape[1],
            BLOCK=triton.next_power_of_2(out.shape[1]),
            HAS_LENGTHS=lengths is not None,
            num_warps=4,
        )
