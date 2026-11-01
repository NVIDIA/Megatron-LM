# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build global tensor layouts for DBuffer."""

import bisect
import itertools
import math
from collections.abc import Iterable

import torch

from .layout import GlobalLayout, Shape, non_leading_numel
from .placement import ShardLayout


def build_row_atomic_layout(
    shapes: Iterable[Shape], dp_size: int, *, block_size: int = 1, shard_layout: ShardLayout
) -> GlobalLayout:
    """Build a RowAtomic/BlockAtomic layout with the builder selected by ``shard_layout``."""
    if shard_layout is ShardLayout.PADDED:
        return build_for_row_atomic(shapes, dp_size=dp_size, block_size=block_size)
    if shard_layout is ShardLayout.UNPADDED:
        return build_for_row_atomic_unpadding(shapes, dp_size=dp_size, block_size=block_size)
    raise ValueError(f"Unsupported shard layout: {shard_layout!r}.")


def build_for_row_atomic(
    shapes: Iterable[Shape], dp_size: int, *, block_size: int = 1
) -> GlobalLayout:
    """Build equal-size rank segments for RowAtomic or BlockAtomic placements.

    This is a DBuffer-specific reimplementation of
    ``param_and_grad_buffer.build_data_parallel_buffer_index``. It keeps only
    the global offset construction and final padding so each rank-local shard
    size is a multiple of ``chunk_size``; DBuffer derives rank-local slices
    later through DTensor placements.

    ``chunk_size`` is the least common multiple of each tensor's row size
    (``shape[1:].numel()``), additionally multiplied by ``block_size`` for
    ``BlockAtomic``. For example, with shapes P0=(2, 6), P1=(4, 4),
    P2=(4, 4), P3=(1, 2), P4=(1, 6), ``chunk_size = LCM(6, 4, 4, 2, 6) = 12``
    and a 5-rank DP layout has equal-size rank shards:

    ```
    rank 0 [ 0, 12): | P0 row 0              | P0 row 1               |
    rank 1 [12, 24): | P1 row 0      | P1 row 1      | P1 row 2       |
    rank 2 [24, 36): | P1 row 3      | P3    | gap   | P2 row 0       |
    rank 3 [36, 48): | P2 row 1      | P2 row 2      | P2 row 3       |
    rank 4 [48, 60): | P4                    | pad                    |
    ```

    The diagram uses four character columns per element; segment widths are
    proportional. Every chunk boundary is aligned to each tensor's row size,
    so each DP shard owns full rows even when fragments fill regular-tensor
    padding gaps.

    Args:
        shapes: Logical tensor shapes in tensor-id order.
        dp_size: Data-parallel shard count for this global layout.
        block_size: Number of consecutive rows kept on one rank. Defaults to
            one for RowAtomic; larger values build a BlockAtomic layout.

    Returns:
        Global layout with row-aligned tensor offsets and a total size padded
        to a multiple of ``chunk_size * dp_size``, so every rank-local shard
        length is a multiple of ``chunk_size``.
    """

    if dp_size <= 0:
        raise ValueError(f"DP size must be positive, got {dp_size}.")
    tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
    chunk_size = math.lcm(1, *_row_atomic_alignments(tensor_shapes, block_size))

    # chunk_size is the packing granularity. Since every tensor row size divides it,
    # DP shard boundaries that are multiples of chunk_size avoid splitting dim-0 rows.
    UNASSIGNED_OFFSET = -1
    tensor_to_offset: list[int] = [UNASSIGNED_OFFSET] * len(tensor_shapes)
    fragment_items = []
    regular_items = []
    for tensor_id, shape in enumerate(tensor_shapes):
        if shape.numel() < chunk_size:
            fragment_items.append((tensor_id, shape))
        else:
            regular_items.append((tensor_id, shape))

    # Regular tensors anchor the layout. Fragments are held back to fill padding
    # gaps left by regular tensors whose sizes are not exact multiples of chunk_size.
    fragment_items.sort(key=lambda id_shape: id_shape[1].numel(), reverse=True)

    next_offset = 0
    while regular_items:
        tensor_id, shape = regular_items.pop(0)
        tensor_numel = shape.numel()
        tensor_to_offset[tensor_id] = next_offset

        if tensor_numel % chunk_size == 0:
            next_offset += tensor_numel
            continue

        gap_offset = next_offset + tensor_numel
        next_offset += _pad_to_multiple(tensor_numel, chunk_size)
        fragment_gap_end = next_offset
        remainder = tensor_numel % chunk_size

        # Try to pair this non-divisible regular tensor with a conjugate regular
        # tensor whose remainder fits in the same chunk_size interval. The
        # conjugate starts in the gap and then continues with full chunk_size
        # intervals after this one.
        conjugate_item = None
        for candidate_item in regular_items[:]:
            _, candidate_shape = candidate_item
            candidate_numel = candidate_shape.numel()
            candidate_remainder = candidate_numel % chunk_size
            if candidate_remainder == 0:
                continue
            if remainder + candidate_remainder <= chunk_size:
                conjugate_item = candidate_item
                regular_items.remove(candidate_item)
                break

        if conjugate_item is not None:
            conjugate_id, conjugate_shape = conjugate_item
            conjugate_numel = conjugate_shape.numel()
            conjugate_remainder = conjugate_numel % chunk_size
            conjugate_offset = next_offset - conjugate_remainder
            tensor_to_offset[conjugate_id] = conjugate_offset
            fragment_gap_end = conjugate_offset
            next_offset += (conjugate_numel // chunk_size) * chunk_size

        # Fill any remaining gap with fragments, keeping each fragment aligned to
        # its own row size so dim-0 rows remain contiguous within DP shards.
        for fragment in fragment_items[:]:
            frag_id, frag_shape = fragment
            frag_numel = frag_shape.numel()
            aligned_gap_offset = _pad_to_multiple(
                gap_offset, block_size * non_leading_numel(frag_shape)
            )
            if aligned_gap_offset + frag_numel > fragment_gap_end:
                continue
            tensor_to_offset[frag_id] = aligned_gap_offset
            gap_offset = aligned_gap_offset + frag_numel
            fragment_items.remove(fragment)

    # Fragments that did not fit into regular-tensor gaps are appended at the tail.
    for frag_id, frag_shape in fragment_items:
        next_offset = _pad_to_multiple(next_offset, block_size * non_leading_numel(frag_shape))
        tensor_to_offset[frag_id] = next_offset
        next_offset += frag_shape.numel()

    size = _pad_to_multiple(next_offset, chunk_size * dp_size)
    segment = size // dp_size
    layout = GlobalLayout(
        tensor_shapes=tensor_shapes,
        tensor_to_offset=tensor_to_offset,
        size=size,
        rank_to_offset=(segment * rank for rank in range(dp_size)),
    )
    layout.validate_for_row_atomic(block_size=block_size)
    return layout


def build_for_row_atomic_unpadding(
    shapes: Iterable[Shape], dp_size: int, *, block_size: int = 1
) -> GlobalLayout:
    """Build padding-free, generally unequal rank segments for RowAtomic or BlockAtomic.

    Here, tensors are packed back to back in tensor-id order, so ``size`` is the sum
    of all tensor numels. Each rank boundary is then placed as close as possible to
    an even ``size / dp_size`` split while landing on a row (or ``block_size``-row)
    boundary of the tensor it falls in, so every rank still owns whole rows or
    blocks. Segment sizes therefore differ by at most one row or block.

    For example, with shapes P0=(2, 6), P1=(4, 4), P2=(1, 2) on 2 ranks, the even
    split at 15 is rounded to P1's row boundary at 16:

    ```
    rank 0 [ 0, 16): | P0 row 0              | P0 row 1               | P1 row 0      |
    rank 1 [16, 30): | P1 row 1      | P1 row 2      | P1 row 3      | P2    |
    ```

    The padded layout would instead use ``chunk_size = LCM(6, 4, 2) = 12``, pad the
    total to a multiple of ``12 * 2``, and allocate 48 elements for 30 elements of data.

    Args:
        shapes: Logical tensor shapes in tensor-id order.
        dp_size: Data-parallel shard count for this global layout.
        block_size: Number of consecutive rows kept on one rank. Defaults to
            one for RowAtomic; larger values build a BlockAtomic layout.

    Returns:
        Global layout whose ``size`` is the total number of logical elements and
        whose rank boundaries never split a row or block.
    """
    if dp_size <= 0:
        raise ValueError(f"DP size must be positive, got {dp_size}.")
    tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
    alignments = _row_atomic_alignments(tensor_shapes, block_size)

    tensor_to_offset: list[int] = []
    # (start, end, alignment) of every nonempty tensor, in ascending start order.
    aligned_spans: list[tuple[int, int, int]] = []
    next_offset = 0
    for shape, alignment in zip(tensor_shapes, alignments):
        tensor_to_offset.append(next_offset)
        if shape.numel() > 0:
            aligned_spans.append((next_offset, next_offset + shape.numel(), alignment))
        next_offset += shape.numel()
    size = next_offset

    def nearest_aligned_boundary(target: int) -> int:
        # Snap ``target`` to the closest row/block boundary of the tensor containing it.
        span_index = bisect.bisect_right([span[0] for span in aligned_spans], target) - 1
        start, end, alignment = aligned_spans[span_index]
        lower = start + ((target - start) // alignment) * alignment
        upper = min(lower + alignment, end)
        return lower if target - lower <= upper - target else upper

    rank_to_offset = [0]
    if size > 0:
        for rank in range(1, dp_size):
            # Round the even split to the nearest element before snapping.
            target = (size * rank + dp_size // 2) // dp_size
            rank_to_offset.append(nearest_aligned_boundary(target))
    else:
        rank_to_offset.extend([0] * (dp_size - 1))

    layout = GlobalLayout(
        tensor_shapes=tensor_shapes,
        tensor_to_offset=tensor_to_offset,
        size=size,
        rank_to_offset=rank_to_offset,
    )
    layout.validate_for_row_atomic(block_size=block_size)
    return layout


def build_for_tensor_atomic(
    shapes: Iterable[Shape], dp_size: int, *, tensor_owners: Iterable[int]
) -> GlobalLayout:
    """Compute global tensor element offsets from a per-tensor owner-rank list.

    Pack tensors by ascending owner rank, preserving tensor-id order within each
    rank. Logical tensor IDs and shapes stay in their original order; only their
    offsets reflect the packing. Rank ``r``'s segment contains exactly its tensors.
    Tensors are never split across ranks. Because segments are packed tightly, there is
    no padding: ``size`` equals the sum of all tensor numels. Segment lengths are
    in general different per rank.

    For example, with shapes P0=(4, 4), P1=(2, 6), P2=(1, 2) and
    ``tensor_owners=(0, 1, 1)`` on 2 ranks:

    ```
    rank 0 [ 0, 16): | P0                      |
    rank 1 [16, 30): | P1             | P2     |
    ```

    Args:
        shapes: Logical tensor shapes in tensor-id order.
        dp_size: Data-parallel shard count for this global layout.
        tensor_owners: Owner rank of each tensor, in tensor-id order. Every rank
            must be an integer in ``[0, dp_size)``.

    Returns:
        Global layout whose per-rank segments are gap-free and whose ``size`` is
        the total number of logical elements.
    """
    if dp_size <= 0:
        raise ValueError(f"DP size must be positive, got {dp_size}.")
    tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
    tensor_owners = tuple(tensor_owners)
    if len(tensor_owners) != len(tensor_shapes):
        raise ValueError(
            "For TensorAtomic placement, the number of tensors "
            f"({len(tensor_shapes)}) must equal the number of tensor owners "
            f"({len(tensor_owners)})."
        )
    for rank in tensor_owners:
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < dp_size:
            raise ValueError(
                "In planning a layout for TensorAtomic placement, the owner of each tensor "
                f"must be an integer within the range of data parallel size [0, {dp_size}), "
                f"but got {rank!r}."
            )

    numels = [shape.numel() for shape in tensor_shapes]

    per_rank = [0] * dp_size
    for numel, owner in zip(numels, tensor_owners):
        per_rank[owner] += numel
    rank_to_offset = tuple(itertools.accumulate(per_rank[:-1], initial=0))
    next_offsets = list(rank_to_offset)
    tensor_to_offset = []
    for numel, owner in zip(numels, tensor_owners):
        tensor_to_offset.append(next_offsets[owner])
        next_offsets[owner] += numel

    layout = GlobalLayout(
        tensor_shapes=tensor_shapes,
        tensor_to_offset=tensor_to_offset,
        size=sum(numels),
        rank_to_offset=rank_to_offset,
    )
    layout.validate_for_tensor_atomic()
    return layout


def _row_atomic_alignments(
    tensor_shapes: tuple[torch.Size, ...], block_size: int
) -> tuple[int, ...]:
    """Return each tensor's row/block alignment, ``block_size * row_size``, in elements."""
    if block_size <= 0:
        raise ValueError(f"Block size must be positive, got {block_size}.")
    alignments = []
    for shape in tensor_shapes:
        row_size = non_leading_numel(shape)
        if row_size <= 0:
            raise ValueError(f"Cannot compute a layout for zero-sized non-leading dims: {shape}.")
        if shape[0] % block_size != 0:
            raise ValueError(
                f"Tensor dim 0 ({shape[0]}) must be divisible by block size {block_size}."
            )
        alignments.append(block_size * row_size)
    return tuple(alignments)


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
