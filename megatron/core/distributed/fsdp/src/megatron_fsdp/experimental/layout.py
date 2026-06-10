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

"""Global tensor layout metadata for DBuffer."""

import dataclasses
import math
from collections.abc import Iterable
from typing import TypeAlias

import torch
from torch.distributed import DeviceMesh

from .placement import Flat, Placement

Shape: TypeAlias = torch.Size | Iterable[int]


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global tensor layout in element coordinates."""

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int

    @classmethod
    def build(cls, shapes: Iterable[Shape], dp_size: int) -> "GlobalLayout":
        """Compute global tensor element offsets and padded size.

        This is a DBuffer-specific reimplementation of
        ``param_and_grad_buffer.build_data_parallel_buffer_index``. It keeps only
        the global offset construction and final padding so each rank-local shard
        size is a multiple of ``chunk_size``; DBuffer derives rank-local slices
        later through DTensor placements.

        The computed layout is compatible with Flat, TensorAtomic, and BlockAtomic,
        even though the latter two are not implemented.

        ``chunk_size`` is the least common multiple of each tensor's row size
        (``shape[1:].numel()``). For example, with shapes P0=(2, 6), P1=(4, 4),
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

        Returns:
            Global layout with row-aligned tensor offsets and a total size padded
            to a multiple of ``chunk_size * dp_size``, so every rank-local shard
            length is a multiple of ``chunk_size``.
        """
        if dp_size <= 0:
            raise ValueError(f"DP size must be positive, got {dp_size}.")

        tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
        chunk_size = 1
        for shape in tensor_shapes:
            row_size = non_leading_numel(shape)
            if row_size <= 0:
                raise ValueError(
                    f"Cannot compute a layout for zero-sized non-leading dims: {shape}."
                )
            chunk_size = math.lcm(chunk_size, row_size)

        # chunk_size is the packing unit. Since every tensor row size divides it,
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
                aligned_gap_offset = _pad_to_multiple(gap_offset, non_leading_numel(frag_shape))
                if aligned_gap_offset + frag_numel > fragment_gap_end:
                    continue
                tensor_to_offset[frag_id] = aligned_gap_offset
                gap_offset = aligned_gap_offset + frag_numel
                fragment_items.remove(fragment)

        # Fragments that did not fit into regular-tensor gaps are appended at the tail.
        for frag_id, frag_shape in fragment_items:
            next_offset = _pad_to_multiple(next_offset, non_leading_numel(frag_shape))
            tensor_to_offset[frag_id] = next_offset
            next_offset += frag_shape.numel()

        return cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=_pad_to_multiple(next_offset, chunk_size * dp_size),
        )

    def __post_init__(self) -> None:
        """Validate tensor offsets are row-aligned, in bounds, and non-overlapping."""

        @dataclasses.dataclass(frozen=True)
        class TensorRange:
            start: int
            end: int
            tensor_id: int

        if self.size < 0:
            raise AssertionError(f"Global layout size {self.size} is negative.")
        if len(self.tensor_shapes) != len(self.tensor_to_offset):
            raise AssertionError(
                "Global layout has mismatched tensor shapes and offsets: "
                f"{len(self.tensor_shapes)} shapes and {len(self.tensor_to_offset)} offsets."
            )

        tensor_ranges: list[TensorRange] = []
        for tensor_id, (shape, start) in enumerate(
            zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
        ):
            if start < 0:
                raise AssertionError(f"Tensor {tensor_id} offset {start} is negative.")

            row_size = non_leading_numel(shape)
            if row_size <= 0:
                raise AssertionError(f"Tensor {tensor_id} has invalid row size {row_size}.")
            if start % row_size != 0:
                raise AssertionError(
                    f"Tensor {tensor_id} offset {start} is not aligned to row size {row_size}."
                )

            end = start + shape.numel()
            if end > self.size:
                raise AssertionError(
                    f"Tensor {tensor_id} range [{start}, {end}) exceeds "
                    f"layout size {self.size}."
                )
            tensor_ranges.append(TensorRange(start, end, tensor_id))

        previous_range: TensorRange | None = None
        for current_range in sorted(tensor_ranges, key=lambda tensor_range: tensor_range.start):
            if previous_range is not None and current_range.start < previous_range.end:
                raise AssertionError(
                    "Global layout tensors overlap: "
                    f"tensor {previous_range.tensor_id} "
                    f"[{previous_range.start}, {previous_range.end}) and "
                    f"tensor {current_range.tensor_id} "
                    f"[{current_range.start}, {current_range.end})."
                )
            previous_range = current_range

    def get_local_range(self, mesh: DeviceMesh, placements: Iterable[Placement]) -> tuple[int, int]:
        """Return this rank's local element offset and length for ``placements``."""
        offset = 0
        numel = self.size
        for axis, placement in reversed(tuple(enumerate(placements))):
            if not isinstance(placement, Flat):
                continue
            axis_size = mesh.size(axis)
            if numel % axis_size != 0:
                raise ValueError(
                    f"Local range size {numel} is not divisible by Flat axis size {axis_size}."
                )
            shard_size = numel // axis_size
            offset += mesh.get_local_rank(axis) * shard_size
            numel = shard_size
        return offset, numel


def non_leading_numel(shape: torch.Size) -> int:
    """Return the number of elements after dim 0 for a non-scalar shape."""
    if len(shape) == 0:
        raise ValueError(f"DBuffer layout does not support 0D tensor shapes: {shape}.")
    return shape[1:].numel()


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
