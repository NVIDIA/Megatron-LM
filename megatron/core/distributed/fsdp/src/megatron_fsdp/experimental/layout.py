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

import bisect
import dataclasses
import itertools
from collections.abc import Iterable
from typing import TypeAlias

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.placement_types import Placement

from .range import Range

Shape: TypeAlias = torch.Size | Iterable[int]


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global tensor layout in element coordinates.

    Use :mod:`.layout_builder` to construct layouts for specific placements.

    Attributes:
        tensor_shapes: Logical tensor shapes in tensor-id order.
        tensor_to_offset: Global element offset of each logical tensor's first element.
        size: Total number of global elements, including any padding or gaps.
        rank_to_offset: Global element offset of each rank segment's first element
            (``dp_size`` entries). Segment ``k`` spans ``rank_to_offset[k]`` up to the
            next entry, or ``size`` for the last segment; see ``rank_size``. On a 1-D
            mesh ``k`` is the rank; on a multi-axis mesh ``get_rank_range`` maps the ranks
            of the Shard axes to ``k``. Segment sizes are equal (``size // dp_size``) for
            the padded ``RowAtomic`` / ``BlockAtomic`` layout of
            ``layout_builder.build_for_row_atomic``; generally uneven for
            ``build_for_row_atomic_unpadding`` and ``build_for_tensor_atomic``.
    """

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int
    rank_to_offset: tuple[int, ...]

    def __init__(
        self,
        tensor_shapes: Iterable[Shape],
        tensor_to_offset: Iterable[int],
        size: int,
        rank_to_offset: Iterable[int],
    ) -> None:
        """Normalize iterable inputs to immutable tuples and validate the layout."""
        object.__setattr__(
            self, "tensor_shapes", tuple(torch.Size(shape) for shape in tensor_shapes)
        )
        object.__setattr__(self, "tensor_to_offset", tuple(tensor_to_offset))
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "rank_to_offset", tuple(rank_to_offset))
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate offsets are in bounds and tensor ranges do not overlap."""

        if self.size < 0:
            raise AssertionError(f"Global layout size {self.size} is negative.")
        if len(self.tensor_shapes) != len(self.tensor_to_offset):
            raise AssertionError(
                "Global layout has mismatched tensor shapes and offsets: "
                f"{len(self.tensor_shapes)} shapes and {len(self.tensor_to_offset)} offsets."
            )
        offsets = self.rank_to_offset
        if not offsets:
            raise AssertionError("rank_to_offset needs at least one entry.")
        if offsets[0] != 0 or offsets[-1] > self.size:
            raise AssertionError(f"rank_to_offset must lie within [0, {self.size}], got {offsets}.")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise AssertionError(f"rank_to_offset must be non-decreasing, got {offsets}.")

        tensor_ranges = [
            self.get_tensor_range(tensor_id) for tensor_id in range(len(self.tensor_shapes))
        ]
        for tensor_id, tensor_range in enumerate(tensor_ranges):
            if tensor_range.start < 0:
                raise AssertionError(f"Tensor {tensor_id} offset {tensor_range.start} is negative.")
            if tensor_range.end > self.size:
                raise AssertionError(
                    f"Tensor {tensor_id} range [{tensor_range.start}, {tensor_range.end}) exceeds "
                    f"layout size {self.size}."
                )

        ordered_ranges = sorted(enumerate(tensor_ranges), key=lambda item: item[1].start)
        for (previous_id, previous_range), (current_id, current_range) in itertools.pairwise(
            ordered_ranges
        ):
            if current_range.start < previous_range.end:
                raise AssertionError(
                    "Global layout tensors overlap: "
                    f"tensor {previous_id} "
                    f"[{previous_range.start}, {previous_range.end}) and "
                    f"tensor {current_id} "
                    f"[{current_range.start}, {current_range.end})."
                )

    def validate_for_row_atomic(self, *, block_size: int = 1) -> None:
        """Check that no rank boundary splits a row (or ``block_size``-row block)."""
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}.")
        for tensor_id, (shape, start) in enumerate(
            zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
        ):
            row_size = non_leading_numel(shape)
            if row_size <= 0 or shape[0] % block_size != 0:
                raise ValueError(
                    f"Tensor {tensor_id} shape {shape} is incompatible with block size "
                    f"{block_size}."
                )
            alignment = block_size * row_size
            end = start + shape.numel()
            # Only boundaries strictly inside (start, end) can split this tensor.
            first = bisect.bisect_right(self.rank_to_offset, start)
            last = bisect.bisect_left(self.rank_to_offset, end)
            for offset in self.rank_to_offset[first:last]:
                if (offset - start) % alignment != 0:
                    raise ValueError(
                        f"Rank offsets {self.rank_to_offset} are incompatible with alignment "
                        f"{alignment} of tensor {tensor_id} {tuple(shape)} at offset {start}."
                    )

    def validate_for_tensor_atomic(self) -> None:
        """Check every nonempty tensor is contained in a single rank segment."""
        for tensor_id, (shape, start) in enumerate(
            zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
        ):
            if shape.numel() == 0:
                continue
            end = start + shape.numel()
            # (The last rank whose offset <= start) == (the first rank whose offset > start) - 1
            rank = bisect.bisect_right(self.rank_to_offset, start) - 1
            if end > self.rank_to_offset[rank] + self.rank_size(rank):
                raise ValueError(
                    f"Tensor {tensor_id} [{start}, {end}) crosses a rank segment boundary "
                    "and is incompatible with TensorAtomic."
                )

    @property
    def dp_size(self) -> int:
        """Number of segments (data-parallel shards) this layout was planned for."""
        return len(self.rank_to_offset)

    def rank_size(self, rank: int) -> int:
        """Number of elements in segment ``rank``."""
        end = self.rank_to_offset[rank + 1] if rank + 1 < self.dp_size else self.size
        return end - self.rank_to_offset[rank]

    @property
    def has_equal_shard_sizes(self) -> bool:
        """Whether every rank's segment has the same numel."""
        return len({self.rank_size(rank) for rank in range(self.dp_size)}) == 1

    def get_rank_range(self, mesh: DeviceMesh, placements: Iterable[Placement], rank: int) -> Range:
        """Return the global buffer element range for a global process rank.

        Args:
            mesh: The device mesh defining the distribution.
            placements: Per-mesh-axis placements.
            rank: Global process rank belonging to ``mesh``.
        """
        coordinates = (mesh.mesh == rank).nonzero(as_tuple=False)
        if coordinates.size(0) == 0:
            raise ValueError(f"Global rank {rank} is not in the device mesh.")
        coordinate = coordinates[0].tolist()
        placements = tuple(placements)
        # Innermost Shard axis is the most significant digit of the shard index.
        shard_index = 0
        num_shards = 1
        shard_axes: list[int] = []
        for axis in reversed(range(len(placements))):
            if not isinstance(placements[axis], Shard):
                continue
            shard_axes.append(axis)
            shard_index = shard_index * mesh.size(axis) + coordinate[axis]
            num_shards *= mesh.size(axis)
        if self.dp_size % num_shards != 0:
            raise ValueError(
                f"Layout was built for {self.dp_size} shards, which cannot be split evenly "
                f"over {num_shards} shard ranks on mesh axes {sorted(shard_axes)}."
            )
        segments_per_rank = self.dp_size // num_shards
        first = shard_index * segments_per_rank
        last = first + segments_per_rank - 1
        start = self.rank_to_offset[first]
        return Range(start, self.rank_to_offset[last] + self.rank_size(last) - start)

    def get_local_range(self, mesh: DeviceMesh, placements: Iterable[Placement]) -> Range:
        """Return this rank's element range in the global buffer for ``placements``."""
        return self.get_rank_range(mesh, placements, mesh.get_rank())

    def get_tensor_range(self, tensor_index: int) -> Range:
        """Return the full tensor's element range in the global buffer."""
        return Range(self.tensor_to_offset[tensor_index], self.tensor_shapes[tensor_index].numel())


def non_leading_numel(shape: torch.Size) -> int:
    """Return the number of elements after dim 0 for a non-scalar shape."""
    if len(shape) == 0:
        raise ValueError(f"DBuffer layout does not support 0D tensor shapes: {shape}.")
    return shape[1:].numel()
