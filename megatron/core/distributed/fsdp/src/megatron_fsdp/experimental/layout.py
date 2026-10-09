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
import math
from collections.abc import Iterable
from typing import TypeAlias

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.placement_types import Placement

Shape: TypeAlias = torch.Size | Iterable[int]


@dataclasses.dataclass(frozen=True)
class TensorRange:
    """Contiguous global element range occupied by one logical tensor."""

    start: int
    end: int
    tensor_id: int


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global tensor layout in element coordinates.

    Attributes:
        tensor_shapes: Logical tensor shapes in tensor-id order.
        tensor_to_offset: Global element offset of each logical tensor's first element.
        size: Total number of global elements, including any padding or gaps.
        rank_to_offset: Global element offset of each rank segment's first element
            (``dp_size`` entries). Segment ``k`` spans ``rank_to_offset[k]`` up to the
            next entry, or ``size`` for the last segment; see ``rank_size``. On a 1-D
            mesh ``k`` is the rank; on a multi-axis mesh ``get_local_range`` maps the ranks
            of the Shard axes to ``k``. Segment sizes are equal (``size // dp_size``) for
            ``RowAtomic`` / ``BlockAtomic``; generally uneven for ``TensorAtomic``.
    """

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int
    rank_to_offset: tuple[int, ...]

    @classmethod
    def build_for_row_atomic(
        cls, shapes: Iterable[Shape], dp_size: int, *, block_size: int = 1
    ) -> "GlobalLayout":
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
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}.")

        tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
        chunk_size = 1
        for shape in tensor_shapes:
            row_size = non_leading_numel(shape)
            if row_size <= 0:
                raise ValueError(
                    f"Cannot compute a layout for zero-sized non-leading dims: {shape}."
                )
            if shape[0] % block_size != 0:
                raise ValueError(
                    f"Tensor dim 0 ({shape[0]}) must be divisible by block size {block_size}."
                )
            chunk_size = math.lcm(chunk_size, block_size * row_size)

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
        layout = cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=size,
            rank_to_offset=tuple(segment * rank for rank in range(dp_size)),
        )
        layout.validate_for_row_atomic(block_size=block_size)
        return layout

    @classmethod
    def build_for_tensor_atomic(
        cls, shapes: Iterable[Shape], dp_size: int, *, tensor_owners: Iterable[int]
    ) -> "GlobalLayout":
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

        layout = cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=sum(numels),
            rank_to_offset=rank_to_offset,
        )
        layout.validate_for_tensor_atomic()
        return layout

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

        tensor_ranges: list[TensorRange] = []
        for tensor_id, (shape, start) in enumerate(
            zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
        ):
            if start < 0:
                raise AssertionError(f"Tensor {tensor_id} offset {start} is negative.")

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

    def validate_for_row_atomic(self, *, block_size: int = 1) -> None:
        """Check equal-size shards and row/block alignment from the actual offsets."""
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}.")
        if not self.has_equal_shard_sizes:
            raise ValueError(
                "Unequal rank segment sizes are incompatible with RowAtomic/BlockAtomic."
            )
        chunk_size = 1
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
            if start % alignment != 0:
                raise ValueError(
                    f"Tensor {tensor_id} offset {start} is incompatible with alignment {alignment}."
                )
            chunk_size = math.lcm(chunk_size, alignment)
        if any(offset % chunk_size for offset in self.rank_to_offset):
            raise ValueError(
                f"Rank offsets {self.rank_to_offset} are incompatible with alignment {chunk_size}."
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

    def get_local_range(self, mesh: DeviceMesh, placements: Iterable[Placement]) -> tuple[int, int]:
        """Return this rank's local element ``(offset, numel)`` for ``placements``."""
        placements = tuple(placements)
        # Innermost Shard axis is the most significant digit of the shard index.
        shard_index = 0
        num_shards = 1
        shard_axes: list[int] = []
        for axis in reversed(range(len(placements))):
            if not isinstance(placements[axis], Shard):
                continue
            shard_axes.append(axis)
            shard_index = shard_index * mesh.size(axis) + mesh.get_local_rank(axis)
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
        return start, self.rank_to_offset[last] + self.rank_size(last) - start


def non_leading_numel(shape: torch.Size) -> int:
    """Return the number of elements after dim 0 for a non-scalar shape."""
    if len(shape) == 0:
        raise ValueError(f"DBuffer layout does not support 0D tensor shapes: {shape}.")
    return shape[1:].numel()


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
