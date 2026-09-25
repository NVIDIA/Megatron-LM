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

from .placement import BlockAtomic, PlacementReference, RowAtomic, TensorAtomic

Shape: TypeAlias = torch.Size | Iterable[int]


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global tensor layout in element coordinates.

    Attributes:
        tensor_shapes: Logical tensor shapes in tensor-id order.
        tensor_to_offset: Global element offset of each logical tensor's first element.
        size: Total number of global elements, including any padding or gaps.
        rank_segment_offsets: Boundaries of the ``dp_size`` contiguous segments this layout
            was planned for, in global element coordinates (``dp_size + 1`` entries). Segment
            ``k`` is ``[rank_segment_offsets[k], rank_segment_offsets[k + 1])``. On a 1-D
            mesh ``k`` is the rank; on a multi-axis mesh ``get_local_range`` maps the ranks
            of the Shard axes to ``k``. Uniform (``size // dp_size``) for ``RowAtomic`` /
            ``BlockAtomic``; generally uneven for ``TensorAtomic``.
        reference: The Shard placement this layout was planned against. ``RowAtomic`` and
            ``BlockAtomic`` produce a row-/block-aligned layout padded to equal-size
            per-rank shards; ``TensorAtomic`` produces a gap-free layout in which each
            rank's contiguous segment holds exactly the tensors it owns.
    """

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int
    rank_segment_offsets: tuple[int, ...]
    reference: PlacementReference = dataclasses.field(default_factory=RowAtomic)

    @classmethod
    def build(
        cls,
        shapes: Iterable[Shape],
        dp_size: int,
        *,
        reference: PlacementReference = RowAtomic(),
        tensor_owners: tuple[int, ...] | None = None,
    ) -> "GlobalLayout":
        """Plan a global layout for ``shapes`` under the given reference placement.

        Dispatches to ``_build_for_tensor_atomic`` for ``TensorAtomic`` and to
        ``_build_for_row_aligned`` for ``RowAtomic`` / ``BlockAtomic``.

        Args:
            shapes: Logical tensor shapes in tensor-id order.
            dp_size: Data-parallel shard count for this global layout.
            reference: Shard placement the layout must be compatible with.
            tensor_owners: Owner rank of each tensor, in tensor-id order. Required for
            ``TensorAtomic`` and must be non-decreasing. ``None`` for
            ``RowAtomic`` / ``BlockAtomic``.

        Returns:
            A validated ``GlobalLayout``.
        """
        if dp_size <= 0:
            raise ValueError(f"DP size must be positive, got {dp_size}.")
        if not isinstance(reference, (RowAtomic, BlockAtomic, TensorAtomic)):
            raise ValueError(
                "'reference' type should be chosen from (RowAtomic, BlockAtomic, TensorAtomic), "
                f"but got {type(reference)}."
            )

        tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
        if isinstance(reference, TensorAtomic):
            if tensor_owners is None:
                raise ValueError("TensorAtomic reference placement requires 'tensor_owners'.")
            tensor_owners = tuple(tensor_owners)
            if len(tensor_owners) != len(tensor_shapes):
                raise ValueError(
                    "For TensorAtomic reference placement, the number of tensors "
                    f"({len(tensor_shapes)}) must equal the number of tensor owners "
                    f"({len(tensor_owners)})."
                )
            return cls._build_for_tensor_atomic(
                tensor_shapes, dp_size=dp_size, tensor_owners=tensor_owners
            )
        if tensor_owners is not None:
            raise ValueError(
                "'tensor_owners' is only supported with a TensorAtomic reference placement, "
                f"got {tensor_owners!r} with {reference!r}."
            )
        return cls._build_for_row_aligned(tensor_shapes, dp_size=dp_size, reference=reference)

    @classmethod
    def _build_for_row_aligned(
        cls,
        tensor_shapes: Iterable[Shape],
        dp_size: int,
        *,
        reference: PlacementReference = RowAtomic(),
    ) -> "GlobalLayout":
        """Compute global tensor element offsets and padded size.

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
            tensor_shapes: Logical tensor shapes in tensor-id order.
            dp_size: Data-parallel shard count for this global layout.
            reference: ``RowAtomic`` or ``BlockAtomic``. For ``BlockAtomic`` its
                ``block_size`` rows are kept together on one rank.

        Returns:
            Global layout with row-aligned tensor offsets and a total size padded
            to a multiple of ``chunk_size * dp_size``, so every rank-local shard
            length is a multiple of ``chunk_size``.
        """

        block_size = reference.block_size if isinstance(reference, BlockAtomic) else 1
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}.")

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
        return cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=size,
            rank_segment_offsets=tuple(segment * rank for rank in range(dp_size + 1)),
            reference=reference,
        )

    @classmethod
    def _build_for_tensor_atomic(
        cls, tensor_shapes: Iterable[Shape], dp_size: int, *, tensor_owners: tuple[int, ...]
    ) -> "GlobalLayout":
        """Compute global tensor element offsets from a per-tensor owner-rank list.

        Tensors are expected to arrive already grouped by owner rank in ascending
        rank order, so the layout is simply the tensors packed back to back.
        Rank ``r``'s segment is the contiguous subrange of the list of tensors owned by ``r``.
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
            tensor_shapes: Logical tensor shapes in tensor-id order.
            dp_size: Data-parallel shard count for this global layout.
            tensor_owners: Owner rank of each tensor, in tensor-id order. Every rank
                must be in ``[0, dp_size)`` and the sequence must be non-decreasing.

        Returns:
            Global layout whose per-rank segments are gap-free and whose ``size`` is
            the total number of logical elements.
        """
        out_of_range = [rank for rank in tensor_owners if not 0 <= rank < dp_size]
        if out_of_range:
            raise ValueError(
                "In planning a layout for TensorAtomic placement, the owner of each tensor "
                f"must be within the range of data parallel size [0, {dp_size}), "
                f"but got {out_of_range}."
            )
        if any(a > b for a, b in zip(tensor_owners, tensor_owners[1:])):
            raise ValueError(f"'tensor_owners' must be non-decreasing, got {tensor_owners}.")

        numels = [shape.numel() for shape in tensor_shapes]
        tensor_to_offset = tuple(itertools.accumulate(numels, initial=0))[:-1]

        per_rank = [0] * dp_size
        for numel, owner in zip(numels, tensor_owners):
            per_rank[owner] += numel
        rank_segment_offsets = tuple(itertools.accumulate(per_rank, initial=0))

        return cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tensor_to_offset,
            size=sum(numels),
            rank_segment_offsets=rank_segment_offsets,
            reference=TensorAtomic(),
        )

    def __post_init__(self) -> None:
        """Validate tensor offsets are in bounds, non-overlapping, and reference-compatible."""

        @dataclasses.dataclass(frozen=True)
        class TensorRange:
            """Contiguous global element range occupied by one logical tensor."""

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
        offsets = self.rank_segment_offsets
        if len(offsets) < 2:
            raise AssertionError(
                f"'rank_segment_offsets' needs at least 2 entries, got {len(offsets)}."
            )
        if offsets[0] != 0 or offsets[-1] != self.size:
            raise AssertionError(f"rank_segment_offsets must span [0, {self.size}), got {offsets}.")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise AssertionError(f"rank_segment_offsets must be non-decreasing, got {offsets}.")

        if isinstance(self.reference, TensorAtomic):
            # Every tensor must sit entirely inside one rank's segment.
            for tensor_id, (shape, start) in enumerate(
                zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
            ):
                end = start + shape.numel()
                # bisect_right skips empty segments that end at ``start``.
                rank = bisect.bisect_right(offsets, start) - 1
                if rank < 0 or rank >= self.dp_size or end > offsets[rank + 1]:
                    raise AssertionError(
                        f"Tensor {tensor_id} [{start}, {end}) is not contained in a single "
                        f"rank segment of {offsets}."
                    )

        tensor_ranges: list[TensorRange] = []
        for tensor_id, (shape, start) in enumerate(
            zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
        ):
            if start < 0:
                raise AssertionError(f"Tensor {tensor_id} offset {start} is negative.")

            if not isinstance(self.reference, TensorAtomic):
                # RowAtomic / BlockAtomic shards may cut through a tensor, so every tensor must
                # start on a row (and block) boundary to keep dim-0 rows intact per rank.
                # TensorAtomic never splits a tensor, so no alignment is required.
                row_size = non_leading_numel(shape)
                if row_size <= 0:
                    raise AssertionError(f"Tensor {tensor_id} has invalid row size {row_size}.")
                if start % (self.block_size * row_size) != 0:
                    raise AssertionError(
                        f"Tensor {tensor_id} offset {start} is not aligned to block size "
                        f"{self.block_size * row_size}."
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

    @property
    def block_size(self) -> int:
        """Number of dim-0 rows kept together per rank; 1 unless the reference is BlockAtomic."""
        return self.reference.block_size if isinstance(self.reference, BlockAtomic) else 1

    @property
    def dp_size(self) -> int:
        """Number of segments (data-parallel shards) this layout was planned for."""
        return len(self.rank_segment_offsets) - 1

    @property
    def rank_segments(self) -> tuple[tuple[int, int], ...]:
        """``(start, numel)`` of each of the ``dp_size`` segments."""
        offsets = self.rank_segment_offsets
        return tuple(
            (offsets[rank], offsets[rank + 1] - offsets[rank]) for rank in range(self.dp_size)
        )

    @property
    def is_uniform(self) -> bool:
        """Whether every rank's segment has the same numel."""
        return len({numel for _, numel in self.rank_segments}) == 1

    def get_local_range(self, mesh: DeviceMesh, placements: Iterable[Placement]) -> tuple[int, int]:
        """Return this rank's local element ``(offset, numel)`` for ``placements``."""
        placements = tuple(placements)
        shard_axes = [
            axis for axis, placement in enumerate(placements) if isinstance(placement, Shard)
        ]
        if not shard_axes:
            return 0, self.size

        innermost_placement = placements[shard_axes[-1]]
        if isinstance(innermost_placement, TensorAtomic) != isinstance(
            self.reference, TensorAtomic
        ):
            raise ValueError(
                f"Placement {innermost_placement!r} is incompatible with layout reference "
                f"{self.reference!r}."
            )

        shard_index = 0
        num_shards = 1
        for axis in reversed(shard_axes):
            shard_index = shard_index * mesh.size(axis) + mesh.get_local_rank(axis)
            num_shards *= mesh.size(axis)
        if self.dp_size % num_shards != 0:
            raise ValueError(
                f"Layout was built for {self.dp_size} shards, which cannot be split evenly "
                f"over {num_shards} shard ranks on mesh axes {shard_axes}."
            )
        segments_per_rank = self.dp_size // num_shards
        start = self.rank_segment_offsets[shard_index * segments_per_rank]
        end = self.rank_segment_offsets[(shard_index + 1) * segments_per_rank]
        return start, end - start


def non_leading_numel(shape: torch.Size) -> int:
    """Return the number of elements after dim 0 for a non-scalar shape."""
    if len(shape) == 0:
        raise ValueError(f"DBuffer layout does not support 0D tensor shapes: {shape}.")
    return shape[1:].numel()


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
