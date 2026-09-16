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
import itertools
import math
from collections.abc import Iterable
from typing import TypeAlias

import torch

from .placement import BlockAtomic, Flat, PlacementReference, TensorAtomic

Shape: TypeAlias = torch.Size | Iterable[int]


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global tensor layout in element coordinates.

    Attributes:
        tensor_shapes: Logical tensor shapes in tensor-id order.
        tensor_to_offset: Global element offset of each logical tensor's first element.
        size: Total number of global elements, including any padding or gaps.
        reference: The Shard placement this layout was planned against. ``Flat`` and
            ``BlockAtomic`` produce a row-/block-aligned layout padded to equal-size
            per-rank shards; ``TensorAtomic`` produces a gap-free layout in which each
            rank's contiguous segment holds exactly the tensors it owns.
    """

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int
    reference: PlacementReference = Flat()

    @classmethod
    def build(
        cls, shapes: Iterable[Shape], dp_size: int, *, reference: PlacementReference = Flat()
    ) -> "GlobalLayout":
        """Plan a global layout for ``shapes`` under the given reference placement.

        Dispatches to ``_build_for_tensor_atomic`` for ``TensorAtomic`` and to
        ``_build_for_row_aligned`` for ``Flat`` / ``BlockAtomic``.

        Args:
            shapes: Logical tensor shapes in tensor-id order.
            dp_size: Data-parallel shard count for this global layout.
            reference: Shard placement the layout must be compatible with.

        Returns:
            A validated ``GlobalLayout``.
        """
        if dp_size <= 0:
            raise ValueError(f"DP size must be positive, got {dp_size}.")
        if not isinstance(reference, (Flat, BlockAtomic, TensorAtomic)):
            raise ValueError(
                "'reference' type should be chosen from (Flat, BlockAtomic, TensorAtomic), "
                f"but got {type(reference)}."
            )

        tensor_shapes = tuple(torch.Size(shape) for shape in shapes)
        if isinstance(reference, TensorAtomic):
            return cls._build_for_tensor_atomic(tensor_shapes, dp_size=dp_size, reference=reference)
        return cls._build_for_row_aligned(tensor_shapes, dp_size=dp_size, reference=reference)

    @classmethod
    def _build_for_row_aligned(
        cls, tensor_shapes: Iterable[Shape], dp_size: int, *, reference: PlacementReference = Flat()
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
            reference: ``Flat`` or ``BlockAtomic``. For ``BlockAtomic`` its
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

        return cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=_pad_to_multiple(next_offset, chunk_size * dp_size),
            reference=reference,
        )

    @classmethod
    def _build_for_tensor_atomic(
        cls, tensor_shapes: Iterable[Shape], dp_size: int, *, reference: TensorAtomic
    ) -> "GlobalLayout":
        """Compute global tensor element offsets from a tensor-to-owner-rank assignment.

        The global buffer is the concatenation of one contiguous segment per rank,
        ordered by rank id. Rank ``r``'s segment holds exactly the tensors that
        ``reference.tensor_to_owner_rank`` assigns to ``r``, packed back to back in
        ascending tensor-id order. Because tensors are never split across ranks
        there is no row- or block-alignment constraint, and because segments are
        packed tightly there is no padding: ``size`` equals the sum of all tensor
        numels. Segment lengths are in general different per rank.

        For example, with shapes P0=(2, 6), P1=(4, 4), P2=(1, 2) and the assignment
        ``{0: 1, 1: 0, 2: 1}`` on 2 ranks:

        ```
        rank 0 [ 0, 16): | P1                      |
        rank 1 [16, 30): | P0             | P2     |
        ```

        Args:
            tensor_shapes: Logical tensor shapes in tensor-id order.
            dp_size: Data-parallel shard count for this global layout.
            reference: ``TensorAtomic`` placement whose ``tensor_to_owner_rank``
                maps every tensor id in ``range(len(tensor_shapes))`` to an owner
                rank in ``[0, dp_size)``.

        Returns:
            Global layout whose per-rank segments are gap-free and whose ``size`` is
            the total number of logical elements.
        """

        tensor_to_owner_rank = reference.tensor_to_owner_rank
        if len(tensor_shapes) != len(tensor_to_owner_rank):
            raise ValueError(
                "In planning a layout for TensorAtomic placement, the number of tensors must "
                "equal to the number of assignment entries, "
                f"but got {len(tensor_shapes)} and {len(tensor_to_owner_rank)}."
            )

        if set(tensor_to_owner_rank.keys()) != set(range(len(tensor_shapes))):
            raise ValueError(
                "In planning a layout for TensorAtomic placement, the keys of "
                "'tensor_to_owner_rank' must exactly match the indices of tensors of range "
                f"({len(tensor_shapes)}), "
                f"but got {tensor_to_owner_rank.keys()}"
            )

        candidate_rank_set = set(tensor_to_owner_rank.values())
        if min(candidate_rank_set) < 0 or max(candidate_rank_set) >= dp_size:
            raise ValueError(
                "In planning a layout for TensorAtomic placement, the owner of each tensor must be "
                f"within the range of data parallel size: [0, {dp_size}), "
                f"but got {min(candidate_rank_set)} for minimum, "
                f"and {max(candidate_rank_set)} for maximum."
            )

        # Group tensor ids by owner rank. Iterating the dict in insertion order and
        # appending keeps each rank's tensor list in a deterministic order.
        rank_to_tensors: dict[int, list[int]] = {rank: [] for rank in range(dp_size)}
        totalel = 0
        for tensor_index, owner_rank in tensor_to_owner_rank.items():
            rank_to_tensors[owner_rank].append(tensor_index)
            totalel += tensor_shapes[tensor_index].numel()

        # Lay out rank segments back to back in ascending rank order, packing each
        # rank's tensors contiguously inside its segment.
        offset = 0
        tensor_to_offset = [-1] * len(tensor_shapes)
        per_rank_numels = [0] * len(rank_to_tensors)
        for rank_id in sorted(rank_to_tensors.keys()):
            tensor_list = rank_to_tensors[rank_id]
            local_offset = 0
            for tensor_index in tensor_list:
                numel = tensor_shapes[tensor_index].numel()
                tensor_to_offset[tensor_index] = offset + local_offset
                local_offset += numel
                per_rank_numels[rank_id] += numel
            offset += local_offset

        return cls(
            tensor_shapes=tensor_shapes,
            tensor_to_offset=tuple(tensor_to_offset),
            size=totalel,
            reference=reference,
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

        if isinstance(self.reference, TensorAtomic):
            # Recompute each rank's segment from the owner assignment and check that
            # every tensor sits inside its owner's segment.
            owner = self.reference.tensor_to_owner_rank
            dp_size = max(owner.values()) + 1
            per_rank = [0] * dp_size
            for tensor_id, rank in owner.items():
                per_rank[rank] += self.tensor_shapes[tensor_id].numel()
            if sum(per_rank) != self.size:
                raise AssertionError(
                    "For TensorAtomic reference placement, the sum of all ranks' local elements "
                    f"({sum(per_rank)}) should equal to total size ({self.size})."
                )
            segment_start = list(itertools.accumulate([0, *per_rank[:-1]]))
            for tensor_id, (shape, start) in enumerate(
                zip(self.tensor_shapes, self.tensor_to_offset)
            ):
                rank = owner[tensor_id]
                lo, hi = segment_start[rank], segment_start[rank] + per_rank[rank]
                if not (lo <= start and start + shape.numel() <= hi):
                    raise AssertionError(
                        f"Tensor {tensor_id} [{start}, {start + shape.numel()}) is out of "
                        f"owner rank {rank} segment [{lo}, {hi})."
                    )

        tensor_ranges: list[TensorRange] = []
        for tensor_id, (shape, start) in enumerate(
            zip(self.tensor_shapes, self.tensor_to_offset, strict=True)
        ):
            if start < 0:
                raise AssertionError(f"Tensor {tensor_id} offset {start} is negative.")

            if not isinstance(self.reference, TensorAtomic):
                # Flat / BlockAtomic shards may cut through a tensor, so every tensor must
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


def non_leading_numel(shape: torch.Size) -> int:
    """Return the number of elements after dim 0 for a non-scalar shape."""
    if len(shape) == 0:
        raise ValueError(f"DBuffer layout does not support 0D tensor shapes: {shape}.")
    return shape[1:].numel()


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
