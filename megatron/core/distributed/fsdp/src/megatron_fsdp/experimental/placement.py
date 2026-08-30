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

"""DBuffer placement definitions.

DBuffer uses PyTorch DTensor's ``Placement``, ``Replicate``, and ``Partial``
types directly. ``Flat`` and ``BlockAtomic`` are DBuffer-specific dim-0
``Shard`` placements whose local storage is part of one flattened buffer.

=============  =============  ====================
Source         Destination    DBuffer operation
=============  =============  ====================
sharded        ``Replicate``  ``allgather()``
``Partial``    sharded        ``reduce_scatter()``
``Partial``    ``Replicate``  ``allreduce()``
``Replicate``  sharded        ``scatter()`` (local)
=============  =============  ====================
"""

from collections.abc import Iterable

from torch.distributed.tensor import Shard
from torch.distributed.tensor.placement_types import Placement

__all__ = ["BlockAtomic", "Flat", "changed_mesh_axis"]


class Flat(Shard):
    """DBuffer-specific flattened dim-0 shard placement."""

    def __init__(self) -> None:
        super().__init__(0)

    def __eq__(self, other: object) -> bool:
        # PyTorch Shard.__eq__ compares only dim, so distinguish Flat from BlockAtomic.
        return isinstance(other, Shard) and other.dim == 0 and not isinstance(other, BlockAtomic)


class BlockAtomic(Shard):
    """Flattened dim-0 shard placement that keeps ``block_size`` rows together."""

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"BlockAtomic block_size must be positive, got {block_size}.")
        super().__init__(0)
        self.block_size = block_size

    def __eq__(self, other: object) -> bool:
        # PyTorch Shard.__eq__ compares only dim, so preserve the block size as well.
        return isinstance(other, BlockAtomic) and self.block_size == other.block_size

    def __repr__(self) -> str:
        return f"BlockAtomic(block_size={self.block_size})"


def changed_mesh_axis(
    old_placements: Iterable[Placement], new_placements: Iterable[Placement]
) -> int | None:
    """Return the changed mesh axis, requiring at most one placement change."""
    changed_axis = None
    for axis, (old_placement, new_placement) in enumerate(
        zip(old_placements, new_placements, strict=True)
    ):
        if old_placement == new_placement:
            continue
        if changed_axis is not None:
            raise NotImplementedError(
                "Expected at most one changed placement axis, "
                f"got changed axes {changed_axis} and {axis}."
            )
        changed_axis = axis
    return changed_axis
