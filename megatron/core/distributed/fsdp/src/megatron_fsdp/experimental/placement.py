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
``Replicate``  sharded        ``view()`` (local)
=============  =============  ====================
"""

from collections.abc import Iterable

import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

__all__ = ["BlockAtomic", "Flat", "changed_mesh_axis", "sharded_reduce_group"]


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


def sharded_reduce_group(mesh: DeviceMesh, placements: Iterable[Placement]) -> dist.ProcessGroup:
    """Return a process group spanning every mesh axis ``placements`` shards over.

    A reduction whose result must see the whole buffer -- such as the amax that
    Transformer Engine MAX-reduces while quantizing MXFP8 master weights -- has to
    include every ``Shard`` axis, because each rank of such an axis owns a
    rank-disjoint block. One sharded axis uses that axis's group; several use the
    flattened group over all of them, because no single mesh axis spans disjoint
    shards on more than one axis. With no sharded axis every rank holds an
    identical copy, so this mesh's axis 0 is the documented, idempotent fallback.
    The default process group must NOT be used: it spans unrelated PP/TP ranks
    holding different parameters and would silently corrupt the result.

    A ``Partial`` axis holds a full-size unreduced contribution rather than a
    rank-disjoint block, so no group can span "its shards"; it is rejected rather
    than silently treated as either replicated or sharded.
    """
    sharded_axes = [
        axis for axis, placement in enumerate(placements) if isinstance(placement, Shard)
    ]
    for axis, placement in enumerate(placements):
        if not isinstance(placement, (Shard, Replicate)):
            raise NotImplementedError(
                f"Unsupported placement {placement!r} on mesh axis {axis}: expected Shard or "
                "Replicate. A Partial axis is a full-size unreduced contribution, not a "
                "rank-disjoint block, so no process group spans its shards."
            )
    if not sharded_axes:
        return mesh.get_group(0)
    if len(sharded_axes) == 1:
        return mesh.get_group(sharded_axes[0])
    if len(sharded_axes) == mesh.ndim:
        flattened_group = getattr(mesh, "_mfsdp_flattened_group", None)
        if flattened_group is not None:
            return flattened_group
        return mesh._flatten().get_group()
    raise NotImplementedError(
        f"Cannot reduce over sharded mesh axes {sharded_axes} of a {mesh.ndim}-dimensional "
        "mesh: they are neither empty, a single axis, nor every axis, so no existing process "
        "group spans exactly their shards."
    )
