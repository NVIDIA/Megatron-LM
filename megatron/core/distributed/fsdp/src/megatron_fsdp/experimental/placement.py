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

These placement concepts are borrowed from PyTorch DTensor placements:
``Replicate`` and ``Partial`` mirror DTensor's placements. ``Flat`` is the
only sharded DBuffer placement implemented so far; it stores dim-0 shards in a
flattened local buffer.

=============  =============  ====================
Source         Destination    DBuffer operation
=============  =============  ====================
sharded        ``Replicate``  ``allgather()``
``Partial``    sharded        ``reduce_scatter()``
``Partial``    ``Replicate``  ``allreduce()``
``Replicate``  sharded        ``scatter()`` (local)
=============  =============  ====================
"""

import dataclasses
from collections.abc import Iterable

import torch.distributed as dist


class Placement:
    """Base class for DBuffer placements."""


MeshAxis = int | str


@dataclasses.dataclass(frozen=True)
class Replicate(Placement):
    """Replicated local buffer placement."""


@dataclasses.dataclass(frozen=True)
class Partial(Placement):
    """Unreduced replicated local buffer placement."""

    reduce_op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM


@dataclasses.dataclass(frozen=True)
class Flat(Placement):
    """Flat per-unit dim-0 sharded local buffer placement."""


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


@dataclasses.dataclass(frozen=True)
class Placements:
    """Per-mesh-axis placements for parameter, gradient, and optimizer buffers."""

    dp_axes: list[MeshAxis]
    parameter: list[Placement]
    gradient: list[Placement]
    optimizer: list[Placement]

    def __post_init__(self) -> None:
        """Validate placement list lengths."""
        axis_count = len(self.dp_axes)
        for name, placements in (
            ("parameter", self.parameter),
            ("gradient", self.gradient),
            ("optimizer", self.optimizer),
        ):
            if len(placements) != axis_count:
                raise ValueError(f"Expected {axis_count} {name} placements, got {len(placements)}.")
