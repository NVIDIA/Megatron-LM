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

"""DBuffer placement definitions."""

import dataclasses
from collections.abc import Iterable

import torch.distributed as dist


class Placement:
    """Base class for DBuffer placements."""


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


def _validate_placement(placement: Placement) -> None:
    if not isinstance(placement, (Replicate, Partial, Flat)):
        raise TypeError(f"Unsupported DBuffer placement: {placement!r}.")


def validate_placements(placements: Iterable[Placement]) -> None:
    """Validate DBuffer placements form a supported contiguous local layout."""
    seen_flat = False
    for placement in placements:
        _validate_placement(placement)
        if isinstance(placement, Flat):
            seen_flat = True
        elif seen_flat:
            raise ValueError(
                "Flat placements must be a suffix of the placement list so each "
                "local buffer is a contiguous global-buffer range."
            )
