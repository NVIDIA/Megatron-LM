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
