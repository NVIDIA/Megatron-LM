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

"""Shared runtime configuration for Megatron-FSDP contexts."""

import enum
from dataclasses import dataclass


class PrefetchOrder(enum.Enum):
    """How to choose the module order used for prefetching."""

    REGISTRATION = enum.auto()
    RECORDED = enum.auto()


@dataclass(frozen=True)
class RuntimePolicy:
    """Configure communication allocation, streams, and prefetch for an FSDP context."""

    use_symmetric_memory: bool = False
    """Allocate communication staging buffers from PyTorch's NCCL symmetric-memory pool."""

    overlap_allgather_reduction: bool = True
    """Use separate all-gather and reduction streams.

    Disable to reduce peak transient memory. See
    https://github.com/NVIDIA/Megatron-LM/issues/6471.
    """

    overlap_outer_communication: bool = False
    """Run DP-outer reduction on a separate stream, independently of all-gather overlap."""

    prefetch_order: PrefetchOrder = PrefetchOrder.REGISTRATION
    """``REGISTRATION`` expects a forward/backward pass matching module
    registration order. ``RECORDED`` gathers parameters on demand and records
    module calls until the first ``post_backward()``. Subsequent passes must
    repeat the complete sequence consistently across data-parallel ranks.
    Recording currently requires the ordinary module hooks; combined 1F1B
    bypasses them. Discard the context if the recorded pass raises.
    """
