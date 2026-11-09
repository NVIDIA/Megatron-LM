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


class GradientFinalizationMode(enum.Enum):
    """Control outer reduction and the completion wait; inner reduction always runs."""

    SKIP_OUTER = enum.auto()
    """Skip outer reduction and the automatic completion wait during accumulation."""

    REDUCE_OUTER = enum.auto()
    """Launch outer reduction; the caller must call ``finish_grad_sync()`` afterward."""

    REDUCE_OUTER_AND_WAIT = enum.auto()
    """Launch outer reduction and wait for all gradient reductions at autograd completion."""


class PrefetchOrder(enum.Enum):
    """How to choose the module order used for prefetching."""

    REGISTRATION = enum.auto()
    RECORDED = enum.auto()


@dataclass(frozen=True)
class RuntimePolicy:
    """Configure communication and gradient synchronization for an FSDP context.

    Attributes:
        use_symmetric_memory: Allocate communication staging buffers from PyTorch's
            NCCL symmetric-memory pool.
        overlap_allgather_reduction: Use separate all-gather and reduction streams.
            Disable to reduce peak transient memory. See
            https://github.com/NVIDIA/Megatron-LM/issues/6471.
        overlap_outer_communication: Run the last-microbatch DP-outer reduction on
            a separate stream, independently of overlap_allgather_reduction.
        grad_sync_mode: Default gradient synchronization mode. Use REDUCE_OUTER
            when the caller waits with finish_grad_sync() after all backward work,
            including delayed weight gradients. finalize_gradients() scopes overrides.
    """

    use_symmetric_memory: bool = False
    overlap_allgather_reduction: bool = True
    overlap_outer_communication: bool = False
    grad_sync_mode: GradientFinalizationMode = GradientFinalizationMode.REDUCE_OUTER_AND_WAIT

    prefetch_order: PrefetchOrder = PrefetchOrder.REGISTRATION
