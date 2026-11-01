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

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    """Configure communication and gradient synchronization for an FSDP context.

    Attributes:
        use_symmetric_memory: Allocate communication staging buffers from PyTorch's
            NCCL symmetric-memory pool.
        unify_communication_stream: Share one stream for all-gathers and
            reduce-scatters to reduce peak transient memory.
        caller_managed_grad_sync: Leave gradient synchronization to the caller
            while retaining automatic weight cleanup. The caller must call
            finish_grad_sync() after all backward work, including delayed weight
            gradients, and before reading or modifying gradients.
        partial_grad_buffers: Number of persistent reduce-scatter input buffers, each
            sized for the largest parameter group outside root modules (or of the roots,
            if there are no other modules) and reused round-robin. Packing a
            group's gradients then waits only for the reduction that last used its
            buffer, instead of every reduction queued on the reduce-scatter stream.
            0 allocates one buffer per reduction on the reduce-scatter stream, as do
            groups larger than a buffer.
            Ignored with use_symmetric_memory, whose buffers come from its own pool.
    """

    use_symmetric_memory: bool = False
    unify_communication_stream: bool = False
    caller_managed_grad_sync: bool = False
    partial_grad_buffers: int = 2

    def __post_init__(self) -> None:
        if self.partial_grad_buffers < 0:
            raise ValueError(
                f"partial_grad_buffers must be non-negative, got {self.partial_grad_buffers}."
            )
