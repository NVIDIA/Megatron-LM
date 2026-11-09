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
        overlap_allgather_reduction: Use separate all-gather and reduction streams.
            Disable to reduce peak transient memory. See
            https://github.com/NVIDIA/Megatron-LM/issues/6471.
        overlap_outer_communication: Run the last-microbatch DP-outer reduction on
            a separate stream, independently of overlap_allgather_reduction.
        caller_managed_grad_sync: Leave gradient synchronization to the caller
            while retaining automatic weight cleanup. The caller must call
            finish_grad_sync() after all backward work, including delayed weight
            gradients, and before reading or modifying gradients.
    """

    use_symmetric_memory: bool = False
    overlap_allgather_reduction: bool = True
    overlap_outer_communication: bool = False
    caller_managed_grad_sync: bool = False
