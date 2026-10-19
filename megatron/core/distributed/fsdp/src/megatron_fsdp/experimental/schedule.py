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

"""Scheduling configuration for the minimal Megatron-FSDP path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulePolicy:
    """Control communication scheduling for one FSDP module.

    ``None`` prefetches one successor, preserving the default behavior. ``0``
    disables prefetching. Positive values specify parameter-element budgets.

    ``reduce_scatter_interval`` counts completed module backwards, not individual
    parameter callbacks. Values greater than one retain one full local-gradient
    buffer per trainable parameter group in main-gradient precision. Mark the
    last microbatch with ``microbatch(context, is_last=True)`` to reduce any
    incomplete window before consuming gradients. All data-parallel ranks must
    use the same policy and backward order. The default reduces every backward.
    """

    forward_prefetch_size: int | None = None
    backward_prefetch_size: int | None = None
    reduce_scatter_interval: int = 1

    def __post_init__(self) -> None:
        """Validate prefetch budgets and the gradient-reduction interval."""
        if type(self.reduce_scatter_interval) is not int or self.reduce_scatter_interval < 1:
            raise ValueError("reduce_scatter_interval must be a positive integer.")
        if self.forward_prefetch_size is not None and self.forward_prefetch_size < 0:
            raise ValueError(
                "forward_prefetch_size must be non-negative, " f"got {self.forward_prefetch_size}."
            )
        if self.backward_prefetch_size is not None and self.backward_prefetch_size < 0:
            raise ValueError(
                "backward_prefetch_size must be non-negative, "
                f"got {self.backward_prefetch_size}."
            )
