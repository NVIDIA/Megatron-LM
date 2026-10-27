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
    """Control prefetching and execution scheduling for one FSDP module.

    Attributes:
        forward_prefetch_size: Forward prefetch budget in parameter elements.
            None prefetches one successor; 0 disables prefetching.
        backward_prefetch_size: Backward prefetch budget in parameter elements.
            None prefetches one successor; 0 disables prefetching.
        register_hooks: Register automatic forward/backward execution hooks.
            Disable when an external scheduler drives the FSDP lifecycle explicitly.
            The state-dict safety hook remains registered independently.
        grad_accumulation_count: Total parameter-gradient callbacks per reduction
            window, including separate GraphTasks and delayed weight gradients.
            None defaults to one callback per owned trainable parameter. The scheduler
            must declare the complete count before sharding the unit.
        unshard_rowwise_in_backward: Whether backward unshards and prefetches of this
            module gather MXFP8 rowwise planes along with columnwise ones, keeping them
            until post_backward. Enable it when activations recomputed in backward use
            this module's MXFP8 weights. Otherwise, backward unshards gather only
            columnwise planes, so either:
            1) a recomputation that does not re-enter this module's forward hooks finds
               no rowwise planes and fails in Transformer Engine; or
            2) a recomputation that does re-enter them gathers the rowwise planes
               separately on demand, without overlap, degrading performance.
    """

    forward_prefetch_size: int | None = None
    backward_prefetch_size: int | None = None
    register_hooks: bool = True
    grad_accumulation_count: int | None = None
    unshard_rowwise_in_backward: bool = False

    def __post_init__(self) -> None:
        """Validate non-negative prefetch budgets."""
        if self.forward_prefetch_size is not None and self.forward_prefetch_size < 0:
            raise ValueError(
                "forward_prefetch_size must be non-negative, " f"got {self.forward_prefetch_size}."
            )
        if self.backward_prefetch_size is not None and self.backward_prefetch_size < 0:
            raise ValueError(
                "backward_prefetch_size must be non-negative, "
                f"got {self.backward_prefetch_size}."
            )
