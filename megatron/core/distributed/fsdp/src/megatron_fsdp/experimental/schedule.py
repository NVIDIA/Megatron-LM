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

    Attributes:
        forward_prefetch_size: How much this module prefetches for its successors
            in forward order. ``None`` (the default) prefetches one successor, ``0``
            disables prefetching, and positive values are parameter-element budgets.
        backward_prefetch_size: The same as ``forward_prefetch_size``, for backward
            order.
        unshard_all_planes_in_backward: Whether backward unshards and prefetches of this
            module gather MXFP8 rowwise planes along with columnwise ones. Enable it when
            activations recomputed in backward use this module's MXFP8 weights without
            re-entering this module's forward hooks, e.g. a checkpointed submodule.
            Otherwise, backward unshards gather only columnwise planes, and the
            recomputation fails in Transformer Engine.
    """

    forward_prefetch_size: int | None = None
    backward_prefetch_size: int | None = None
    unshard_all_planes_in_backward: bool = False

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
