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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import FsdpModule


@dataclass(frozen=True)
class SchedulePolicy:
    """Control communication scheduling for one FSDP module.

    ``None`` prefetches one successor, preserving the default behavior. ``0``
    disables prefetching. Positive values specify parameter-element budgets.
    """

    forward_prefetch_size: int | None = None
    backward_prefetch_size: int | None = None

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


@dataclass(frozen=True)
class TraceEvent:
    """One logical materialization or release, including its consumption phase."""

    kind: str
    module: "FsdpModule"
    phase: str = "none"


class TraceAndReplayScheduler:
    """Observe complete iterations and optimize communication without ordering compute.

    A divergent iteration runs demand-only and is discarded. The next complete
    iteration is traced afresh. Speculative and retained materializations are
    released on divergence, iteration completion, and abort.

    All ranks must follow collective-compatible control flow. Local divergence
    recovery cannot undo speculative collectives already submitted, and cannot
    make rank-divergent execution safe.
    """

    def __init__(self) -> None:
        self._plan: list[TraceEvent] = []
        self._events: list[TraceEvent] = []
        self._position = 0
        self._active = False
        self._diverged = False
        self._held: dict[int, FsdpModule] = {}
        self._touched: dict[int, FsdpModule] = {}

    def begin_iteration(self) -> None:
        """Begin one global batch, including all its microbatches."""
        if self._active:
            raise RuntimeError("An FSDP trace iteration is already active.")
        self._active = True
        self._events = []
        self._position = 0
        self._diverged = False
        self._touched = {}

    def end_iteration(self) -> None:
        """Compile a full trace or validate replay, then release speculative storage."""
        if not self._active:
            raise RuntimeError("No FSDP trace iteration is active.")
        if self._plan and self._position != len(self._plan):
            self._diverged = True
        self._release_held()
        self._release_touched()
        self._plan = [] if self._diverged else list(self._events)
        self._active = False

    def abort_iteration(self) -> None:
        """Discard an interrupted iteration and release speculative storage."""
        self._release_held()
        self._release_touched()
        self._plan = []
        self._events = []
        self._active = False

    def _record(self, event: TraceEvent) -> bool:
        if not self._active:
            raise RuntimeError("Call context.begin_iteration() before using trace replay.")
        replaying = bool(self._plan) and not self._diverged
        if replaying:
            if self._position >= len(self._plan) or self._plan[self._position] != event:
                self._diverged = True
                self._release_held()
                replaying = False
            else:
                self._position += 1
        self._events.append(event)
        return replaying

    def record_unshard(self, module: "FsdpModule", phase: str) -> None:
        """Validate a demand unshard before using any speculative materialization."""
        self._record(TraceEvent("unshard", module, phase))
        self._touched[id(module)] = module
        self._held.pop(id(module), None)

    def prefetch(self, module: "FsdpModule", phase: str) -> None:
        """Prefetch one safe successor after the current demand wait."""
        if phase == "none" or not self._plan or self._diverged:
            return
        budget = (
            module._schedule_policy.backward_prefetch_size
            if phase == "backward"
            else module._schedule_policy.forward_prefetch_size
        )
        if budget == 0:
            return
        released: set[int] = set()
        for event in self._plan[self._position :]:
            if event.kind == "reshard":
                released.add(id(event.module))
            elif event.module is not module and id(event.module) not in released:
                if event.module._unshard_event is None:
                    self._held[id(event.module)] = event.module
                    event.module._unshard_parameter_groups()
                return

    def record_reshard(self, module: "FsdpModule") -> bool:
        """Retain storage only when the next logical operation consumes it again."""
        replaying = self._record(TraceEvent("reshard", module))
        if replaying and self._position < len(self._plan):
            successor = self._plan[self._position]
            if successor.kind == "unshard" and successor.module is module:
                self._held[id(module)] = module
                return True
        self._held.pop(id(module), None)
        return False

    def _release_held(self) -> None:
        for module in self._held.values():
            module._reshard_parameter_groups()
        self._held.clear()

    def _release_touched(self) -> None:
        for module in self._touched.values():
            if module._unshard_event is not None:
                module._reshard_parameter_groups()
        self._touched.clear()
