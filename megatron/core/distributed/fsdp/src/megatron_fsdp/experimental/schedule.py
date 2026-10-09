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


@dataclass(frozen=True)
class PlanAction:
    """Communication annotation compiled for one logical trace occurrence."""

    prefetch_target: "FsdpModule | None" = None
    retain: bool = False


class TraceAndReplayScheduler:
    """Observe complete iterations and optimize communication without ordering compute.

    Replay validates every logical occurrence before executing its compiled action.
    Divergent or truncated replay raises after cleanup and invalidates the plan.
    Cleanup does not restore module hook phases or unwind interrupted forward or
    autograd execution. A replay mismatch is fatal to the current training
    execution: discard the failed graph and construct a fresh model/runtime
    rather than resume it. Invalidation between completed iterations is safe
    and allows the next iteration to trace a deliberately changed pattern.

    All ranks must follow collective-compatible control flow. Local divergence
    recovery cannot undo speculative collectives already submitted, and cannot
    make rank-divergent execution safe.
    """

    def __init__(self, max_reuse_distance: int | None = 0) -> None:
        if max_reuse_distance is not None and max_reuse_distance < 0:
            raise ValueError("max_reuse_distance must be non-negative or None.")
        self.max_reuse_distance = max_reuse_distance
        self._plan: list[TraceEvent] = []
        self._actions: list[PlanAction] = []
        self._events: list[TraceEvent] = []
        self._position = 0
        self._active = False
        self._held: dict[int, FsdpModule] = {}
        self._touched: dict[int, FsdpModule] = {}

    def begin_iteration(self) -> None:
        """Begin one global batch, including all its microbatches."""
        if self._active:
            raise RuntimeError("An FSDP trace iteration is already active.")
        self._active = True
        self._events = []
        self._position = 0
        self._touched = {}

    def end_iteration(self) -> None:
        """Compile a full trace or validate replay, then release speculative storage."""
        if not self._active:
            raise RuntimeError("No FSDP trace iteration is active.")
        if self._plan and self._position != len(self._plan):
            self.abort_iteration()
            raise RuntimeError("FSDP trace replay ended before all logical events were consumed.")
        self._release_held()
        self._release_touched()
        if not self._plan:
            self._plan = list(self._events)
            self._compile_actions()
        self._active = False

    def abort_iteration(self) -> None:
        """Invalidate the plan and release storage, without restoring hook lifecycle state.

        After interrupted module execution, use a fresh model/runtime. Calling
        this between completed iterations safely prepares a new execution pattern.
        """
        self._release_held()
        self._release_touched()
        self._plan = []
        self._actions = []
        self._events = []
        self._active = False

    def _record(self, event: TraceEvent) -> PlanAction:
        if not self._active:
            raise RuntimeError("Call context.begin_iteration() before using trace replay.")
        if self._plan:
            if self._position >= len(self._plan) or self._plan[self._position] != event:
                self.abort_iteration()
                raise RuntimeError("FSDP trace replay diverged from its logical event sequence.")
            action = self._actions[self._position]
            self._position += 1
            return action
        self._events.append(event)
        return PlanAction()

    def unshard(self, module: "FsdpModule", prefetch: str = "none") -> None:
        """Execute a validated demand gather and wait, then its annotated prefetch."""
        action = self._record(TraceEvent("unshard", module, prefetch))
        self._touched[id(module)] = module
        self._held.pop(id(module), None)
        module._unshard_parameter_groups()
        assert module._unshard_event is not None
        module.context.current_stream().wait_event(module._unshard_event)
        target = action.prefetch_target
        if target is not None and target._unshard_event is None:
            self._held[id(target)] = target
            target._unshard_parameter_groups()

    def reshard(self, module: "FsdpModule") -> None:
        """Execute a validated release, retaining only a bounded planned reuse."""
        action = self._record(TraceEvent("reshard", module))
        if action.retain:
            self._held[id(module)] = module
            return
        self._held.pop(id(module), None)
        module._reshard_parameter_groups()

    def _compile_actions(self) -> None:
        actions = []
        for position, event in enumerate(self._plan):
            target = None
            retain = False
            if event.kind == "reshard" and self.max_reuse_distance is not None:
                for successor_position in range(position + 1, len(self._plan)):
                    successor = self._plan[successor_position]
                    if successor.module is event.module:
                        retain = (
                            successor.kind == "unshard"
                            and successor_position - position - 1 <= self.max_reuse_distance
                        )
                        break
            if event.kind == "unshard" and event.phase != "none":
                budget = (
                    event.module._schedule_policy.backward_prefetch_size
                    if event.phase == "backward"
                    else event.module._schedule_policy.forward_prefetch_size
                )
                if budget != 0:
                    released: set[int] = set()
                    for successor in self._plan[position + 1 :]:
                        if successor.kind == "reshard":
                            released.add(id(successor.module))
                        elif (
                            successor.module is not event.module
                            and id(successor.module) not in released
                        ):
                            target = successor.module
                            break
            actions.append(PlanAction(target, retain))
        self._actions = actions

    def _release_held(self) -> None:
        for module in self._held.values():
            module._reshard_parameter_groups()
        self._held.clear()

    def _release_touched(self) -> None:
        for module in self._touched.values():
            if module._unshard_event is not None:
                module._reshard_parameter_groups()
        self._touched.clear()
