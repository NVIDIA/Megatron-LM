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

"""Record and replay demand occurrences over a complete FSDP iteration."""

from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from .execution_order import ExecutionOrder

if TYPE_CHECKING:
    from .context import FsdpContext
    from .module import FsdpModule

DemandPhase = Literal["forward", "backward", "recompute", "none"]


class IterationOrder:
    """Own execution and prefetch iterators over one weak occurrence sequence.

    State lives on FsdpContext, not a ContextVar, since autograd hooks may run
    on another thread. One context cannot execute overlapping iteration scopes.
    """

    def __init__(self) -> None:
        self._orders: dict[Hashable, tuple[ExecutionOrder, tuple[DemandPhase, ...]]] = {}
        self._order: ExecutionOrder | None = None
        self._phases: list[DemandPhase] | tuple[DemandPhase, ...] = ()
        self._execution: Iterator | None = None
        self._unshard: Iterator | None = None
        self._execution_position = 0
        self._unshard_position = 0
        self._recording = False

    @property
    def active(self) -> bool:
        return self._order is not None

    @property
    def recording(self) -> bool:
        return self._recording

    @contextmanager
    def iteration(self, key: Hashable, *, rerecord: bool = False) -> Iterator[None]:
        """Record once per schedule key, then validate complete cursor replay."""
        if self.active:
            raise RuntimeError("FSDP iteration scopes cannot overlap on the same context.")
        self._recording = rerecord or key not in self._orders
        if self._recording:
            self._order, self._phases = ExecutionOrder(), []
        else:
            self._order, self._phases = self._orders[key]
        self._execution = iter(self._order)
        self._unshard = iter(self._order)
        self._execution_position = self._unshard_position = 0
        try:
            yield
            if self._execution_position != len(self._order):
                raise RuntimeError(
                    "FSDP iteration ended before consuming its recorded demand order."
                )
            if self._recording:
                self._orders[key] = self._order, tuple(self._phases)
        except BaseException:
            self._orders.pop(key, None)
            raise
        finally:
            self._order = self._execution = self._unshard = None
            self._phases = ()
            self._recording = False
            self._execution_position = self._unshard_position = 0

    def record(self, module: "FsdpModule", phase: DemandPhase) -> None:
        """Observe a demand only during the initial recording iteration."""
        if not self._recording or self._order is None:
            raise RuntimeError("No FSDP iteration recording is active.")
        self._order.append(module)
        self._phases.append(phase)

    def advance(self) -> tuple["FsdpModule", DemandPhase]:
        """Consume the next occurrence; the caller validates its observed demand."""
        if self._execution is None:
            raise RuntimeError("No FSDP iteration is active.")
        try:
            module = next(self._execution)
        except StopIteration:
            raise RuntimeError("FSDP demand exceeded the recorded iteration order.") from None
        phase = self._phases[self._execution_position]
        self._execution_position += 1
        return module, phase

    def prefetch(self, budget: int | None) -> None:
        """Advance a separate, monotonic cursor over upcoming demand occurrences.

        Targets depend only on order and budget, not materialization state.
        The gather helper owns idempotence. An intervening release can still
        invalidate a speculative gather; demand must recheck and wait normally.
        """
        if not self.active or self._recording or budget == 0:
            return
        while self._unshard_position < self._execution_position:
            next(self._unshard)
            self._unshard_position += 1
        size = 0
        for module in self._unshard:
            self._unshard_position += 1
            module._unshard_parameter_groups()
            size += module.num_parameter_elements
            if budget is None or size >= budget:
                break


@contextmanager
def iteration(
    context: "FsdpContext", *, key: Hashable = "default", rerecord: bool = False
) -> Iterator[None]:
    """Scope the complete local schedule, including every VPP model chunk.

    All DP peers must follow compatible stable schedules. Use separate keys
    for training/validation and changed configurations, or rerecord explicitly.
    Failures invalidate the cache but do not roll back computation. The caller
    still owns finish_grad_sync after all backward/delayed-wgrad producers.
    """
    context.ensure_finalized()
    with context.iteration_order.iteration(key, rerecord=rerecord):
        # Covers optimizer updates and parameter initialization before any
        # speculative gather, including roots bypassed by combined execution.
        context.allgather_stream.wait_stream(context.current_stream())
        yield
