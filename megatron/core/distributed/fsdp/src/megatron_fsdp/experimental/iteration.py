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
from weakref import ref

from .execution_order import ExecutionOrder

if TYPE_CHECKING:
    from .context import FsdpContext
    from .module import FsdpModule

DemandPhase = Literal["forward", "backward", "recompute", "none"]
Demand = tuple[ref["FsdpModule"], DemandPhase]


class IterationOrder:
    """Own recorded occurrences and independent execution/prefetch cursors.

    State lives on FsdpContext, not a ContextVar, since autograd hooks may run
    on another thread. One context cannot execute overlapping iteration scopes.
    """

    def __init__(self) -> None:
        self._orders: dict[Hashable, ExecutionOrder[Demand]] = {}
        self._order: ExecutionOrder[Demand] | None = None
        self._recording = False
        self._prefetch_position = 0

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
            self._order = ExecutionOrder()
        else:
            self._order = self._orders[key]
        self._order.reset()
        self._prefetch_position = 0
        try:
            yield
            if self._order.position != len(self._order):
                raise RuntimeError(
                    "FSDP iteration ended before consuming its recorded demand order."
                )
            if self._recording:
                self._orders[key] = self._order
        except BaseException:
            self._orders.pop(key, None)
            raise
        finally:
            self._prefetch_position = 0
            self._order = None
            self._recording = False

    def record(self, module: "FsdpModule", phase: DemandPhase) -> None:
        """Observe a demand only during the initial recording iteration."""
        if not self._recording or self._order is None:
            raise RuntimeError("No FSDP iteration recording is active.")
        self._order.append((ref(module), phase))

    def advance(self) -> tuple["FsdpModule", DemandPhase]:
        """Consume the next occurrence; the caller validates its observed demand."""
        if self._order is None:
            raise RuntimeError("No FSDP iteration is active.")
        module_ref, phase = self._order.advance()
        return self._resolve(module_ref), phase

    @staticmethod
    def _resolve(module_ref: ref["FsdpModule"]) -> "FsdpModule":
        module = module_ref()
        if module is None:
            raise RuntimeError("A module in the recorded FSDP iteration was destroyed.")
        return module

    def prefetch(self, budget: int | None) -> None:
        """Advance a separate, monotonic cursor over upcoming demand occurrences.

        Targets depend only on order and budget, not materialization state.
        The gather helper owns idempotence. An intervening release can still
        invalidate a speculative gather; demand must recheck and wait normally.
        """
        if not self.active or self._recording or budget == 0:
            return
        self._prefetch_position = max(self._prefetch_position, self._order.position)
        size = 0
        while True:
            offset = self._prefetch_position - self._order.position + 1
            demand = self._order.next_item(offset)
            if demand is None:
                break
            module = self._resolve(demand[0])
            self._prefetch_position += 1
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
