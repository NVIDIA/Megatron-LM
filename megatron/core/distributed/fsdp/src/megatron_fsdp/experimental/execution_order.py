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

"""One weak lifecycle event source with independent filtered cursors."""

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import NamedTuple
from weakref import ref

import torch
from torch import nn

from .phase import Phase


class ExecutionAction(enum.Enum):
    """Weight requests, reshards and completed local gradient contributions.

    UNSHARD records a weight request, including reuse of already gathered weights.
    """

    UNSHARD = enum.auto()
    RESHARD = enum.auto()
    GRAD_READY = enum.auto()


@dataclass(frozen=True)
class ExecutionEvent:
    """An event with optional byte observations.

    ``event_id`` is assigned by the owning ExecutionOrder, not a microbatch or
    module ID. Byte observations and IDs do not affect event matching. Callers
    supply the observations. Missing measurements stay None, distinct from a real zero.
    """

    module: nn.Module
    phase: Phase | None
    action: ExecutionAction = ExecutionAction.UNSHARD
    allocated_bytes: int | None = field(default=None, compare=False)
    event_id: int | None = field(default=None, compare=False)
    # MXFP8 can release rowwise weights while columnwise weights stay live.
    rowwise_only: bool = False
    # Full-owner gathered capacity, including representations retained by a partial release.
    unsharded_bytes: int | None = field(default=None, compare=False)
    finalizes_outer: bool = False


class _StoredEvent(NamedTuple):
    module: ref[nn.Module]
    phase: Phase | None
    action: ExecutionAction
    allocated_bytes: int | None
    rowwise_only: bool
    unsharded_bytes: int | None
    finalizes_outer: bool


class ExecutionOrder:
    """Own an append-only weak event source, without a consumer cursor.

    IDs are zero-based append indices local to this source. Expired modules do
    not remove or renumber events. Plans refer to IDs without editing the source.
    """

    def __init__(self, items: Iterable[ExecutionEvent] = ()) -> None:
        """Create a source that records repeated events separately."""
        self._items: list[_StoredEvent] = []
        for item in items:
            self.append(item)

    def record_event(self, item: ExecutionEvent, *, device: torch.device) -> int:
        """Sample rank-local allocated bytes at this eager boundary and append the event.

        The caller chooses the observation point. The sample does not synchronize
        CUDA or measure an interval peak.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Record FSDP events before CUDA graph capture.")
        return self.append(replace(item, allocated_bytes=torch.cuda.memory_allocated(device)))

    def append(self, item: ExecutionEvent) -> int:
        """Append an event and return its assigned ID without advancing any view."""
        if type(item.rowwise_only) is not bool or (
            item.rowwise_only and item.action is not ExecutionAction.RESHARD
        ):
            raise ValueError("rowwise_only is a boolean flag for RESHARD events only.")
        if type(item.finalizes_outer) is not bool or (
            item.finalizes_outer and item.action is not ExecutionAction.GRAD_READY
        ):
            raise ValueError("finalizes_outer is a boolean flag for GRAD_READY events only.")
        for name in ("allocated_bytes", "unsharded_bytes"):
            value = getattr(item, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
        event_id = len(self._items)
        self._items.append(
            _StoredEvent(
                ref(item.module),
                item.phase,
                item.action,
                item.allocated_bytes,
                item.rowwise_only,
                item.unsharded_bytes,
                item.finalizes_outer,
            )
        )
        return event_id

    def __len__(self) -> int:
        """Number of events, including those whose modules have expired."""
        return len(self._items)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over live events without changing any view cursor."""
        for event_id in range(len(self._items)):
            if (event := self.seek(event_id)) is not None:
                yield event

    def seek(self, event_id: int) -> ExecutionEvent | None:
        """Read an event by ID without moving a cursor.

        Return None if the ID is absent or its module has expired. The returned
        event holds a strong reference to its module.
        """
        if not 0 <= event_id < len(self._items):
            return None
        item = self._items[event_id]
        module = item.module()
        if module is None:
            return None
        return ExecutionEvent(
            module,
            item.phase,
            item.action,
            item.allocated_bytes,
            event_id,
            item.rowwise_only,
            item.unsharded_bytes,
            item.finalizes_outer,
        )

    def view(self, *, action: ExecutionAction | None = None) -> "ExecutionOrderView":
        """Select an action without copying event records or sharing cursor state."""
        return ExecutionOrderView(self, action=action)


class ExecutionOrderView:
    """An action-filtered index into one source, with its own local cursor.

    Views select events present at creation. Cursor positions count only selected
    events; returned events keep their IDs from the source. Only IDs are stored,
    so neither source nor view keeps modules alive.
    """

    def __init__(self, source: ExecutionOrder, *, action: ExecutionAction | None = None) -> None:
        """Create a view of ``source`` with its own cursor."""
        self._source = source
        self._wrap = action is not None
        self._event_ids = tuple(
            event_id
            for event_id, item in enumerate(source._items)
            if action is None or item.action is action
        )
        self._position = 0

    @property
    def source(self) -> ExecutionOrder:
        """The sole owner of this view's event records."""
        return self._source

    @property
    def position(self) -> int:
        """Local index of the next event to consume, distinct from its event ID."""
        return self._position

    def __len__(self) -> int:
        """Number of selected events, including those with expired modules."""
        return len(self._event_ids)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over live selected events without advancing this cursor."""
        for index in range(len(self)):
            if (event := self._source.seek(self._event_ids[index])) is not None:
                yield event

    def peek(self, offset: int = 0) -> ExecutionEvent | None:
        """Read relative to this cursor without wrapping or advancing."""
        index = self._position + offset
        return self._source.seek(self._event_ids[index]) if 0 <= index < len(self) else None

    def successors(self) -> Iterator[ExecutionEvent]:
        """Look ahead without wrapping, stopping at an expired module reference."""
        offset = 1
        while (event := self.peek(offset)) is not None:
            yield event
            offset += 1

    def advance(self) -> None:
        """Consume one occurrence; only filtered hint views wrap automatically."""
        if self._position >= len(self._event_ids):
            raise RuntimeError("Cannot advance an empty or exhausted FSDP execution order.")
        self._position += 1
        if self._wrap and self._position == len(self._event_ids):
            self._position = 0

    def reset(self) -> None:
        """Start the next complete execution after its caller validates completion."""
        self._position = 0
