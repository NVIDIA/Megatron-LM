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

"""Weak, phase-tagged execution sequences with a sequential cursor."""

from collections.abc import Iterable, Iterator
from typing import NamedTuple
from weakref import ref

from torch import nn

from .phase import Phase


class ExecutionEvent(NamedTuple):
    """A module and its compute phase."""

    module: nn.Module
    phase: Phase


class ExecutionOrder:
    """Store an execution trace with one sequential cursor across all phases."""

    def __init__(self, items: Iterable[ExecutionEvent] = ()) -> None:
        """Create a weak sequence, preserving every repeated occurrence."""
        self._items: list[tuple[ref[nn.Module], Phase]] = []
        self._position = 0
        for item in items:
            self.append(item)

    @property
    def position(self) -> int:
        """Index of the next event to consume."""
        return self._position

    def append(self, item: ExecutionEvent) -> None:
        """Append an event without consuming it."""
        self._items.append((ref(item.module), item.phase))

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over live occurrences without retaining their modules."""
        for module_ref, phase in self._items:
            module = module_ref()
            if module is not None:
                yield ExecutionEvent(module, phase)

    def peek(self, offset: int = 0) -> ExecutionEvent | None:
        """Read an event relative to the cursor without wrapping or advancing."""
        index = self._position + offset
        if not 0 <= index < len(self._items):
            return None
        module_ref, phase = self._items[index]
        module = module_ref()
        return None if module is None else ExecutionEvent(module, phase)

    def successors(self) -> Iterator[ExecutionEvent]:
        """Look ahead from the current occurrence without wrapping or advancing."""
        offset = 1
        while (event := self.peek(offset)) is not None:
            yield event
            offset += 1

    def advance(self) -> None:
        """Consume one occurrence and automatically wrap at the end of the trace."""
        if not self._items:
            raise RuntimeError("Cannot advance an empty FSDP execution order.")
        self._position = (self._position + 1) % len(self._items)
