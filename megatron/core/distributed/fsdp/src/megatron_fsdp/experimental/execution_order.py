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

"""Weakly held construction or recorded module execution sequences."""

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
    """An ordered sequence of module/phase occurrences."""

    def __init__(self, items: Iterable[ExecutionEvent] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[tuple[ref[nn.Module], Phase]] = []
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: ExecutionEvent) -> None:
        """Append an occurrence."""
        key = (ref(item.module), item.phase)
        self._items.append(key)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over occurrences in execution order."""
        for index in range(len(self)):
            yield self[index]

    def reset_cursor(self) -> None:
        """Reset traversal without changing the recorded sequence."""
        self._position = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def position(self) -> int:
        """Index of the next occurrence; len(order) means traversal is exhausted."""
        return self._position

    def __getitem__(self, index: int) -> ExecutionEvent:
        """Read an occurrence without advancing the cursor."""
        module_ref, phase = self._items[index]
        module = module_ref()
        if module is None:
            raise RuntimeError("A module in the FSDP execution order was destroyed.")
        return ExecutionEvent(module=module, phase=phase)

    def advance(self) -> ExecutionEvent:
        """Consume the next occurrence."""
        if self._position >= len(self._items):
            raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
        result = self[self._position]
        self._position += 1
        return result
