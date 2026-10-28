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
from typing import TYPE_CHECKING
from weakref import ref

from .phase import Phase

if TYPE_CHECKING:
    from .module import FsdpModule

ExecutionEvent = tuple["FsdpModule", Phase]


class ExecutionOrder:
    """Look up unique modules and validate repeated module-call occurrences.

    A module appearing once can unambiguously select its position, preserving
    construction-order prefetch for reordered or repeated demand calls. Modules
    appearing more than once must match the next occurrence at the cursor.
    Prefetch lookahead never advances that cursor. Events are (module, phase)
    pairs; modules are held weakly inside the order.
    """

    def __init__(self, items: Iterable[ExecutionEvent] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[tuple[ref["FsdpModule"], Phase]] = []
        # An index identifies a unique item; None marks an ambiguous repeated item.
        self._unique_index: dict[tuple[ref["FsdpModule"], Phase], int | None] = {}
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: ExecutionEvent) -> None:
        """Append an occurrence and mark previously seen items as repeated."""
        module, phase = item
        key = (ref(module), phase)
        self._unique_index[key] = None if key in self._unique_index else len(self._items)
        self._items.append(key)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over occurrences in execution order."""
        for index in range(len(self)):
            yield self[index]

    def reset_cursor(self, position: int = 0) -> None:
        """Reset traversal without changing the recorded sequence."""
        self._position = position

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
        return module, phase

    def advance(self, item: ExecutionEvent | None = None) -> ExecutionEvent:
        """Locate a unique demand call or consume the next expected repeated call.

        With no item, consume the next occurrence without lookup.
        """
        if item is not None:
            module, phase = item
            key = (ref(module), phase)
            index = self._unique_index.get(key)
            if index is None:
                if not self._items or self._items[self._position % len(self._items)] != key:
                    raise RuntimeError(
                        "FSDP module calls diverged from the expected prefetch order."
                    )
                index = self._position % len(self._items)
            self._position = index
        if self._position >= len(self._items):
            raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
        result = self[self._position]
        self._position += 1
        return result
