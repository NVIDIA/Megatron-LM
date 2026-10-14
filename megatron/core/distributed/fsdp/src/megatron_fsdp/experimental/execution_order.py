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
from typing import Generic, TypeVar
from weakref import ref

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Store inferred construction order or recorded runtime occurrences.

    Construction order contains unique modules and looks up successors from the
    module being called, tolerating skipped or reordered execution. Recorded
    order preserves repeated calls and validates each occurrence with a cursor.
    Prefetch lookahead never advances the demand-call cursor.
    """

    def __init__(self, items: Iterable[T] = (), *, recorded: bool = False) -> None:
        """Create a construction order or a strict recorded sequence.

        Args:
            items: Initial module sequence.
            recorded: Preserve repeated occurrences and validate demand-call order.
                Otherwise, resynchronize the cursor when construction order differs
                from actual execution.
        """
        self._items: list[ref[T]] = []
        self._recorded = recorded
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append a module; repeated occurrences are allowed only in recorded order."""
        if not self._recorded and item in self:
            raise ValueError("Construction execution order does not support duplicate items.")
        self._items.append(ref(item))

    def __contains__(self, item: object) -> bool:
        """Return whether the sequence contains a live item."""
        return any(candidate is item for candidate in self)

    def __iter__(self) -> Iterator[T]:
        """Iterate over live occurrences in execution order."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def __reversed__(self) -> Iterator[T]:
        """Iterate over live occurrences in reverse order."""
        for item_ref in reversed(self._items):
            item = item_ref()
            if item is not None:
                yield item

    def _find_index(self, item: T) -> int:
        """Find a construction-order module by identity when the cursor differs."""
        for index, item_ref in enumerate(self._items):
            if item_ref() is item:
                return index
        raise KeyError(item)

    def advance(self, item: T) -> None:
        """Consume a demand call, resynchronizing only construction-order mismatches.

        Calls following the sequence advance in constant time. Construction-order
        mismatches search the sequence; recorded-order mismatches raise instead.
        """
        if self._items and self._items[self._position]() is item:
            index = self._position
        elif self._recorded:
            raise RuntimeError("FSDP module calls diverged from the recorded prefetch order.")
        else:
            index = self._find_index(item)
        self._position = (index + 1) % len(self._items)

    def next_item(self, item: T, offset: int = 1) -> T | None:
        """Look ahead without advancing the demand-call cursor.

        Recorded orders use the last consumed occurrence. Construction orders
        also support lookup from a different module without moving the cursor.
        """
        if not self._items:
            raise KeyError(item)
        index = (self._position - 1) % len(self._items)
        if not self._recorded and self._items[index]() is not item:
            index = self._find_index(item)
        next_index = index + offset
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()
