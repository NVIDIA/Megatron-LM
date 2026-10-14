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
    """Track module-call occurrences with weak references and a strict cursor.

    Repeated modules occupy distinct positions. Every demand call must match
    the next occurrence; prefetch lookahead never advances the cursor.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[ref[T]] = [ref(item) for item in items]
        self._position = 0

    def append(self, item: T) -> None:
        """Append one occurrence, including repeated calls to the same module."""
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

    def advance(self, item: T) -> None:
        """Consume the next expected call, restarting after the sequence ends."""
        if not self._items or self._items[self._position]() is not item:
            raise RuntimeError("FSDP module calls diverged from the expected prefetch order.")
        self._position = (self._position + 1) % len(self._items)

    def next_item(self, offset: int = 1) -> T | None:
        """Look ahead from the last consumed occurrence without moving the cursor."""
        if not self._items:
            return None
        index = (self._position - 1) % len(self._items)
        next_index = index + offset
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()
