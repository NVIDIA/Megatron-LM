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

"""Generic execution sequences with occurrence-based traversal."""

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Consume ordered occurrences with an explicitly reset execution cursor.

    Repeated items are ordinary occurrences. The caller validates each item
    returned by advance(); lookahead never changes execution progress. Callers
    own item lifetimes and can store weak references in their payloads.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[T] = list(items)
        self._position = 0

    def append(self, item: T) -> None:
        """Append one occurrence."""
        self._items.append(item)

    def reset(self) -> None:
        """Start a new traversal."""
        self._position = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def position(self) -> int:
        """Number of occurrences consumed in the current traversal."""
        return self._position

    def __iter__(self) -> Iterator[T]:
        """Iterate over all occurrences without changing execution progress."""
        return iter(self._items)

    def advance(self) -> T:
        """Consume the next occurrence without lookup or automatic wrapping."""
        if self._position >= len(self._items):
            raise RuntimeError("Calls exceeded the execution order; reset it first.")
        item = self._items[self._position]
        self._position += 1
        return item

    def next_item(self, offset: int = 1) -> T | None:
        """Look ahead from the last consumed occurrence without moving the cursor."""
        next_index = self._position - 1 + offset
        if next_index < 0 or next_index >= len(self._items):
            return None
        return self._items[next_index]
