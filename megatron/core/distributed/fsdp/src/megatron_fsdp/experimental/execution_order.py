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
from typing import TYPE_CHECKING, Generic, TypeVar, cast
from weakref import ref

if TYPE_CHECKING:
    from .module import FsdpModule

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Consume weak occurrences with an explicitly reset execution cursor.

    Repeated items are ordinary occurrences. The caller validates each item
    returned by advance(); lookahead never changes execution progress.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create a weak execution sequence, preserving repeated occurrences."""
        self._items: list[ref[T]] = [ref(item) for item in items]
        self._position = 0

    def append(self, item: T) -> None:
        """Append one weakly held occurrence."""
        self._items.append(ref(item))

    def reset(self) -> None:
        """Start a new traversal; root hooks own static traversal boundaries."""
        self._position = 0

    def __iter__(self) -> Iterator[T]:
        """Iterate over live occurrences in execution order."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def advance(self) -> T:
        """Consume the next occurrence without lookup or automatic wrapping."""
        if self._position >= len(self._items):
            raise RuntimeError("FSDP calls exceeded the static execution order; reset it first.")
        item = self._items[self._position]()
        if item is None:
            raise RuntimeError("A module in the FSDP execution order was destroyed.")
        self._position += 1
        return item

    def prefetch(self, prefetch_size: int | None = None) -> Iterator[T]:
        """Yield successors without advancing execution.

        None selects one occurrence; otherwise select occurrences until their
        parameter count reaches the budget or the order ends. Zero disables
        prefetch. Repeated modules retain their place in the sequence.
        """
        prefetched_size = 0
        for index in range(self._position, len(self._items)):
            if prefetch_size is not None and prefetched_size >= prefetch_size:
                return
            item = self._items[index]()
            if item is None:
                return
            yield item
            if prefetch_size is None:
                return
            prefetched_size += cast("FsdpModule", item).num_parameter_elements
