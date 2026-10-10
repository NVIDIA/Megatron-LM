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

"""Ordered sequence with indexed item lookup."""

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar
from weakref import WeakKeyDictionary, ref

T = TypeVar("T")


class IndexedOrder(Generic[T]):
    """Insertion order with weakly held items and successor lookup."""

    def __init__(self, items: Iterable[T] | None = None) -> None:
        """Create a static order or replay a recorded sequence of occurrences.

        Without ``items``, append unique items and look up successors by identity.
        With ``items``, preserve duplicates and consume one occurrence per
        ``next_items`` call, restarting after the complete sequence is consumed.
        """
        self._items: list[ref[T]] = []
        self._index_by_item: WeakKeyDictionary[T, int] = WeakKeyDictionary()
        self._position: int | None = None
        if items is not None:
            self._items = [ref(item) for item in items]
            self._position = 0

    def append(self, item: T) -> None:
        """Append ``item`` to the order.

        Args:
            item: Item to append.

        Raises:
            ValueError: If ``item`` is already present in the order.
        """
        if self._position is not None:
            raise ValueError("Cannot append to a recorded order.")
        if item in self._index_by_item:
            raise ValueError("IndexedOrder does not support duplicate items.")
        self._index_by_item[item] = len(self._items)
        self._items.append(ref(item))

    def __iter__(self) -> Iterator[T]:
        """Iterate over live items in order, including repeated occurrences."""
        return self._live_items(0)

    def _live_items(self, start: int) -> Iterator[T]:
        for index in range(start, len(self._items)):
            item = self._items[index]()
            if item is not None:
                yield item

    def next_items(self, item: T) -> Iterator[T]:
        """Consume ``item`` and return its successors within this execution.

        A recorded sequence validates each call before advancing its position.
        Static orders retain identity-based lookup independent of call order.
        """
        if self._position is None:
            index = self._index_by_item[item]
        else:
            index = self._position
            if not self._items or self._items[index]() is not item:
                raise RuntimeError("FSDP module calls diverged from the recorded prefetch order.")
            self._position = (index + 1) % len(self._items)
        return self._live_items(index + 1)
