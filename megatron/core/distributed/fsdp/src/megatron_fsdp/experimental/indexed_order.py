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

from collections.abc import Iterator
from typing import Generic, TypeVar
from weakref import WeakKeyDictionary, ref

T = TypeVar("T")


class IndexedOrder(Generic[T]):
    """Insertion order with weakly held items and successor lookup."""

    def __init__(self) -> None:
        """Create an empty indexed order."""
        self._items: list[ref[T]] = []
        self._index_by_item: WeakKeyDictionary[T, int] = WeakKeyDictionary()

    def append(self, item: T) -> None:
        """Append ``item`` to the order.

        Args:
            item: Item to append.

        Raises:
            ValueError: If ``item`` is already present in the order.
        """
        if item in self._index_by_item:
            raise ValueError("IndexedOrder does not support duplicate items.")
        self._index_by_item[item] = len(self._items)
        self._items.append(ref(item))

    def __iter__(self) -> Iterator[T]:
        """Iterate over live items in order."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def next_item(self, item: T) -> T | None:
        """Return the live item that follows ``item``, if any."""
        index = self._index_by_item[item]
        next_index = index + 1
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()
