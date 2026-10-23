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
from weakref import WeakKeyDictionary, ref

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Look up unique modules and validate repeated module-call occurrences.

    A module appearing once can unambiguously select its position, preserving
    construction-order prefetch for reordered or repeated demand calls. Modules
    appearing more than once must match the next occurrence at the cursor.
    Prefetch lookahead never advances that cursor.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create a weak execution sequence, preserving repeated occurrences."""
        self._items: list[ref[T]] = []
        # An index identifies a unique item; None marks an ambiguous repeated item.
        self._unique_index: WeakKeyDictionary[T, int | None] = WeakKeyDictionary()
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append an occurrence and mark previously seen items as repeated."""
        self._unique_index[item] = None if item in self._unique_index else len(self._items)
        self._items.append(ref(item))

    def __iter__(self) -> Iterator[T]:
        """Iterate over live occurrences in execution order."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def advance(self, item: T) -> None:
        """Locate a unique demand call or consume the next expected repeated call."""
        index = self._unique_index.get(item)
        if index is None:
            if not self._items or self._items[self._position]() is not item:
                raise RuntimeError("FSDP module calls diverged from the expected prefetch order.")
            index = self._position
        self._position = (index + 1) % len(self._items)

    def next_item(self, offset: int = 1) -> T | None:
        """Look ahead from the last consumed occurrence without moving the cursor."""
        if not self._items:
            return None
        index = (self._position - 1) % len(self._items)
        next_index = index + offset
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()
