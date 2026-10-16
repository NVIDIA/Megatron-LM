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

"""A weak occurrence sequence with independent execution and prefetch cursors."""

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar
from weakref import ref

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Track execution and prefetch independently without identity lookup.

    Execution starts a new pass after consuming the complete sequence; prefetch
    never wraps into the next pass. Ordinary iteration does not move either
    cursor. Missing owners are errors rather than skipped occurrences.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items: list[ref[T]] = [ref(item) for item in items]
        self.reset()

    @property
    def position(self) -> int:
        """Number of execution occurrences consumed in the current pass."""
        return self._execution_position

    def reset(self) -> None:
        """Start a fresh execution and prefetch traversal."""
        self._execution_position = self._prefetch_position = 0

    def advance(self) -> T:
        """Consume the next execution occurrence, restarting completed passes."""
        if not self._items:
            raise RuntimeError("The FSDP execution order is empty.")
        if self._execution_position == len(self):
            self.reset()
        item = self._dereference(self._items[self._execution_position])
        self._execution_position += 1
        return item

    def next_prefetch(self) -> T | None:
        """Consume one upcoming occurrence without advancing execution."""
        self._prefetch_position = max(self._prefetch_position, self._execution_position)
        if self._prefetch_position == len(self):
            return None
        item = self._dereference(self._items[self._prefetch_position])
        self._prefetch_position += 1
        return item

    def append(self, item: T) -> None:
        """Append one weakly held occurrence, preserving repetitions."""
        self._items.append(ref(item))

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _dereference(item_ref: ref[T]) -> T:
        item = item_ref()
        if item is None:
            raise RuntimeError("A module in the FSDP execution order was destroyed.")
        return item

    def __iter__(self) -> Iterator[T]:
        """Traverse weak occurrences without moving or retaining cursor items."""
        return map(self._dereference, self._items)
