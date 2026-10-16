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

"""A weak occurrence sequence with independent caller-owned iterators."""

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar
from weakref import ref

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Store occurrences without identity lookup or implicit cursor resets.

    Callers can traverse the same sequence independently for execution and
    prefetch. A missing item is an error: skipping it would change occurrence
    positions. Appends are visible to an iterator used during recording.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items: list[ref[T]] = [ref(item) for item in items]

    def append(self, item: T) -> None:
        """Append one weakly held occurrence, preserving repetitions."""
        self._items.append(ref(item))

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        """Traverse occurrences strictly without consuming another iterator."""
        for item_ref in self._items:
            item = item_ref()
            if item is None:
                raise RuntimeError("A module in the recorded FSDP execution order was destroyed.")
            yield item
