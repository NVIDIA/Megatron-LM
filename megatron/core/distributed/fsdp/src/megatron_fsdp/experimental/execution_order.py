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
from typing import TYPE_CHECKING, Generic, Literal, TypeVar
from weakref import ref

if TYPE_CHECKING:
    from .module import FsdpModule

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

    def reset_cursor(self, position: int = 0) -> None:
        """Reset traversal position without changing the recorded sequence."""
        self._position = position

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
            raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
        item = self._items[self._position]
        self._position += 1
        return item

    def __getitem__(self, index: int) -> T:
        """Read an occurrence without changing execution progress."""
        return self._items[index]


ExecutionPhase = Literal["forward", "backward", "recompute"]
ExecutionEvent = tuple[ref["FsdpModule"], ExecutionPhase]


class IterationOrder(ExecutionOrder[ExecutionEvent]):
    """Select static, recorded, or demand-only prefetch for a shared context.

    ``mode`` selects the order source. In ``record_replay`` mode, ``captured``
    distinguishes initial recording from replay of the completed sequence.
    ``active`` tracks the iteration scope independently of the selected strategy.
    State lives on FsdpContext because autograd hooks can run on another thread.
    The scheduler owns iteration boundaries and must explicitly recapture when
    the training schedule changes.
    """

    def __init__(self) -> None:
        super().__init__()
        # Only construction-defined orders have separate F/B sections.
        self.backward_start = 0
        self.captured = False
        self.active = False
        self.mode: Literal["static", "record_replay", "demand"] = "static"

    def append(self, item: ExecutionEvent) -> None:
        """Append an occurrence before the recorded sequence is complete."""
        assert not self.captured, "Cannot append to an already captured order."
        super().append(item)

    @property
    def recording(self) -> bool:
        return self.active and self.mode == "record_replay" and not self.captured
