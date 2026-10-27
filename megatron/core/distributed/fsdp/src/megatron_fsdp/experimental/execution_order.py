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

import enum
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Generic, TypeVar
from weakref import ref

from .phase import Phase

if TYPE_CHECKING:
    from .module import FsdpModule

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Look up unique modules and validate repeated module-call occurrences.

    A module appearing once can unambiguously select its position, preserving
    construction-order prefetch for reordered or repeated demand calls. Modules
    appearing more than once must match the next occurrence at the cursor.
    Prefetch lookahead never advances that cursor.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[T] = []
        # An index identifies a unique item; None marks an ambiguous repeated item.
        self._unique_index: dict[T, int | None] = {}
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append an occurrence and mark previously seen items as repeated."""
        self._unique_index[item] = None if item in self._unique_index else len(self._items)
        self._items.append(item)

    def __iter__(self) -> Iterator[T]:
        """Iterate over occurrences in execution order."""
        return iter(self._items)

    def reset_cursor(self, position: int = 0) -> None:
        """Reset traversal without changing the recorded sequence."""
        self._position = position

    def __len__(self) -> int:
        return len(self._items)

    @property
    def position(self) -> int:
        """Index of the next occurrence; len(order) means traversal is exhausted."""
        return self._position

    def __getitem__(self, index: int) -> T:
        """Read an occurrence without advancing the cursor."""
        return self._items[index]

    def advance(self, item: T | None = None) -> T:
        """Locate a unique demand call or consume the next expected repeated call.

        With no item, consume the next occurrence without lookup.
        """
        if item is not None:
            index = self._unique_index.get(item)
            if index is None:
                if not self._items or self._items[self._position % len(self._items)] != item:
                    raise RuntimeError(
                        "FSDP module calls diverged from the expected prefetch order."
                    )
                index = self._position % len(self._items)
            self._position = index
        if self._position >= len(self._items):
            raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
        result = self._items[self._position]
        self._position += 1
        return result


class PrefetchMode(enum.Enum):
    """Source of the parameter prefetch order.

    PRESET uses a phase-tagged forward/backward order defined during construction,
    without requiring an iteration scope. Prefetch stays within each phase;
    recomputation gathers on demand.
    RECORD_REPLAY applies only inside training iteration scopes. It records
    phase-tagged module calls in the first scope and prefetches from that order
    in later scopes. Recording gathers on demand. Scope exit switches to DEMAND,
    retaining the captured order for the next training scope.
    DEMAND gathers parameters only when needed, without recording, replaying, or
    prefetching. Forward-only callers temporarily use this mode so validation does
    not consume the training order.
    """

    PRESET = enum.auto()
    RECORD_REPLAY = enum.auto()
    DEMAND = enum.auto()


ExecutionEvent = tuple[ref["FsdpModule"], Phase]
