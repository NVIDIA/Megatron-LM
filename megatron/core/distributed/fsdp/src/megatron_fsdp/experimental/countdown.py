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

"""A self-resetting countdown."""

from collections.abc import Hashable, Mapping


class Countdown:
    """Countdown that automatically re-arms after reaching zero."""

    def __init__(self, initial_value: int) -> None:
        """Create a countdown starting at ``initial_value``."""
        if initial_value < 0:
            raise ValueError(f"Countdown initial_value must be non-negative, got {initial_value}.")
        self._initial_value = initial_value
        self._value = initial_value

    @property
    def initial_value(self) -> int:
        """Return the number of decrements in one countdown cycle."""
        return self._initial_value

    def decrement(self) -> bool:
        """Decrement and return whether the countdown completed this call."""
        self._value -= 1
        completed = self._value == 0
        if completed:
            self._value = self._initial_value
        return completed


class MultiplicityReadiness:
    """Readiness tracker that becomes ready after a specified multiplicity of events."""

    def __init__(self, expected: Mapping[Hashable, int]) -> None:
        """Track completion against ``expected``, the per-key multiplicity."""
        self._expected = dict(expected)
        self._counts: dict[Hashable, int] = {key: 0 for key in expected}

    @property
    def expected_total(self) -> int:
        """Return the total number of marks a correct window must produce."""
        return sum(self._expected.values())

    @property
    def expected(self) -> Mapping[Hashable, int]:
        """Return the expected multiplicity for each key."""
        return self._expected

    def mark(self, key: Hashable) -> None:
        """Record one contribution of ``key``.

        An unknown key is a programming error in the caller's key space, not
        something to ignore: silently dropping it would hide a real contribution
        from the accounting. A count above the declared multiplicity is recorded as
        an over-fire, which is the signal that the multiplicity was under-declared.
        """
        if key not in self._counts:
            raise KeyError(f"{key!r} is not a known parameter of this unit.")
        self._counts[key] += 1
        if self._counts[key] > self._expected[key]:
            raise ValueError(f"{key!r} over-fired: {self._counts[key]} > {self._expected[key]}")

    def missing(self) -> frozenset[Hashable]:
        """Return the keys that have contributed fewer times than expected."""
        return frozenset(key for key, count in self._counts.items() if count < self._expected[key])

    def seal(self) -> None:
        """Check the window at its boundary, raising for any part-way counted key.

        A clean window -- never started, or complete and already closed by the
        finalize path -- seals silently; a key with ``0 < count < expected`` is a
        contribution in flight that must not slide into the next window.

        Call this at the per-step gradient boundary; the production seam is the
        MFSDP v2 adapter's ``finish_grad_sync``.
        """
        partial = {
            key: (count, self._expected[key])
            for key, count in self._counts.items()
            if 0 < count < self._expected[key]
        }
        if partial:
            detail = ", ".join(
                f"{key!r}: {count}/{expected}" for key, (count, expected) in partial.items()
            )
            raise ValueError(f"gradient window sealed mid-flight: {detail}")

    def is_complete(self) -> bool:
        """Return whether every key reached its declared multiplicity."""
        return not self.missing()

    def reset(self) -> None:
        """Reset the readiness tracker."""
        for key in self._counts:
            self._counts[key] = 0
