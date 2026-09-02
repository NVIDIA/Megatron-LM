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
