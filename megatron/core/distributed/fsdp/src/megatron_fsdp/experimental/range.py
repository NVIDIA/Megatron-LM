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

"""Contiguous element ranges and intersection helpers."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Range:
    """Contiguous element range in a caller-defined coordinate system."""

    start: int
    numel: int

    @property
    def end(self) -> int:
        """Exclusive end of the range."""
        return self.start + self.numel


def intersect_ranges(first: Range, second: Range) -> Range:
    """Intersect two element ranges in the same coordinate system.

    Return the later starting offset and the overlap length, or zero length when
    the ranges do not overlap.
    """
    start = max(first.start, second.start)
    end = min(first.end, second.end)
    return Range(start, max(0, end - start))
