# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Estimate shared backbone plus expanded work for group assignment only.

The existing group-coherent sharder can use these integer weights while its
packing planner continues to receive real sequence lengths. Group sums are
proportional to physical_weight * star_backbone_tokens + expanded_weight *
expanded_tokens. This is an estimate: it excludes deeper response-prefix
sharing, packing padding, attention geometry and communication costs.
"""

from collections.abc import Sequence


def estimate_shared_prefix_row_work(
    *,
    group_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    prompt_lengths: Sequence[int],
    physical_weight: int = 4,
    expanded_weight: int = 1,
) -> list[int]:
    """Keep both backbone and repeated MTP work in the balancing objective."""
    if not (len(group_ids) == len(sequence_lengths) == len(prompt_lengths)):
        raise ValueError("Group, sequence and prefix metadata lengths differ")
    if not group_ids:
        raise ValueError("Cannot estimate empty work")
    if any(type(w) is not int or w < 0 for w in (physical_weight, expanded_weight)):
        raise ValueError("Work weights must be nonnegative integers")
    if physical_weight + expanded_weight == 0:
        raise ValueError("At least one work component must be enabled")
    groups: dict[str, list[int]] = {}
    for index, (group, length, prefix) in enumerate(
        zip(group_ids, sequence_lengths, prompt_lengths, strict=True)
    ):
        if not isinstance(group, str) or not group:
            raise ValueError("Invalid prompt-group identity")
        if type(length) is not int or type(prefix) is not int or not 0 <= prefix <= length:
            raise ValueError("Prefix must fit within the real sequence")
        groups.setdefault(group, []).append(index)
    sizes = {len(indices) for indices in groups.values()}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError("Complete equal-size rollout groups are required")
    size = next(iter(sizes))
    # Capture may publish masked placeholders with no prompt, or verified rows
    # with different boundaries. Those groups cannot share a star. Balance their
    # full dense work while leaving the actual planner metadata untouched.
    cost_prompt_lengths = list(prompt_lengths)
    for indices in groups.values():
        if len({prompt_lengths[index] for index in indices}) != 1:
            for index in indices:
                cost_prompt_lengths[index] = 0
    # Multiply by group size to distribute the shared prefix without rounding.
    # A common positive factor leaves the existing greedy sharder unchanged.
    return [
        size * (physical_weight + expanded_weight) * length - physical_weight * (size - 1) * prefix
        for length, prefix in zip(sequence_lengths, cost_prompt_lengths, strict=True)
    ]
