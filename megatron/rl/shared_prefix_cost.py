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
packing planner continues to receive real sequence lengths. Work is
proportional to physical_weight * star_backbone_tokens + expanded_weight *
expanded_tokens, summed over execution slots: every slot of a group runs its
own star with its own prompt copy, so a group split into K slots stores its
prompt K times. This is an estimate: it excludes deeper response-prefix
sharing, packing padding, attention geometry and communication costs.
"""

import math
from collections.abc import Sequence


def estimate_shared_prefix_row_work(
    *,
    group_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    prompt_lengths: Sequence[int],
    physical_weight: int = 4,
    expanded_weight: int = 1,
    row_slot_ids: Sequence[int] | None = None,
) -> list[int]:
    """Keep both backbone and repeated MTP work in the balancing objective.

    Args:
        group_ids: Prompt-group identity per row. Groups must have equal sizes.
        sequence_lengths: Unpadded prompt-plus-completion length per row.
        prompt_lengths: Prompt length per row.
        physical_weight: Weight of backbone tokens, which store a prompt once
            per shared slot.
        expanded_weight: Weight of expanded (per-row, MTP) tokens.
        row_slot_ids: Optional group-local execution slot per row, as planned by
            ``plan_fixed_execution_slots``. ``None`` treats each group as one
            slot.

    Returns:
        One nonnegative integer weight per row. All weights share one positive
        scale factor, so only their relative values are meaningful.
    """
    if not (len(group_ids) == len(sequence_lengths) == len(prompt_lengths)):
        raise ValueError("Group, sequence and prefix metadata lengths differ")
    if not group_ids:
        raise ValueError("Cannot estimate empty work")
    if row_slot_ids is not None and len(row_slot_ids) != len(group_ids):
        raise ValueError("Execution slot IDs must have one entry per row")
    if any(type(w) is not int or w < 0 for w in (physical_weight, expanded_weight)):
        raise ValueError("Work weights must be nonnegative integers")
    if physical_weight + expanded_weight == 0:
        raise ValueError("At least one work component must be enabled")
    groups: dict[str, list[int]] = {}
    slots: dict[tuple[str, int], list[int]] = {}
    for index, (group, length, prefix) in enumerate(
        zip(group_ids, sequence_lengths, prompt_lengths, strict=True)
    ):
        if not isinstance(group, str) or not group:
            raise ValueError("Invalid prompt-group identity")
        if type(length) is not int or type(prefix) is not int or not 0 <= prefix <= length:
            raise ValueError("Prefix must fit within the real sequence")
        slot = 0 if row_slot_ids is None else row_slot_ids[index]
        if type(slot) is not int or slot < 0:
            raise ValueError("Execution slot IDs must be nonnegative integers")
        groups.setdefault(group, []).append(index)
        slots.setdefault((group, slot), []).append(index)
    if len({len(indices) for indices in groups.values()}) != 1:
        raise ValueError("Complete equal-size rollout groups are required")
    # Capture may publish masked placeholders with no prompt, or verified rows
    # with different boundaries. Those slots cannot share a star. Balance their
    # full dense work while leaving the actual planner metadata untouched. A
    # single-row slot has no peer and shares nothing either.
    shared_prompt_lengths = [0] * len(group_ids)
    for indices in slots.values():
        if len(indices) > 1 and len({prompt_lengths[index] for index in indices}) == 1:
            for index in indices:
                shared_prompt_lengths[index] = prompt_lengths[index]
    # A slot of n rows stores its prompt once instead of n times. Spread the
    # (n - 1) * prefix credit evenly over its rows; a common scale divisible by
    # every slot size keeps the weights integral and the greedy sharder unchanged.
    scale = math.lcm(*(len(indices) for indices in slots.values()))
    slot_sizes = [0] * len(group_ids)
    for indices in slots.values():
        for index in indices:
            slot_sizes[index] = len(indices)
    return [
        scale * (physical_weight + expanded_weight) * length
        - physical_weight * (size - 1) * prefix * (scale // size)
        for length, prefix, size in zip(
            sequence_lengths, shared_prompt_lengths, slot_sizes, strict=True
        )
    ]
