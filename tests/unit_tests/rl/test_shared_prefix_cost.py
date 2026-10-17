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

import math
import random

import pytest

from megatron.rl.shared_prefix_cost import estimate_shared_prefix_row_work


def _slot_work(lengths, prefixes, *, physical_weight=4, expanded_weight=1):
    """Executed work of one slot: one prompt copy when its rows can share."""
    expanded = sum(lengths)
    shares = len(lengths) > 1 and len(set(prefixes)) == 1
    backbone = expanded - (len(lengths) - 1) * prefixes[0] if shares else expanded
    return physical_weight * backbone + expanded_weight * expanded


def test_clean_group_weights_one_prompt_copy_per_group():
    work = estimate_shared_prefix_row_work(
        group_ids=["a", "a", "b", "b"], sequence_lengths=[5, 7, 4, 9], prompt_lengths=[2, 2, 3, 3]
    )
    # Group size 2 scales every weight: 2 * (4 * star backbone + 1 * expanded).
    assert work == [42, 62, 28, 78]
    assert sum(work[:2]) == 2 * _slot_work([5, 7], [2, 2])
    assert sum(work[2:]) == 2 * _slot_work([4, 9], [3, 3])


def test_outlier_group_is_balanced_as_dense_work():
    work = estimate_shared_prefix_row_work(
        group_ids=["a", "a"], sequence_lengths=[5, 7], prompt_lengths=[2, 3]
    )
    assert work == [2 * 5 * 5, 2 * 5 * 7]


def test_single_completion_groups_have_no_sharing_credit():
    work = estimate_shared_prefix_row_work(
        group_ids=["a", "b"], sequence_lengths=[5, 9], prompt_lengths=[2, 4]
    )
    assert work == [5 * 5, 5 * 9]


def test_every_execution_slot_stores_its_own_prompt_copy():
    lengths, prompts = [5, 5, 5, 5], [2, 2, 2, 2]
    one_slot = estimate_shared_prefix_row_work(
        group_ids=["a"] * 4, sequence_lengths=lengths, prompt_lengths=prompts
    )
    two_slots = estimate_shared_prefix_row_work(
        group_ids=["a"] * 4,
        sequence_lengths=lengths,
        prompt_lengths=prompts,
        row_slot_ids=[0, 0, 1, 1],
    )
    assert sum(one_slot) == 4 * _slot_work(lengths, prompts)
    assert sum(two_slots) == 2 * 2 * _slot_work([5, 5], [2, 2])
    # Two prompt copies cost more backbone work than one.
    assert sum(two_slots) / 2 > sum(one_slot) / 4


def test_slot_weights_are_proportional_to_executed_work():
    rng = random.Random(0)
    for _ in range(200):
        group_size = rng.randint(1, 6)
        groups = [f"g{index}" for index in range(rng.randint(1, 4)) for _ in range(group_size)]
        prompts, lengths, slots = [], [], []
        for group in dict.fromkeys(groups):
            prompt = rng.randint(0, 5)
            num_slots = rng.randint(1, group_size)
            for row in range(group_size):
                prompts.append(prompt if rng.random() > 0.1 else rng.randint(0, 5))
                lengths.append(prompts[-1] + rng.randint(0, 9))
                slots.append(row % num_slots)
        work = estimate_shared_prefix_row_work(
            group_ids=groups, sequence_lengths=lengths, prompt_lengths=prompts, row_slot_ids=slots
        )
        assert all(value >= 0 for value in work)
        members = {}
        for index, key in enumerate(zip(groups, slots)):
            members.setdefault(key, []).append(index)
        scale = math.lcm(*(len(indices) for indices in members.values()))
        for indices in members.values():
            expected = _slot_work([lengths[i] for i in indices], [prompts[i] for i in indices])
            assert sum(work[i] for i in indices) == scale * expected


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(group_ids=["a", "a", "b"]), "equal-size"),
        (dict(row_slot_ids=[0]), "one entry per row"),
        (dict(row_slot_ids=[0, -1, 0]), "nonnegative integers"),
        (dict(group_ids=["a", "", "a"]), "Invalid prompt-group identity"),
    ],
)
def test_estimator_rejects_invalid_metadata(kwargs, message):
    arguments = dict(
        group_ids=["a", "a", "a"], sequence_lengths=[3, 3, 3], prompt_lengths=[1, 1, 1]
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        estimate_shared_prefix_row_work(**arguments)
