# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checks for the training loop's batch-size schedule transition decision."""

import pytest

from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_current_global_batch_size,
    get_num_microbatches,
    init_num_microbatches_calculator,
    update_num_microbatches,
)
from megatron.training.training import is_batch_size_transition


@pytest.fixture
def step_schedule():
    """Start a step schedule at global batch size 8 with one sample per data-parallel rank."""

    def start(schedule):
        init_num_microbatches_calculator(
            rank=0, micro_batch_size=1, data_parallel_size=8, step_batch_size_schedule=schedule
        )
        return get_num_microbatches(), get_current_global_batch_size()

    yield start
    destroy_num_microbatches_calculator()


def test_transition_detected_when_rounded_microbatch_count_is_unchanged(step_schedule):
    num_microbatches, global_batch_size = step_schedule("0:8 8:12")
    update_num_microbatches(8, consistency_check=False)
    # The unchecked update rounds 12 / 8 down to the old count of one microbatch.
    assert get_num_microbatches() == num_microbatches
    assert get_current_global_batch_size() == 12
    assert is_batch_size_transition(num_microbatches, global_batch_size, True)
    # The consistency check that follows would reject 12 at data-parallel size 8.
    with pytest.raises(AssertionError):
        update_num_microbatches(8, consistency_check=True)


def test_transition_detected_on_divisible_schedule_step(step_schedule):
    num_microbatches, global_batch_size = step_schedule("0:8 8:16")
    update_num_microbatches(8, consistency_check=False)
    assert get_num_microbatches() == 2
    assert is_batch_size_transition(num_microbatches, global_batch_size, True)
    assert is_batch_size_transition(num_microbatches, global_batch_size, False)


def test_no_transition_before_schedule_step(step_schedule):
    num_microbatches, global_batch_size = step_schedule("0:8 8:12")
    update_num_microbatches(4, consistency_check=False)
    assert not is_batch_size_transition(num_microbatches, global_batch_size, True)


def test_global_batch_size_ignored_without_step_schedule(step_schedule):
    num_microbatches, global_batch_size = step_schedule("0:8 8:12")
    update_num_microbatches(8, consistency_check=False)
    # RL without sequence packing resizes the global batch every step without a schedule.
    assert not is_batch_size_transition(num_microbatches, global_batch_size, False)
