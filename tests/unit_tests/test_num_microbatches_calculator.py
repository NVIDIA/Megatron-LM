# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

import megatron.core.num_microbatches_calculator as mb_calculator


def test_init_num_microbatches_calculator():
    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    mb_calculator.init_num_microbatches_calculator(0, 32, 8, 2, False)
    assert mb_calculator.get_num_microbatches() == 2
    assert mb_calculator.get_current_global_batch_size() == 32

    with pytest.raises(AssertionError):
        mb_calculator.init_num_microbatches_calculator(0, 32, 8, 2, False)

    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    mb_calculator.init_num_microbatches_calculator(0, 32, 8, 3, True)
    assert mb_calculator.get_num_microbatches() == 1
    assert mb_calculator.get_current_global_batch_size() == 32
    assert mb_calculator.get_current_running_global_batch_size() == 24

    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    mb_calculator.init_num_microbatches_calculator(0, 33, 8, 2, True)
    assert mb_calculator.get_num_microbatches() == 2
    assert mb_calculator.get_current_global_batch_size() == 33
    assert mb_calculator.get_current_running_global_batch_size() == 32


def test_reconfigure_num_microbatches_calculator():
    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    mb_calculator.init_num_microbatches_calculator(0, 32, 8, 2, False)
    assert mb_calculator.get_num_microbatches() == 2
    assert mb_calculator.get_current_global_batch_size() == 32

    mb_calculator.reconfigure_num_microbatches_calculator(0, 16, 8, 2, False)
    assert mb_calculator.get_num_microbatches() == 1
    assert mb_calculator.get_current_global_batch_size() == 16


def test_get_num_microbatches():
    mb_calculator.reconfigure_num_microbatches_calculator(0, 16, 8, 2, False)
    assert mb_calculator.get_num_microbatches() == 1

    mb_calculator.reconfigure_num_microbatches_calculator(0, 16, 4, 3, True)
    assert mb_calculator.get_num_microbatches() == 1


def test_get_current_global_batch_size():
    mb_calculator.reconfigure_num_microbatches_calculator(0, 16, 4, 2, False)
    assert mb_calculator.get_current_global_batch_size() == 16

    mb_calculator.reconfigure_num_microbatches_calculator(0, 16, 4, 3, True)
    assert mb_calculator.get_current_global_batch_size() == 16
    assert mb_calculator.get_current_running_global_batch_size() == 12


def test_get_micro_batch_size():
    mb_calculator.reconfigure_num_microbatches_calculator(0, 16, 8, 2, False)
    assert mb_calculator.get_micro_batch_size() == 8


def test_build_num_microbatches_calculator():
    temp_calculator = mb_calculator._build_num_microbatches_calculator(0, 32, 8, 2, False)
    assert temp_calculator.get() == 2
    assert temp_calculator.get_current_global_batch_size() == 32
    assert type(temp_calculator) is mb_calculator.ConstantNumMicroBatchesCalculator

    with pytest.raises(ValueError):
        mb_calculator._build_num_microbatches_calculator(
            0, None, 8, 2, True, step_batch_size_schedule="0:16 100:32"
        )



class TestConstantNumMicroBatchesCalculator:
    def setup_method(self, method):
        self.mb_calculator = mb_calculator.ConstantNumMicroBatchesCalculator(32, 8, 2, False, 0)

    def test_constructor(self):
        assert type(self.mb_calculator) is mb_calculator.ConstantNumMicroBatchesCalculator
        assert self.mb_calculator.num_micro_batches == 2
        assert self.mb_calculator.current_global_batch_size == 32
        assert self.mb_calculator.micro_batch_size == 8

    def test_get(self):
        assert self.mb_calculator.get() == 2

    def test_get_current_global_batch_size(self):
        assert self.mb_calculator.get_current_global_batch_size() == 32


def test_step_batch_size_schedule_allows_past_entries_smaller_than_dp():
    """Test that step batch size schedule does not crash when early schedule entries
    are smaller than micro_batch_size * data_parallel_size.

    This happens when scaling to more GPUs mid-training: the initial batch sizes
    in the schedule are smaller than the new GPU count can support, but training
    has progressed past those entries. The divisibility check should only apply
    to the CURRENT batch size (after checkpoint loading), not all schedule entries.
    """
    # micro=1, dp=512, micro*dp=512
    # Schedule starts at 256 which is < 512, but later entries are fine.
    # This should NOT raise during construction.
    calc = mb_calculator.StepBatchsizeNumMicroBatchesCalculator(
        micro_batch_size=1,
        data_parallel_size=512,
        decrease_batch_size_if_needed=False,
        rank=0,
        schedule="0:256 200000:512 600000:1024 1500000:2048 3000000:4096 6000000:6144",
    )

    # At init (consumed_samples=0), batch=256 < micro*dp=512.
    # No consistency_check at init, so no error.
    assert calc.current_global_batch_size == 256

    # After training past the first entry, batch=512. Consistency check passes.
    calc.update(200000, consistency_check=True)
    assert calc.current_global_batch_size == 512
    assert calc.num_micro_batches == 1

    # Later entries work fine.
    calc.update(1500000, consistency_check=True)
    assert calc.current_global_batch_size == 2048
    assert calc.num_micro_batches == 4

    calc.update(6000000, consistency_check=True)
    assert calc.current_global_batch_size == 6144
    assert calc.num_micro_batches == 12


def test_step_batch_size_schedule_consistency_check_fails_when_batch_too_small():
    """Test that consistency_check=True raises when current batch size is smaller
    than micro_batch_size * data_parallel_size.

    This is the scenario caught by the runtime check in setup_model_and_optimizer:
    GPUs were scaled up but the current schedule entry yields a batch size too small
    for the new world size.
    """
    calc = mb_calculator.StepBatchsizeNumMicroBatchesCalculator(
        micro_batch_size=1,
        data_parallel_size=512,
        decrease_batch_size_if_needed=False,
        rank=0,
        schedule="0:256 200000:512 600000:1024",
    )

    # At consumed_samples=0, batch=256 < micro*dp=512.
    # consistency_check=True should fail because 256 % 512 != 0.
    with pytest.raises(AssertionError):
        calc.update(0, consistency_check=True)


def test_step_batch_size_schedule_rejects_decrease_batch_size_if_needed():
    with pytest.raises(ValueError):
        mb_calculator.StepBatchsizeNumMicroBatchesCalculator(
            micro_batch_size=1,
            data_parallel_size=512,
            decrease_batch_size_if_needed=True,
            rank=0,
            schedule="0:256 200000:512 600000:1024",
        )
