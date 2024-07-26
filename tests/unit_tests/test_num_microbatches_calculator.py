from typing import List, Optional

import pytest

import megatron.core.num_microbatches_calculator as mb_calculator


def test_init_num_microbatches_calculator():
    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    mb_calculator.init_num_microbatches_calculator(0, None, 32, 8, 2)
    assert mb_calculator.get_num_microbatches() == 2
    assert mb_calculator.get_current_global_batch_size() == 32

    with pytest.raises(AssertionError):
        mb_calculator.init_num_microbatches_calculator(0, None, 32, 8, 2)


def test_reconfigure_num_microbatches_calculator():
    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    mb_calculator.init_num_microbatches_calculator(0, None, 32, 8, 2)
    assert mb_calculator.get_num_microbatches() == 2
    assert mb_calculator.get_current_global_batch_size() == 32

    mb_calculator.reconfigure_num_microbatches_calculator(0, None, 16, 8, 2)
    assert mb_calculator.get_num_microbatches() == 1
    assert mb_calculator.get_current_global_batch_size() == 16

    mb_calculator.reconfigure_num_microbatches_calculator(0, [16, 16, 96], 32, 8, 2)
    assert mb_calculator.get_num_microbatches() == 1
    assert mb_calculator.get_current_global_batch_size() == 16


def test_get_num_microbatches():
    mb_calculator.reconfigure_num_microbatches_calculator(0, None, 16, 8, 2)
    assert mb_calculator.get_num_microbatches() == 1


def test_get_current_global_batch_size():
    mb_calculator.reconfigure_num_microbatches_calculator(0, None, 16, 8, 2)
    assert mb_calculator.get_current_global_batch_size() == 16


def test_get_micro_batch_size():
    mb_calculator.reconfigure_num_microbatches_calculator(0, None, 16, 8, 2)
    assert mb_calculator.get_micro_batch_size() == 8


def test_update_num_microbatches():
    mb_calculator.reconfigure_num_microbatches_calculator(0, [16, 8, 96], 32, 4, 2)
    assert mb_calculator.get_num_microbatches() == 2
    mb_calculator.update_num_microbatches(48, False)
    assert mb_calculator.get_num_microbatches() == 3

    mb_calculator.reconfigure_num_microbatches_calculator(0, [16, 8, 96], 32, 8, 2)
    with pytest.raises(AssertionError):
        mb_calculator.update_num_microbatches(49, True)

    mb_calculator.reconfigure_num_microbatches_calculator(0, None, 32, 8, 2)
    mb_calculator.update_num_microbatches(16)
    assert mb_calculator.get_num_microbatches() == 2


def test_build_num_microbatches_calculator():
    temp_calculator = mb_calculator.build_num_microbatches_calculator(0, None, 32, 8, 2)
    assert temp_calculator.get() == 2
    assert temp_calculator.get_current_global_batch_size() == 32
    assert type(temp_calculator) is mb_calculator.ConstantNumMicroBatchesCalculator

    temp_calculator = mb_calculator.build_num_microbatches_calculator(0, [16, 16, 48], 32, 8, 2)
    assert temp_calculator.get() == 1
    assert temp_calculator.get_current_global_batch_size() == 16
    assert type(temp_calculator) is mb_calculator.RampupBatchsizeNumMicroBatchesCalculator


class TestConstantNumMicroBatchesCalculator:
    def setup_method(self, method):
        self.mb_calculator = mb_calculator.ConstantNumMicroBatchesCalculator(32, 8, 2)

    def test_constructor(self):
        assert type(self.mb_calculator) is mb_calculator.ConstantNumMicroBatchesCalculator
        assert self.mb_calculator.num_micro_batches == 2
        assert self.mb_calculator.current_global_batch_size == 32
        assert self.mb_calculator.micro_batch_size == 8

    def test_get(self):
        assert self.mb_calculator.get() == 2

    def test_get_current_global_batch_size(self):
        assert self.mb_calculator.get_current_global_batch_size() == 32


class TestRampupBatchsizeNumMicroBatchesCalculator:
    def setup_method(self, method):
        self.mb_calculator = mb_calculator.RampupBatchsizeNumMicroBatchesCalculator(
            32, 8, 2, 16, 16, 48
        )

    def test_constructor(self):
        assert type(self.mb_calculator) is mb_calculator.RampupBatchsizeNumMicroBatchesCalculator
        assert self.mb_calculator.global_batch_size == 32
        assert self.mb_calculator.micro_batch_size == 8
        assert self.mb_calculator.data_parallel_size == 2
        assert self.mb_calculator.start_global_batch_size == 16
        assert self.mb_calculator.batch_size_increment == 16
        assert self.mb_calculator.ramup_samples == 48
        assert self.mb_calculator.micro_batch_times_data_parallel_size == 16
        assert self.mb_calculator.num_micro_batches == 1

    def test_get(self):
        assert self.mb_calculator.get() == 1

    def test_get_current_global_batch_size(self):
        assert self.mb_calculator.get_current_global_batch_size() == 16


def test_ramp_up():
    mb_calculator.reconfigure_num_microbatches_calculator(0, [16, 16, 96], 32, 8, 2)
    consumed_samples = 0
    count = 0
    expected_consumed_samples = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]

    while consumed_samples < 256:
        consumed_samples += mb_calculator.get_current_global_batch_size()
        count += 1
        assert consumed_samples == expected_consumed_samples[count]
        mb_calculator.update_num_microbatches(consumed_samples, True)
