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
