# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron Core number of micro-batches calculators."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# TODO: global_var merge into mcore?
_GLOBAL_NUM_MICROBATCHES_CALCULATOR: Union[
    'ConstantNumMicroBatchesCalculator', 'RampupBatchsizeNumMicroBatchesCalculator'
] = None


def get_num_microbatches() -> int:
    """Get number of micro-batches."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size() -> int:
    """Get current global batch size."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def get_micro_batch_size() -> int:
    """Get micro batch size."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_micro_batch_size()


def get_current_running_global_batch_size() -> int:
    """Get current running global batch size, taking into account number of DP replicas might be
    incompatible with true global batch size if `decrease_batch_size_if_needed` is True."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_running_global_batch_size()


def update_num_microbatches(
    consumed_samples: int, consistency_check: Optional[bool] = True, verbose: Optional[bool] = False
) -> None:
    """Update number of micro-batches.

    Args:
        consumed_samples (int): Number of samples consumed.
        consistency_check (bool, optional): Option to check current schedule's consistency. Defaults to True.
        verbose (bool, optional): Option to control logging. Defaults to False.
    """
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check, verbose)


def init_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Initialize number of micro-batches calculator.

    Args:
        rank (int): Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]): Rampup batch size.
        global_batch_size (int): Global batch size for the model.
        micro_batch_size (int): Micro batch size at initialization.
        data_parallel_size (int): Data parallel size.
        decrease_batch_size_if_needed (bool): If true, scale down batch size to ensure divisibility by DP size * microbatch size. Default false.
    """
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    assert (
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None
    ), 'num microbatches calculator is already initialized.'

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = _build_num_microbatches_calculator(
        rank,
        rampup_batch_size,
        global_batch_size,
        micro_batch_size,
        data_parallel_size,
        decrease_batch_size_if_needed,
    )


def reconfigure_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool,
) -> None:
    """Reconfigure number of micro-batches calculator.

    Args:
        rank (int): Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
        global_batch_size (int): Global batch size for the model.
        micro_batch_size (int): Micro batch size at initialization.
        data_parallel_size (int): Data parallel size.
        decrease_batch_size_if_needed (bool): If true, scale down batch size to ensure divisibility by DP size * microbatch size.
    """
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = _build_num_microbatches_calculator(
        rank,
        rampup_batch_size,
        global_batch_size,
        micro_batch_size,
        data_parallel_size,
        decrease_batch_size_if_needed,
    )


def _build_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool,
) -> Union['ConstantNumMicroBatchesCalculator', 'RampupBatchsizeNumMicroBatchesCalculator']:
    """Build number of micro-batches calculator.

    Args:
        rank (int): Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
        global_batch_size (int): Global batch size for the model.
        micro_batch_size (int): Micro batch size at initialization.
        data_parallel_size (int): Data parallel size.
        decrease_batch_size_if_needed (bool): If true, scale down batch size to ensure divisibility by DP size * microbatch size.
    """

    # Constant num micro-batches.
    if rampup_batch_size is None:
        num_microbatches_calculator = ConstantNumMicroBatchesCalculator(
            global_batch_size,
            micro_batch_size,
            data_parallel_size,
            decrease_batch_size_if_needed,
            rank,
        )
        if rank == 0:
            logger.info(
                f'setting number of micro-batches to constant {num_microbatches_calculator.get()}'
            )
    # Batch size ramp up num micro-batches.
    else:
        assert len(rampup_batch_size) == 3, (
            'expected the following '
            'format: --rampup-batch-size <start batch size> '
            '<batch size incerement> <ramp-up samples>'
        )
        start_global_batch_size = int(rampup_batch_size[0])
        batch_size_increment = int(rampup_batch_size[1])
        ramup_samples = int(rampup_batch_size[2])
        if rank == 0:
            logger.info(
                f'will use batch size rampup starting from global batch size {start_global_batch_size} to global batch size {global_batch_size} with batch size increments {batch_size_increment} over {ramup_samples} samples.'
            )
        num_microbatches_calculator = RampupBatchsizeNumMicroBatchesCalculator(
            global_batch_size,
            micro_batch_size,
            data_parallel_size,
            decrease_batch_size_if_needed,
            rank,
            start_global_batch_size,
            batch_size_increment,
            ramup_samples,
        )

    return num_microbatches_calculator


def _round(batch_size: int, divisor: int) -> int:
    """Round `batch_size` down to nearest batch size divisible by `divisor`."""
    return (batch_size // divisor) * divisor


class NumMicroBatchesCalculator(ABC):
    """Base class for number of micro-batches calculator."""

    def __init__(self) -> None:
        self.num_micro_batches = None
        self.current_global_batch_size = None
        self.micro_batch_size = None
        self.current_running_global_batch_size = None

    def get(self) -> int:
        """Get number of micro-batches."""
        return self.num_micro_batches

    def get_current_global_batch_size(self) -> int:
        """Get current global batch size."""
        return self.current_global_batch_size

    def get_micro_batch_size(self) -> int:
        """Get current global batch size."""
        return self.micro_batch_size

    def get_current_running_global_batch_size(self) -> int:
        """Get current running global batch size. If decrease_batch_size_if_needed is False, this just equals global batch size."""
        return self.current_running_global_batch_size

    @abstractmethod
    def update(self, consumed_samples, consistency_check, verbose=False) -> None:
        pass


class ConstantNumMicroBatchesCalculator(NumMicroBatchesCalculator):
    """Calculator of number of micro-batches with constant global batch size.

    Args:
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        data_parallel_size (int): Data parallel size.
        decrease_batch_size_if_needed (bool): If true, decrease batch size to ensure divisibility by DP size * microbatch size (if needed).
        rank (int): Rank (to determine whether logging should be performed).
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
        decrease_batch_size_if_needed: bool,
        rank: int,
    ) -> None:

        micro_batch_times_data_parallel_size = micro_batch_size * data_parallel_size
        if decrease_batch_size_if_needed:
            running_global_batch_size = _round(
                global_batch_size, micro_batch_times_data_parallel_size
            )
            assert running_global_batch_size % micro_batch_times_data_parallel_size == 0
            if rank == 0:
                logger.info(
                    f'decreasing batch size from {global_batch_size} to {running_global_batch_size}'
                )
            self.num_micro_batches = (
                running_global_batch_size // micro_batch_times_data_parallel_size
            )
        else:
            assert global_batch_size % micro_batch_times_data_parallel_size == 0, (
                'global batch size ({}) is not divisible by micro batch size ({})'
                ' times data parallel size ({})'.format(
                    global_batch_size, micro_batch_size, data_parallel_size
                )
            )
            running_global_batch_size = global_batch_size
            self.num_micro_batches = global_batch_size // micro_batch_times_data_parallel_size
        assert (
            self.num_micro_batches >= 1
        ), 'number of micro-batches should be at least 1, got {}.'.format(self.num_micro_batches)

        self.current_global_batch_size = global_batch_size
        self.current_running_global_batch_size = running_global_batch_size
        self.micro_batch_size = micro_batch_size

    def update(self, consumed_samples, consistency_check, verbose=False) -> None:
        pass


class RampupBatchsizeNumMicroBatchesCalculator(NumMicroBatchesCalculator):
    """Calculator of number of micro-batches with ramp up global batch size.
    Over
        steps = (global-batch-size - start-batch-size) / batch_size_increment
    increment batch size from start-batch-size to global-batch-size using
        rampup-samples / steps
    samples.

    Args:
        global_batch_size (int): Global batch size post rampup.
        micro_batch_size (int): Micro batch size.
        data_parallel_size (int): Data parallel size.
        decrease_batch_size_if_needed (bool): If true, decrease batch size to ensure divisibility by DP size * microbatch size (if needed).
        rank (int): Rank (to determine whether logging should be performed).
        start_global_batch_size (int): Global batch size to start with.
        batch_size_increment (int): Global batch size increments.
        ramup_samples (int): Number of samples to use ramp up global
            batch size from `start_global_batch_size` to `global_batch_size`.
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
        decrease_batch_size_if_needed: bool,
        rank: int,
        start_global_batch_size: int,
        batch_size_increment: int,
        ramup_samples: int,
    ) -> None:
        assert global_batch_size > 0, 'global batch size should be positive, got {}.'.format(
            global_batch_size
        )
        assert start_global_batch_size > 0, 'start batch size should be positive, got {}.'.format(
            start_global_batch_size
        )
        assert batch_size_increment > 0, 'batch size increment should be positive, got {}.'.format(
            batch_size_increment
        )
        assert ramup_samples >= 0, 'ramp-up samples should be non-negative, got {}.'.format(
            ramup_samples
        )

        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.decrease_batch_size_if_needed = decrease_batch_size_if_needed
        self.rank = rank
        self.start_global_batch_size = start_global_batch_size
        self.batch_size_increment = batch_size_increment
        self.ramup_samples = ramup_samples

        self.micro_batch_times_data_parallel_size = self.micro_batch_size * self.data_parallel_size
        assert self.micro_batch_times_data_parallel_size > 0
        self.current_global_batch_size = None

        diff_batch_size = self.global_batch_size - self.start_global_batch_size
        assert (
            diff_batch_size >= 0
        ), 'expected global batch size to be greater than or equal to start batch size, got {} and {}.'.format(
            self.global_batch_size, self.start_global_batch_size
        )
        assert diff_batch_size % batch_size_increment == 0, (
            'expected '
            'global batch size interval ({}) to be divisible by global batch '
            'size increment ({})'.format(diff_batch_size, batch_size_increment)
        )

        num_increments = diff_batch_size // self.batch_size_increment
        self.rampup_samples_per_increment = self.ramup_samples / num_increments

        # Initialize number of microbatches.
        self.update(0, False)

    def update(
        self, consumed_samples: int, consistency_check: bool, verbose: Optional[bool] = False
    ) -> None:
        """Update number of micro-batches.

        Args:
            consumed_samples (int): Number of samples consumed.
            consistency_check (bool): Option to check current schedule's consistency.
            verbose (bool, optional): Option to control logging. Defaults to False.
        """

        # Update current global batch size.
        global_batch_size_changed = False
        old_current_global_batch_size = self.current_global_batch_size
        if consumed_samples > self.ramup_samples:
            self.current_global_batch_size = self.global_batch_size
        else:
            steps = int(consumed_samples / self.rampup_samples_per_increment)
            self.current_global_batch_size = (
                self.start_global_batch_size + steps * self.batch_size_increment
            )
            assert self.current_global_batch_size <= self.global_batch_size

        if old_current_global_batch_size != self.current_global_batch_size:
            global_batch_size_changed = True
        if self.rank == 0 and global_batch_size_changed and verbose:
            logger.info(
                f'ramping up batch size from {old_current_global_batch_size} to {self.current_global_batch_size}'
            )

        # Check consistency of the current global batch size.
        if consistency_check and not self.decrease_batch_size_if_needed:
            assert (
                self.current_global_batch_size % self.micro_batch_times_data_parallel_size == 0
            ), (
                'current global '
                'batch size ({}) is not divisible by micro-batch-size ({}) times'
                'data parallel size ({})'.format(
                    self.current_global_batch_size, self.micro_batch_size, self.data_parallel_size
                )
            )

        if (
            self.decrease_batch_size_if_needed
            and self.current_global_batch_size % self.micro_batch_times_data_parallel_size != 0
        ):
            self.current_running_global_batch_size = _round(
                self.current_global_batch_size, self.micro_batch_times_data_parallel_size
            )
            if self.rank == 0 and global_batch_size_changed and verbose:
                logger.info(
                    f'decreasing batch size from {self.current_global_batch_size} to {self.current_running_global_batch_size}'
                )
            assert (
                self.current_running_global_batch_size % self.micro_batch_times_data_parallel_size
                == 0
            )
        else:
            self.current_running_global_batch_size = self.current_global_batch_size

        self.num_micro_batches = (
            self.current_running_global_batch_size // self.micro_batch_times_data_parallel_size
        )
