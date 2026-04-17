# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron Core number of microbatches calculators."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# TODO: global_var merge into mcore?
_GLOBAL_NUM_MICROBATCHES_CALCULATOR: 'ConstantNumMicroBatchesCalculator' = None


def get_num_microbatches() -> int:
    """Get number of microbatches."""
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


def unset_num_microbatches_calculator():
    """Unset microbatches calculator.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py`
    for an example.
    """
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def init_num_microbatches_calculator(
    rank: int,
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Initialize number of microbatches calculator. Supporting backward compatibility.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        global_batch_size (int):
            Global batch size for the model.
        micro_batch_size (int):
            Micro batch size at initialization.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool, optional):
            If true, scale down batch size to ensure divisibility by DP size * microbatch size.
            Defaults to False.
    """
    _configure_global_num_microbatches_calculator(
        rank,
        global_batch_size,
        micro_batch_size,
        data_parallel_size,
        decrease_batch_size_if_needed,
        init=True,
    )


def destroy_num_microbatches_calculator():
    """Destroy number of microbatches calculator."""
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def reconfigure_num_microbatches_calculator(
    rank: int,
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Reconfigure number of microbatches calculator. Supporting backward compatibility.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        global_batch_size (int):
            Global batch size for the model.
        micro_batch_size (int):
            Micro batch size at initialization.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool, optional):
            If true, scale down batch size to ensure divisibility by DP size * microbatch size.
            Defaults to False.
    """
    _configure_global_num_microbatches_calculator(
        rank,
        global_batch_size,
        micro_batch_size,
        data_parallel_size,
        decrease_batch_size_if_needed,
        init=False,
    )


def _configure_global_num_microbatches_calculator(
    rank: int,
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
    init: bool = False,
) -> None:
    """Configure number of microbatches calculator. Can be used for initialization and
    reconfiguration.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        global_batch_size (int):
            Global batch size for the model.
        micro_batch_size (int):
            Micro batch size at initialization.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool, optional):
            If true, scale down batch size to ensure divisibility by DP size * microbatch size.
            Defaults to False.
        init (bool, optional):
            If true, initialize the calculator. Defaults to False.
    """
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    if init:
        assert (
            _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None
        ), 'num microbatches calculator is already initialized.'

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = ConstantNumMicroBatchesCalculator(
        global_batch_size, micro_batch_size, data_parallel_size, decrease_batch_size_if_needed, rank
    )
    if rank == 0:
        logger.info(
            f'setting number of microbatches to constant '
            f'{_GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()}'
        )


def _round(batch_size: int, divisor: int) -> int:
    """Round `batch_size` down to nearest batch size divisible by `divisor`."""
    return (batch_size // divisor) * divisor


class NumMicroBatchesCalculator(ABC):
    """Base class for number of microbatches calculator."""

    def __init__(self) -> None:
        self.num_micro_batches = None
        self.current_global_batch_size = None
        self.micro_batch_size = None
        self.current_running_global_batch_size = None

    def get(self) -> int:
        """Get number of microbatches."""
        return self.num_micro_batches

    def get_current_global_batch_size(self) -> int:
        """Get current global batch size."""
        return self.current_global_batch_size

    def get_micro_batch_size(self) -> int:
        """Get current global batch size."""
        return self.micro_batch_size

    def get_current_running_global_batch_size(self) -> int:
        """Get current running global batch size. If decrease_batch_size_if_needed is False,
        this just equals global batch size."""
        return self.current_running_global_batch_size

    @abstractmethod
    def update(self, consumed_samples, consistency_check, verbose=False) -> None:
        """Update number of microbatches."""
        pass


class ConstantNumMicroBatchesCalculator(NumMicroBatchesCalculator):
    """Calculator of number of microbatches with constant global batch size.

    Args:
        global_batch_size (int):
            Global batch size.
        micro_batch_size (int):
            Micro batch size.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool):
            If true, decrease batch size to ensure divisibility by DP size * microbatch size
            (if needed).
        rank (int):
            Rank (to determine whether logging should be performed).
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
                    f'to keep divisiblity by micro_batch_size={micro_batch_size} * '
                    f'data_parallel_size={data_parallel_size}'
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
        ), 'number of microbatches should be at least 1, got {}.'.format(self.num_micro_batches)

        self.current_global_batch_size = global_batch_size
        self.current_running_global_batch_size = running_global_batch_size
        self.micro_batch_size = micro_batch_size

    def update(self, consumed_samples, consistency_check, verbose=False) -> None:
        pass
