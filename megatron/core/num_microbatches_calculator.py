# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron Core number of microbatches calculators."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# TODO: global_var merge into mcore?
_GLOBAL_NUM_MICROBATCHES_CALCULATOR: Union[
    'ConstantNumMicroBatchesCalculator', 'RampupBatchsizeNumMicroBatchesCalculator', 'StepBatchsizeNumMicroBatchesCalculator'
] = None


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


def update_num_microbatches(
    consumed_samples: int, consistency_check: bool = True, verbose: bool = False
) -> None:
    """Update number of microbatches.

    Args:
        consumed_samples (int):
            Number of samples consumed.
        consistency_check (bool, optional):
            Option to check current schedule's consistency. Defaults to True.
        verbose (bool, optional):
            Option to control logging. Defaults to False.
    """
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check, verbose)


def unset_num_microbatches_calculator():
    """Unset microbatches calculator.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py`
    for an example.
    """
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def init_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    step_batch_size_schedule: Optional[str],
    seq_length: Optional[int],
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Initialize number of microbatches calculator. Supporting backward compatibility.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]):
            Rampup batch size, should be in format of [start_global_batch_size,
            batch_size_increment, ramup_samples].
        global_batch_size (int):
            Global batch size for the model.
        micro_batch_size (int):
            Micro batch size at initialization.
        data_parallel_size (int):
            Data parallel size.
        step_batch_size_schedule (Optional[str]):
            Step batch size schedule string in format "THRESHOLD:BS THRESHOLD:BS ...".
            Thresholds support suffixes: K (1e3), M (1e6), B (1e9), T (1e12).
            Example: "0:768 250B:1536 500B:3072 750B:6144"
        seq_length (Optional[int]):
            Sequence length for token-to-sample conversion when using step_batch_size_schedule.
            If provided, thresholds are interpreted as tokens. If None, thresholds are samples.
        decrease_batch_size_if_needed (bool, optional):
            If true, scale down batch size to ensure divisibility by DP size * microbatch size.
            Defaults to False.
    """
    _configure_global_num_microbatches_calculator(
        rank,
        rampup_batch_size,
        global_batch_size,
        micro_batch_size,
        data_parallel_size,
        step_batch_size_schedule,
        seq_length,
        decrease_batch_size_if_needed,
        init=True,
    )


def destroy_num_microbatches_calculator():
    """Destroy number of microbatches calculator."""
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def reconfigure_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    step_batch_size_schedule: Optional[str],
    seq_length: Optional[int],
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Reconfigure number of microbatches calculator. Supporting backward compatibility.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]):
            Rampup batch size, should be in format of
            [start_global_batch_size, batch_size_increment, ramup_samples].
        global_batch_size (int):
            Global batch size for the model.
        micro_batch_size (int):
            Micro batch size at initialization.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool, optional):
            If true, scale down batch size to ensure divisibility by DP size * microbatch size.
            Defaults to False.
        step_batch_size_schedule (Optional[str]):
            Step batch size schedule string in format "THRESHOLD:BS THRESHOLD:BS ...".
            Thresholds support suffixes: K (1e3), M (1e6), B (1e9), T (1e12).
            Example: "0:768 250B:1536 500B:3072 750B:6144"
        seq_length (Optional[int]):
            Sequence length for token-to-sample conversion when using step_batch_size_schedule.
            If provided, thresholds are interpreted as tokens. If None, thresholds are samples.
    """
    _configure_global_num_microbatches_calculator(
        rank,
        rampup_batch_size,
        global_batch_size,
        micro_batch_size,
        data_parallel_size,
        step_batch_size_schedule,
        seq_length,
        decrease_batch_size_if_needed,
        init=False,
    )


def _configure_global_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    step_batch_size_schedule: Optional[str],
    seq_length: Optional[int],
    decrease_batch_size_if_needed: bool = False,
    init: bool = False,
) -> None:
    """Configure number of microbatches calculator. Can be used for initialization and
    reconfiguration.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]):
            Rampup batch size, should be in format of
            [start_global_batch_size, batch_size_increment, ramup_samples].
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
        step_batch_size_schedule (Optional[str]):
            Step batch size schedule string in format "THRESHOLD:BS THRESHOLD:BS ...".
            Thresholds support suffixes: K (1e3), M (1e6), B (1e9), T (1e12).
            Example: "0:768 250B:1536 500B:3072 750B:6144"
        seq_length (Optional[int]):
            Sequence length for token-to-sample conversion when using step_batch_size_schedule.
            If provided, thresholds are interpreted as tokens. If None, thresholds are samples.
    """
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    if init:
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
        step_batch_size_schedule,
        seq_length,
    )


def _build_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: Optional[int],
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool,
    step_batch_size_schedule: Optional[str] = None,
    seq_length: Optional[int] = None,
) -> Union['ConstantNumMicroBatchesCalculator', 'RampupBatchsizeNumMicroBatchesCalculator', 'StepBatchsizeNumMicroBatchesCalculator']:
    """Build number of microbatches calculator. Internal helper method.

    Args:
        rank (int):
            Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]):
            Rampup batch size, should be in format of
            [start_global_batch_size, batch_size_increment, ramup_samples].
        global_batch_size (Optional[int]):
            Global batch size for the model. Required for constant and rampup modes.
            Ignored when step_batch_size_schedule is provided.
        micro_batch_size (int):
            Micro batch size at initialization.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool):
            If true, scale down batch size to ensure divisibility by DP size * microbatch size.
        step_batch_size_schedule (Optional[str]):
            Step batch size schedule string in format "THRESHOLD:BS THRESHOLD:BS ...".
            Thresholds support suffixes: K (1e3), M (1e6), B (1e9), T (1e12).
            Example: "0:768 250B:1536 500B:3072 750B:6144"
        seq_length (Optional[int]):
            Sequence length for token-to-sample conversion when using step_batch_size_schedule.
            If provided, thresholds are interpreted as tokens. If None, thresholds are samples.
    """

    # Validate mutually exclusive options
    if step_batch_size_schedule is not None and rampup_batch_size is not None:
        raise ValueError(
            'Cannot specify both --step-batch-size-schedule and --rampup-batch-size'
        )

    # Step batch size schedule
    if step_batch_size_schedule is not None:
        if decrease_batch_size_if_needed:
            raise ValueError(
                'Cannot specify both --step-batch-size-schedule and '
                '--decrease-batch-size-if-needed'
            )
        if global_batch_size is not None and rank == 0:
            logger.warning(
                '--global-batch-size is ignored when using --step-batch-size-schedule'
            )
        num_microbatches_calculator = StepBatchsizeNumMicroBatchesCalculator(
            micro_batch_size=micro_batch_size,
            data_parallel_size=data_parallel_size,
            decrease_batch_size_if_needed=decrease_batch_size_if_needed,
            rank=rank,
            schedule=step_batch_size_schedule,
            seq_length=seq_length,
        )

    # Batch size ramp up
    elif rampup_batch_size is not None:
        assert global_batch_size is not None, (
            '--global-batch-size is required when using --rampup-batch-size'
        )
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
                f'will use batch size rampup starting from global batch size '
                f'{start_global_batch_size} to global batch size {global_batch_size} with batch'
                f'size increments {batch_size_increment} over {ramup_samples} samples.'
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

    # Constant batch size
    else:
        assert global_batch_size is not None, (
            '--global-batch-size is required when not using --step-batch-size-schedule'
        )
        num_microbatches_calculator = ConstantNumMicroBatchesCalculator(
            global_batch_size,
            micro_batch_size,
            data_parallel_size,
            decrease_batch_size_if_needed,
            rank,
        )
        if rank == 0:
            logger.info(
                f'setting number of microbatches to constant {num_microbatches_calculator.get()}'
            )

    return num_microbatches_calculator


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
        """Update number of microbatches depending on batch size rampup."""
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


class RampupBatchsizeNumMicroBatchesCalculator(NumMicroBatchesCalculator):
    """Calculator of number of microbatches with batch size rampup.
    Over `steps = (global-batch-size - start-batch-size) / batch_size_increment` increment batch
    size from start-batch-size to global-batch-size using rampup-samples / steps
    samples.

    Args:
        global_batch_size (int):
            Global batch size post rampup.
        micro_batch_size (int):
            Micro batch size.
        data_parallel_size (int):
            Data parallel size.
        decrease_batch_size_if_needed (bool):
            If true, decrease batch size to ensure divisibility by DP size * microbatch size
            (if needed).
        rank (int):
            Rank (to determine whether logging should be performed).
        start_global_batch_size (int):
            Global batch size to start with.
        batch_size_increment (int):
            Global batch size increments.
        ramup_samples (int):
            Number of samples to use ramp up global
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
        assert diff_batch_size >= 0, (
            'expected global batch size to be greater than or equal to start batch size, '
            f'got {self.global_batch_size} and {self.start_global_batch_size}'
        )
        assert diff_batch_size % batch_size_increment == 0, (
            'expected '
            f'global batch size interval ({diff_batch_size}) to be divisible by global batch '
            f'size increment ({batch_size_increment})'
        )

        num_increments = diff_batch_size // self.batch_size_increment
        self.rampup_samples_per_increment = self.ramup_samples / num_increments

        # Initialize number of microbatches.
        self.update(0, consistency_check=False, verbose=True)

    def update(self, consumed_samples: int, consistency_check: bool, verbose: bool = False) -> None:
        """Update number of microbatches.

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
            if old_current_global_batch_size is None:
                logger.info(f'setting initial batch size to {self.current_global_batch_size}')
            else:
                logger.info(
                    f'ramping up batch size from {old_current_global_batch_size} to '
                    f'{self.current_global_batch_size}'
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
                    f'decreasing batch size from {self.current_global_batch_size} to '
                    f'{self.current_running_global_batch_size} to keep divisiblity by '
                    f'micro_batch_size={self.micro_batch_size} * '
                    f'data_parallel_size={self.data_parallel_size}'
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

class StepBatchsizeNumMicroBatchesCalculator(NumMicroBatchesCalculator):
    """Calculator of number of microbatches with arbitrary step-wise batch size schedule.

    Args:
        micro_batch_size (int): Micro batch size.
        data_parallel_size (int): Data parallel size.
        decrease_batch_size_if_needed (bool): Must be False. Step schedules do not support
            decreasing batch size for divisibility.
        rank (int): Rank for logging.
        schedule (str): Schedule string in format "THRESHOLD:BS THRESHOLD:BS ...".
            Thresholds support suffixes: K (1e3), M (1e6), B (1e9), T (1e12).
            Examples:
                "0:768 250B:1536 500B:3072 750B:6144" (thresholds in tokens)
                "0:768 61035156250:1536" (thresholds in samples)
        seq_length (int, optional): Sequence length for token-to-sample conversion.
            If provided, thresholds are interpreted as tokens and converted to samples.
            If None, thresholds are interpreted as samples directly.
    """

    def __init__(
        self,
        micro_batch_size: int,
        data_parallel_size: int,
        decrease_batch_size_if_needed: bool,
        rank: int,
        schedule: str,
        seq_length: Optional[int] = None,
    ) -> None:
        super().__init__()

        if decrease_batch_size_if_needed:
            raise ValueError(
                'Step batch size schedules do not support decrease_batch_size_if_needed'
            )

        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.decrease_batch_size_if_needed = decrease_batch_size_if_needed
        self.rank = rank
        self.seq_length = seq_length

        self.micro_batch_times_data_parallel_size = micro_batch_size * data_parallel_size
        assert self.micro_batch_times_data_parallel_size > 0

        # Parse schedule string
        self.schedule = self._parse_schedule(schedule, seq_length)

        # Validate schedule
        self._validate_schedule()

        self.global_batch_size = self.schedule[-1][1]
        self.current_global_batch_size = None

        if rank == 0:
            logger.info(f'> initializing step batch size schedule')
            logger.info(f'  raw schedule string: "{schedule}"')
            logger.info(f'  seq_length: {seq_length} (thresholds interpreted as {"tokens" if seq_length else "samples"})')
            logger.info(f'  micro_batch_size: {micro_batch_size}')
            logger.info(f'  data_parallel_size: {data_parallel_size}')
            logger.info(f'  decrease_batch_size_if_needed: {decrease_batch_size_if_needed}')
            logger.info(f'step batch size schedule ({len(self.schedule)} steps):')
            for threshold, batch_size in self.schedule:
                num_microbatches = batch_size // self.micro_batch_times_data_parallel_size
                if seq_length:
                    tokens = threshold * seq_length
                    logger.info(f'  >= {tokens:,} tokens ({threshold:,} samples) -> batch_size={batch_size}, num_microbatches={num_microbatches}')
                else:
                    logger.info(f'  >= {threshold:,} samples -> batch_size={batch_size}, num_microbatches={num_microbatches}')
        # Initialize
        self.update(0, consistency_check=False, verbose=True)

    @staticmethod
    def _parse_numeric_value(value_str: str) -> int:
        """Parse numeric value with optional suffix (K, M, B, T)."""
        value_str = value_str.strip().upper()

        multiplier = 1
        if value_str.endswith('T'):
            multiplier = 1_000_000_000_000
            value_str = value_str[:-1]
        elif value_str.endswith('B'):
            multiplier = 1_000_000_000
            value_str = value_str[:-1]
        elif value_str.endswith('M'):
            multiplier = 1_000_000
            value_str = value_str[:-1]
        elif value_str.endswith('K'):
            multiplier = 1_000
            value_str = value_str[:-1]

        return int(float(value_str) * multiplier)

    @classmethod
    def _parse_schedule(
        cls, schedule_str: str, seq_length: Optional[int]
    ) -> List[Tuple[int, int]]:
        """Parse schedule string into list of (threshold_samples, batch_size) tuples.

        Args:
            schedule_str: Space-separated "THRESHOLD:BATCH_SIZE" pairs.
            seq_length: If provided, convert thresholds from tokens to samples.

        Returns:
            List of (threshold_samples, batch_size) tuples, sorted by threshold.
        """
        schedule = []
        entries = schedule_str.strip().replace(',', ' ').split()

        for entry in entries:
            if ':' not in entry:
                raise ValueError(
                    f'Invalid schedule entry "{entry}". Expected format: "THRESHOLD:BATCH_SIZE"'
                )

            threshold_str, batch_size_str = entry.split(':', 1)
            threshold = cls._parse_numeric_value(threshold_str)
            batch_size = cls._parse_numeric_value(batch_size_str)

            # Convert tokens to samples if seq_length provided
            if seq_length is not None:
                threshold = threshold // seq_length

            schedule.append((threshold, batch_size))

        # Sort by threshold ascending
        schedule.sort(key=lambda x: x[0])

        return schedule

    def _validate_schedule(self) -> None:
        """Validate the parsed schedule."""
        assert len(self.schedule) > 0, 'schedule must have at least one entry'
        assert self.schedule[0][0] == 0, (
            f'first schedule entry must have threshold 0, got {self.schedule[0][0]}'
        )

        # Check strictly increasing thresholds
        for i in range(1, len(self.schedule)):
            assert self.schedule[i][0] > self.schedule[i - 1][0], (
                f'schedule thresholds must be strictly increasing, '
                f'got {self.schedule[i - 1][0]} before {self.schedule[i][0]}'
            )

        # Validate batch sizes are positive.
        # NOTE: divisibility by micro_batch_size * data_parallel_size is NOT checked here
        # because early schedule entries may be smaller than the current GPU configuration
        # (e.g., after scaling up GPUs mid-training). Divisibility of the CURRENT batch size
        # is checked at runtime in update() when consistency_check=True (after checkpoint loading).
        for threshold, batch_size in self.schedule:
            assert batch_size > 0, f'batch size must be positive, got {batch_size}'

    def _get_batch_size_for_samples(self, consumed_samples: int) -> int:
        """Get the batch size for the given number of consumed samples."""
        batch_size = self.schedule[0][1]
        for threshold, bs in self.schedule:
            if consumed_samples >= threshold:
                batch_size = bs
            else:
                break
        return batch_size

    @staticmethod
    def _round(batch_size: int, divisor: int) -> int:
        """Round batch_size down to nearest value divisible by divisor."""
        return (batch_size // divisor) * divisor

    def update(self, consumed_samples: int, consistency_check: bool, verbose: bool = False) -> None:
        """Update number of microbatches based on consumed samples.

        Args:
            consumed_samples (int): Number of samples consumed.
            consistency_check (bool): Check divisibility constraints.
            verbose (bool): Enable logging.
        """
        old_current_global_batch_size = self.current_global_batch_size
        self.current_global_batch_size = self._get_batch_size_for_samples(consumed_samples)

        global_batch_size_changed = old_current_global_batch_size != self.current_global_batch_size

        if self.rank == 0 and global_batch_size_changed and verbose:
            if old_current_global_batch_size is None:
                logger.info(f'setting initial batch size to {self.current_global_batch_size}')
            else:
                logger.info(
                    f'stepping batch size from {old_current_global_batch_size} to '
                    f'{self.current_global_batch_size} at {consumed_samples:,} samples'
                )

        # Consistency check
        if consistency_check and not self.decrease_batch_size_if_needed:
            assert self.current_global_batch_size % self.micro_batch_times_data_parallel_size == 0, (
                f'current global batch size ({self.current_global_batch_size}) is not divisible by '
                f'micro_batch_size ({self.micro_batch_size}) * '
                f'data_parallel_size ({self.data_parallel_size})'
            )

        # Handle decrease_batch_size_if_needed
        if (
            self.decrease_batch_size_if_needed
            and self.current_global_batch_size % self.micro_batch_times_data_parallel_size != 0
        ):
            self.current_running_global_batch_size = self._round(
                self.current_global_batch_size, self.micro_batch_times_data_parallel_size
            )
            if self.rank == 0 and global_batch_size_changed and verbose:
                logger.info(
                    f'adjusted running batch size to {self.current_running_global_batch_size} '
                    f'for divisibility'
                )
        else:
            self.current_running_global_batch_size = self.current_global_batch_size

        self.num_micro_batches = (
            self.current_running_global_batch_size // self.micro_batch_times_data_parallel_size
        )
