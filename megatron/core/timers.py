# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List

import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from megatron.core.utils import is_torch_min_version

try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base

logger = logging.getLogger(__name__)


class TimerBase(ABC):
    """Timer base class."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def start(self, barrier=False):
        """Start the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before starting. Defaults to False.
        """
        pass

    @abstractmethod
    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset timer."""
        pass

    @abstractmethod
    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Args:
            reset (bool, optional): Resets timer before restarting. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time.
        """
        pass


class DummyTimer(TimerBase):
    """Dummy Timer."""

    def __init__(self):
        super().__init__('dummy timer')

    def start(self, barrier=False):
        return

    def stop(self, barrier=False):
        return

    def reset(self):
        return

    def elapsed(self, reset=True, barrier=False):
        raise Exception(
            'dummy timer should not be used to calculate elapsed time, '
            'check if timer\'s log_level <= self._log_level.'
        )

    def active_time(self):
        """Returns the cumulative duration the timer has been active.
        Note: Not supported for DummyTimer.
        """
        raise Exception(
            'active timer should not be used to calculate elapsed time, '
            'check if timer\'s log_level <= self._log_level.'
        )


class Timer(TimerBase):
    """
    Timer class with ability to start/stop.

    Timing is taken from CUDA events, not the host clock, so start() and stop()
    do not synchronize. An interval is resolved only once its events have
    completed, which means a measurement is reported a few calls after the
    interval it describes. Reporting is lagged but lossless: every interval is
    reported exactly once, in order, so an aggregate over many iterations is
    exact.

    The alternative -- host clock plus torch.cuda.synchronize() -- fences the
    whole training loop on every call, which both costs the sync and hides any
    other host stall behind it.

    Comment on using `barrier`: If this flag is passed, then all
    the caller processes will wait till all reach the timing routine.
    It is up to the user to make sure all the ranks in `barrier_group`
    call it otherwise, it will result in a hang.
    Comment on `barrier_group`: By default it is set to None which
    in torch distributed land, it will result in the global communicator.
    """

    # Cap on unresolved windows. The CUDA launch queue bounds how far the host
    # can run ahead in practice; this only stops events accumulating without
    # limit if the device stalls, and costs one wait when hit.
    _MAX_PENDING = 1024

    def __init__(self, name):
        """Initialize Timer.

        Args:
            name (str): Name of the timer.
        """
        super().__init__(name)
        self._active_time = 0.0
        self._started = False
        # Note that None will default to the global process group
        self._barrier_group = None
        self._start_event = None
        # Current window: intervals whose events have not completed yet, the
        # seconds already resolved out of it, and how many intervals it holds.
        self._open_window: Deque = deque()
        self._open_resolved = 0.0
        self._open_count = 0
        # Windows closed by elapsed(), as (pending, resolved_seconds, count).
        self._closed_windows: Deque = deque()
        # Resolved window durations in seconds, oldest first, not yet reported.
        self._ready: Deque = deque()
        self._last_reported = 0.0

    def set_barrier_group(self, barrier_group):
        """Sets barrier group.

        Args:
            barrier_group (ProcessGroup): Torch ProcessGroup for barrier.
        """
        self._barrier_group = barrier_group

    def start(self, barrier=False):
        """Start the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before starting. Defaults to False.
        """
        assert not self._started, 'timer has already been started'
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        self._started = True

    def stop(self, barrier=False):
        """Stop the timer.

        Args:
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.
        """
        assert self._started, 'timer is not started'
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        stop_event = torch.cuda.Event(enable_timing=True)
        stop_event.record()
        self._open_window.append((self._start_event, stop_event))
        self._open_count += 1
        self._start_event = None
        self._started = False
        # Fold away whatever has completed, so a window spanning many intervals
        # holds only the events still in flight.
        self._drain_open()

    def _drain_open(self):
        """Resolve completed intervals of the current window into a partial sum."""
        while self._open_window and self._open_window[0][1].query():
            start_event, stop_event = self._open_window.popleft()
            # elapsed_time is milliseconds; this class reports seconds.
            self._open_resolved += start_event.elapsed_time(stop_event) / 1000.0

    def _resolve(self, block=False):
        """Move every completed window from the pending queue to the ready queue.

        Events complete in stream order, so the last event of a window standing
        in for the whole window is sufficient.

        Args:
            block (bool): Wait for the first window rather than returning empty.
                Used only for the very first report, which has no earlier value
                to stand in for it.
        """
        while self._closed_windows:
            pending, resolved, count = self._closed_windows[0]
            if pending and not pending[-1][1].query():
                if not block and len(self._closed_windows) <= self._MAX_PENDING:
                    break
                # Either the first report, or backlogged and holding too many
                # events. Wait rather than report nothing / accumulate forever.
                pending[-1][1].synchronize()
            self._closed_windows.popleft()
            if count == 0:
                # Nothing was timed in this window. Reporting 0.0 for it would
                # blow up the throughput the caller divides out of it.
                continue
            total = resolved + sum(s.elapsed_time(e) for s, e in pending) / 1000.0
            self._ready.append(total)
            self._active_time += total
            block = False  # got one; go back to never waiting

    def reset(self):
        """Reset timer, discarding intervals not yet reported."""
        # Don't reset _active_time
        self._open_window.clear()
        self._open_resolved = 0.0
        self._open_count = 0
        self._closed_windows.clear()
        self._ready.clear()
        self._started = False

    def set_elapsed(self, value):
        """Directly set the elapsed time.

        This is useful for injecting pre-computed timing values (e.g., startup
        timestamps) into the timer so they can be reported via timers.log().

        Args:
            value (float): The elapsed time value in seconds.
        """
        self._ready.append(value)
        self._last_reported = value

    def elapsed(self, reset=True, barrier=False):
        """Calculates the elapsed time and restarts timer.

        Reports the oldest interval whose events have completed, so the value
        describes a slightly earlier window than the one just closed. Until the
        first window resolves, reports the last value seen -- never 0.0, which
        callers divide by to get throughput.

        Args:
            reset (bool, optional): Consume the reported value, so the next call
                returns the following interval. When False, peek without
                consuming. Defaults to True.
            barrier (bool, optional): Synchronizes ranks before stopping. Defaults to False.

        Returns:
            float: Elapsed time in seconds.
        """
        _started = self._started
        # If the timing in progress, end it first.
        if self._started:
            self.stop(barrier=barrier)
        # Everything recorded since the last call belongs to the window we close
        # here. Restart immediately so the caller's remaining work lands in the
        # next window rather than in a gap between windows.
        self._drain_open()
        self._closed_windows.append(
            (self._open_window, self._open_resolved, self._open_count)
        )
        self._open_window = deque()
        self._open_resolved = 0.0
        self._open_count = 0
        if _started:
            self.start(barrier=barrier)
        # The first report has nothing to fall back on and the caller divides by
        # it, so pay one wait for it. Once per run.
        self._resolve(block=self._last_reported == 0.0)
        if self._ready:
            self._last_reported = self._ready.popleft() if reset else self._ready[0]
        return self._last_reported

    def active_time(self):
        """Calculates the cumulative duration for which the timer has been active"""
        return self._active_time


class Timers:
    """Class for a group of Timers."""

    def __init__(self, log_level, log_option):
        """Initialize group of timers.

        Args:
            log_level (int): Log level to control what timers are enabled.
            log_option (str): Setting for logging statistics over ranks for all the timers.
                              Allowed: ['max', 'minmax', 'all'].
        """
        self._log_level = log_level
        allowed_log_options = set(['max', 'minmax', 'all'])
        assert (
            log_option in allowed_log_options
        ), 'input log option {} is invalid. It must be one of {}'.format(
            log_option, allowed_log_options
        )
        self._log_option = log_option
        self._timers = {}
        self._log_levels = {}
        self._dummy_timer = DummyTimer()
        self._max_log_level = 2

    def __call__(self, name, log_level=None):
        """Call timer with name and log level."""
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:
            if log_level is not None:
                assert log_level == self._log_levels[name], (
                    'input log level {} does not match already existing '
                    'log level {} for {} timer'.format(log_level, self._log_levels[name], name)
                )
            return self._timers[name]
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:
            log_level = self._max_log_level
        assert (
            log_level <= self._max_log_level
        ), 'log level {} is larger than max supported log level {}'.format(
            log_level, self._max_log_level
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:
            return self._dummy_timer
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)
        self._log_levels[name] = log_level
        return self._timers[name]

    def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        """Returns elapsed times of timers in names.
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.

        Args:
            names (List[str]): list of timer names
            reset (bool): reset the timer after recording the elapsed time
            barrier (bool): if set, do a global barrier before time measurments

        Returns:
            torch.tensor: Tensor of size [world_size, len(names)] with times in float.
        """

        if len(names) == 0:
            return None

        # First make sure all the callers are in sync.
        if barrier:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros(
            (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()
        )
        for i, name in enumerate(names):
            if name in self._timers:
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)

        # See the note above for why we are not using gather.
        dist_all_gather_func(rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1))

        return rank_name_to_time

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
        # Using Python built-in methods to avoid the overhead of PyTorch operations.
        rank_name_to_time = (
            rank_name_to_time.permute(1, 0).tolist() if rank_name_to_time is not None else None
        )
        name_to_min_max_time = {}
        for i, name in enumerate(names):
            # filter out the ones we did not have any timings for
            rank_to_time = list(filter(lambda x: x > 0.0, rank_name_to_time[i]))
            # If the timer exists:
            if len(rank_to_time) > 0:
                name_to_min_max_time[name] = (
                    min(rank_to_time) / normalizer,
                    max(rank_to_time) / normalizer,
                )
        return name_to_min_max_time

    def _get_global_min_max_time_string(self, names, reset, barrier, normalizer, max_only):
        """Report strings for max/minmax times across all ranks."""
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if not name_to_min_max_time:
            return None
        if max_only:
            output_string = 'max time across ranks (ms):'
        else:
            output_string = '(min, max) time across ranks (ms):'
        for name in name_to_min_max_time:
            min_time, max_time = name_to_min_max_time[name]
            if max_only:
                output_string += '\n    {}: {:.2f}'.format((name + ' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(
                    (name + ' ').ljust(48, '.'), min_time, max_time
                )
        return output_string

    def _get_all_ranks_time_string(self, names, reset, barrier, normalizer):
        """Report times across all ranks."""
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)

        output_string = 'times across ranks (ms):'
        no_reported_timing = True
        for i, name in enumerate(names):
            not_yet_found = True
            for rank in range(torch.distributed.get_world_size()):
                if rank_name_to_time[rank, i] > 0:
                    no_reported_timing = False
                    if not_yet_found:
                        not_yet_found = False
                        output_string += '\n  {}:'.format(name)
                    output_string += '\n     rank {:2d}: {:.2f}'.format(
                        rank, rank_name_to_time[rank, i] / normalizer
                    )
        if no_reported_timing:
            return None
        return output_string

    def get_all_timers_string(
        self,
        names: List[str] = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Returns the output string with logged timer values according to configured options.

        Args:
            names (List[str]): Names of the timers to log. If None, all registered timers are
                               fetched. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.

        Raises:
            Exception: Raises if log option is invalid.

        Returns:
            str: Formatted string with the timer values.
        """

        if names == None:  # get all registered timers
            names = self._timers.keys()

        assert normalizer > 0.0
        if self._log_option in ['max', 'minmax']:
            max_only = False
            if self._log_option == 'max':
                max_only = True
            output_string = self._get_global_min_max_time_string(
                names, reset, barrier, normalizer / 1000.0, max_only
            )
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception('unknown timing log option {}'.format(self._log_option))
        return output_string

    def log(
        self,
        names: List[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """logs the timers passed in names to stdout. Example usage is to log average per step
           value for timer 'foo', this function can be called with normalizer factor set to logging
           interval.

        Args:
            names (List[str]): Names of the timers to log.
            rank (int, optional): logs the timers to a specific rank. If set to None, logs to the
                                  last rank. Defaults to None.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.
        """

        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)
        # If no input rank is provided, log on last rank.
        if rank is None:
            rank = torch.distributed.get_world_size() - 1
        if rank == torch.distributed.get_rank() and output_string is not None:
            logger.info(output_string)

    def write(
        self,
        names: List[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Write timers to a tensorboard writer.
        Note that we only report maximum time across ranks to tensorboard.

        Args:
            names (List[str]): Names of the timers to log.
            writer (SummaryWriter): Tensorboard SummaryWriter object
            iteration (int): Current iteration.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                if isinstance(writer, SummaryWriter) and SummaryWriter is not None:
                    writer.add_scalar(name + '-time', max_time, iteration)
                elif writer == wandb and wandb is not None:
                    writer.log({name + '-time': max_time}, iteration)
