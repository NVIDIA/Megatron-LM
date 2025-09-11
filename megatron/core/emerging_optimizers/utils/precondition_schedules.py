import math
from abc import ABC, abstractmethod
from typing import Dict

__all__ = [
    "LinearSchedule",
    "CosineSchedule",
    "StepSchedule",
]


class PreconditionSchedule(ABC):
    """Base class for precondition frequency schedules.

    This class provides a unified interface for creating different types of
    precondition frequency schedules. All schedules are callable and take
    the current step as input, returning the frequency for that step.

    The frequency represents how often to update the preconditioner:
    - frequency = 1 means update every step (most frequent)
    - frequency = 10 means update every 10 steps (less frequent)

    Args:
        min_freq: Minimum frequency (most frequent updates)
        max_freq: Maximum frequency (least frequent updates)
        start_step: Step at which to start applying the schedule (before this, uses min_freq)
    """

    def __init__(self, min_freq: int = 1, max_freq: int = 100, start_step: int = 0):
        """Initialize the schedule with frequency bounds."""
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        if max_freq < min_freq:
            raise ValueError("max_freq must be >= min_freq")
        if start_step < 0:
            raise ValueError("start_step must be non-negative")

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.start_step = start_step

    def __call__(self, step: int) -> int:
        """Get the frequency for the given step.

        Args:
            step: Current training step

        Returns:
            Frequency for the given step, clamped to [min_freq, max_freq]
        """
        if step < 0:
            raise ValueError("step must be non-negative")

        # Before start_step, use min_freq (most frequent updates)
        if step < self.start_step:
            return self.min_freq

        return max(self.min_freq, min(self.max_freq, self._compute_frequency(step)))

    @abstractmethod
    def _compute_frequency(self, step: int) -> int:
        """Override this method in subclasses to implement the schedule logic.

        Args:
            step: Current training step

        Returns:
            Computed frequency (before clamping to bounds)
        """
        pass


class LinearSchedule(PreconditionSchedule):
    """Linear transition from frequent to infrequent preconditioning.

    This schedule linearly interpolates between min_freq and max_freq over
    a specified number of transition steps. After the transition period,
    the frequency remains at max_freq.
    """

    def __init__(self, min_freq: int = 1, max_freq: int = 100, transition_steps: int = 10000, start_step: int = 0):
        """Initialize linear schedule.

        Args:
            min_freq: Starting frequency (most frequent updates)
            max_freq: Ending frequency (least frequent updates)
            transition_steps: Number of steps over which to transition
            start_step: Step at which to start applying the schedule (before this, uses min_freq)
        """
        super().__init__(min_freq, max_freq, start_step)
        if transition_steps <= 0:
            raise ValueError("transition_steps must be positive")
        self.transition_steps = transition_steps

    def _compute_frequency(self, step: int) -> int:
        if step <= self.transition_steps:
            # Linear interpolation
            progress = step / self.transition_steps
            return int(self.min_freq + (self.max_freq - self.min_freq) * progress)
        else:
            return self.max_freq


class CosineSchedule(PreconditionSchedule):
    """Cosine schedule that oscillates between frequencies.

    This schedule uses a cosine wave to smoothly transition between min_freq
    and max_freq over a specified period. This can be useful for cyclical
    training strategies or a single cosine increase.
    """

    def __init__(self, min_freq: int = 1, max_freq: int = 50, transition_steps: int = 20000, start_step: int = 0):
        """Initialize cosine schedule.

        Args:
            min_freq: Minimum frequency in the oscillation
            max_freq: Maximum frequency in the oscillation
            transition_steps: Number of steps over which to transition
            start_step: Step at which to start applying the schedule (before this, uses min_freq)
        """
        super().__init__(min_freq, max_freq, start_step)
        if transition_steps <= 0:
            raise ValueError("transition_steps must be positive")
        self.transition_steps = transition_steps

    def _compute_frequency(self, step: int) -> int:
        progress = (1 + math.cos(math.pi * (step % self.transition_steps) / self.transition_steps)) / 2
        current_freq = self.max_freq - (self.max_freq - self.min_freq) * progress
        return int(current_freq)


class StepSchedule(PreconditionSchedule):
    """Step-wise schedule with predefined frequency changes at specific steps.

    This schedule allows you to specify exact frequencies at specific step
    thresholds. The frequency remains constant between thresholds.

    Example:
        # Different frequencies for different training phases
        schedule = StepSchedule({
            0: 1,      # Update every step for first 1000 steps
            1000: 5,   # Update every 5 steps from 1000-4999
            5000: 10,  # Update every 10 steps from 5000-9999
            10000: 25  # Update every 25 steps from 10000 onwards
        })
    """

    def __init__(self, schedule_dict: Dict[int, int], start_step: int = 0):
        """Initialize with a dictionary mapping steps to frequencies.

        Args:
            schedule_dict: Dictionary mapping step thresholds to frequencies
                           - Keys must be non-negative integers (steps)
                           - Values must be positive integers (frequencies)
            start_step: Step at which to start applying the schedule (before this, uses min_freq)
        """
        if not schedule_dict:
            raise ValueError("schedule_dict cannot be empty")

        # Validate inputs
        for step, freq in schedule_dict.items():
            if not isinstance(step, int) or step < 0:
                raise ValueError("All step thresholds must be non-negative integers")
            if not isinstance(freq, int) or freq < 1:
                raise ValueError("All frequencies must be positive integers")

        self.sorted_steps = sorted(schedule_dict.keys())
        self.schedule_dict = schedule_dict

        # Set min/max based on the schedule
        frequencies = list(schedule_dict.values())
        super().__init__(min(frequencies), max(frequencies), start_step)

    def _compute_frequency(self, step: int) -> int:
        current_freq = self.schedule_dict[self.sorted_steps[0]]  # Default to first value
        for threshold in self.sorted_steps:
            if step >= threshold:
                current_freq = self.schedule_dict[threshold]
            else:
                break
        return current_freq
