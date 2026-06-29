# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State machine definitions for the data parallel inference coordinator."""

from enum import Enum, auto

from megatron.core.inference.headers import Headers


class CoordinatorState(Enum):
    """State machine for the coordinator."""

    RUNNING = auto()
    PAUSED = auto()
    SUSPENDED = auto()
    STOPPING = auto()


_ALL_STATES = frozenset(CoordinatorState)

# Control-signal state machine, expressed declaratively.
# header -> (states the signal is allowed from, resulting state or None for no
# transition, states in which the signal is a silent no-op rather than a logged
# rejection). Consumed by the control-signal handler.
CONTROL_TRANSITIONS = {
    Headers.PAUSE: (
        frozenset({CoordinatorState.RUNNING}),
        CoordinatorState.PAUSED,
        frozenset({CoordinatorState.PAUSED, CoordinatorState.SUSPENDED}),
    ),
    Headers.UNPAUSE: (
        frozenset({CoordinatorState.PAUSED}),
        CoordinatorState.RUNNING,
        frozenset(),
    ),
    Headers.SUSPEND: (
        frozenset({CoordinatorState.PAUSED}),
        CoordinatorState.SUSPENDED,
        frozenset(),
    ),
    Headers.RESUME: (
        frozenset({CoordinatorState.SUSPENDED}),
        CoordinatorState.PAUSED,
        frozenset(),
    ),
    Headers.STOP: (
        frozenset({CoordinatorState.PAUSED, CoordinatorState.SUSPENDED}),
        CoordinatorState.STOPPING,
        frozenset(),
    ),
    # No state change; broadcast in any state so engines stay in sync.
    Headers.SET_GENERATION_EPOCH: (_ALL_STATES, None, frozenset()),
}
