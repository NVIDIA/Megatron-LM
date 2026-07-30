# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State machine definitions for the data parallel inference coordinator."""

from dataclasses import dataclass, field
from enum import Enum, auto

from megatron.core.inference.headers import Headers


class CoordinatorState(Enum):
    """State machine for the coordinator."""

    RUNNING = auto()
    PAUSED = auto()
    SUSPENDED = auto()
    STOPPING = auto()


_ALL_STATES = frozenset(CoordinatorState)


@dataclass(frozen=True)
class ControlTransition:
    """A single rule in the control-signal state machine.

    Attributes:
        allowed_from: States the signal may be applied from.
        new_state: State to move to once applied, or None to leave the state
            unchanged (e.g. a pure broadcast such as SET_GENERATION_EPOCH).
        idempotent_in: States in which the signal is a silent no-op rather than
            a logged rejection (e.g. a redundant PAUSE while already paused).
    """

    allowed_from: frozenset
    new_state: CoordinatorState | None
    idempotent_in: frozenset = field(default_factory=frozenset)


# Control-signal state machine, expressed declaratively and consumed by the
# control-signal handler.
CONTROL_TRANSITIONS = {
    Headers.PAUSE: ControlTransition(
        allowed_from=frozenset({CoordinatorState.RUNNING}),
        new_state=CoordinatorState.PAUSED,
        idempotent_in=frozenset({CoordinatorState.PAUSED, CoordinatorState.SUSPENDED}),
    ),
    Headers.UNPAUSE: ControlTransition(
        allowed_from=frozenset({CoordinatorState.PAUSED}), new_state=CoordinatorState.RUNNING
    ),
    Headers.SUSPEND: ControlTransition(
        allowed_from=frozenset({CoordinatorState.PAUSED}), new_state=CoordinatorState.SUSPENDED
    ),
    Headers.RESUME: ControlTransition(
        allowed_from=frozenset({CoordinatorState.SUSPENDED}), new_state=CoordinatorState.PAUSED
    ),
    Headers.STOP: ControlTransition(
        allowed_from=frozenset({CoordinatorState.PAUSED, CoordinatorState.SUSPENDED}),
        new_state=CoordinatorState.STOPPING,
    ),
    # No state change; broadcast in any state so engines stay in sync.
    Headers.SET_GENERATION_EPOCH: ControlTransition(allowed_from=_ALL_STATES, new_state=None),
}
