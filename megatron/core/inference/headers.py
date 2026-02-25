# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum, auto


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.
    """

    CONNECT = auto()
    CONNECT_ACK = auto()
    SUBMIT_REQUEST = auto()
    ENGINE_REPLY = auto()
    PAUSE = auto()
    UNPAUSE = auto()
    SUSPEND = auto()
    RESUME = auto()
    INCREMENT_STALENESS = auto()
    STOP = auto()
    DISCONNECT = auto()


class EngineState(Enum):
    """State machine for the inference engine."""

    RUNNING = auto()  # Processing requests
    PAUSING = auto()  # PAUSE received; waiting for EP consensus + world barrier
    PAUSED = auto()  # Globally confirmed idle
    SUSPENDING = auto()  # SUSPEND received; offloading GPU; waiting for world barrier
    SUSPENDED = auto()  # GPU offloaded, all ranks confirmed
    RESUMING = auto()  # RESUME received; onloading GPU; waiting for world barrier
    STOPPING = auto()  # STOP received; futures cancelled; waiting for world barrier
    STOPPED = auto()  # All ranks confirmed; teardown complete


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
