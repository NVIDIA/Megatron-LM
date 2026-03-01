# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.

    Values are explicit integers for stable wire protocol compatibility.
    Headers are sent as binary bytes (header.value.to_bytes()) in ZMQ multipart frames.
    """

    ENGINE_CONNECT = 0
    CLIENT_CONNECT = 1
    ACK = 2
    MICROBATCH_SYNC = 3
    SUBMIT_REQUEST = 4
    ENGINE_REPLY = 5
    PAUSE = 6
    UNPAUSE = 7
    SUSPEND = 8
    RESUME = 9
    INCREMENT_STALENESS = 10
    STOP = 11
    DISCONNECT = 12
    SHUTDOWN = 13


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
