# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.
    """

    CONNECT = 0
    ACK = 1
    MICROBATCH_SYNC = 2
    SUBMIT_REQUEST = 3
    ENGINE_REPLY = 4
    PAUSE = 5
    UNPAUSE = 6
    STOP = 7
