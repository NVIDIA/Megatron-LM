# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.
    """

    CONNECT = 0
    CONNECT_ACK = 1
    SUBMIT_REQUEST = 2
    ENGINE_REPLY = 3
    PAUSE = 4
    PAUSE_ACK = 5
    UNPAUSE = 6
    STOP = 7
