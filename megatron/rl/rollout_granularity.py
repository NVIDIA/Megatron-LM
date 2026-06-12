# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from enum import Enum


class RLRolloutGranularity(str, Enum):
    """Granularity for RL rollout submission or consumption."""

    ROLLOUT = 'R'
    GROUP = 'G'
    BATCH = 'B'

    def __str__(self):
        return self.value
