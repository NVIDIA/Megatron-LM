# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from typing import Literal

SubmissionGranularity = Literal["R", "G", "B"]
ConsumptionGranularity = Literal["G", "B"]

# Coarseness order of the granularity ladder (rollout < group < batch).
GRANULARITY_RANK: dict[str, int] = {"R": 0, "G": 1, "B": 2}
