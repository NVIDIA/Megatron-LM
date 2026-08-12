# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from typing import Literal

SubmissionGranularity = Literal["R", "G", "E", "B"]
ConsumptionGranularity = Literal["G", "E", "B"]

# Coarseness order of the granularity ladder (rollout < group < env < batch).
GRANULARITY_RANK: dict[str, int] = {"R": 0, "G": 1, "E": 2, "B": 3}
