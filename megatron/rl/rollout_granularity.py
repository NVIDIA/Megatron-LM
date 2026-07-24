# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from typing import Literal

SubmissionGranularity = Literal["R", "G", "E", "B"]
ConsumptionGranularity = Literal["G", "E", "B"]
ReleaseState = Literal["inferred", "assembled", "env_assembled", "consumed"]


RELEASE_STATE_BY_SUBMISSION: dict[SubmissionGranularity, ReleaseState] = {
    "R": "inferred",
    "G": "assembled",
    "E": "env_assembled",
    "B": "consumed",
}

# Coarseness order of the granularity ladder (rollout < group < env < batch).
# Consumption must be no finer than submission.
GRANULARITY_RANK: dict[str, int] = {"R": 0, "G": 1, "E": 2, "B": 3}
