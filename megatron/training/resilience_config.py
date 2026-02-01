# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Literal

@dataclass(kw_only=True)
class RerunStateMachineConfig:
    """Configuration for the rerun state machine used for result validation or stats."""

    error_injection_rate: int = 0
    """Rate at which to inject unexpected results, e.g. 1000 means
    once every 1000 result validations"""

    error_injection_type: Literal["correct_result", "transient_error", "persistent_error"] = "transient_error"
    """Type of error to inject. """

    rerun_mode: Literal["disabled", "validate_results", "report_stats"] = "validate_results"
    """Use re-run engine to validate results (default) or to emit stats
    on variability of computations due to non-deterministic algorithms."""

    check_for_nan_in_loss: bool = True
    """Check for NaN in the loss."""

    check_for_spiky_loss: bool = False
    """Check for spiky loss."""
