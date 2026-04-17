# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional


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


@dataclass(kw_only=True)
class StragglerDetectionConfig:
    """Configuration settings for detecting and logging GPU stragglers."""

    log_straggler: bool = False
    """If set, tracks and logs straggler per GPU."""

    straggler_ctrlr_port: int = 65535
    """Port number to toggle StragglerDetector on/off at runtime"""

    straggler_minmax_count: int = 1
    """Number of ranks to report with high/low estimated throughput"""

    disable_straggler_on_startup: bool = False
    """If set, StragglerDetector is disabled on startup."""


@dataclass(kw_only=True)
class FaultInjectorConfig:
    """Configuration for fault injection testing via nvidia_resiliency_ext."""

    fault_injector_ranks: Optional[str] = None
    """Comma-separated list of ranks to inject faults on."""

    fault_injector_num_ranks: Optional[int] = None
    """Number of ranks to inject faults on (random selection)."""

    fault_injector_fault_types: Optional[str] = None
    """Comma-separated list of fault types to inject (e.g. 'hang,crash')."""

    fault_injector_fault_probabilities: Optional[str] = None
    """Comma-separated list of fault probabilities (normalized at runtime)."""

    fault_injector_fault_delay: Optional[float] = None
    """Force a specific fault delay in seconds from training start or delay_start_iteration."""

    fault_injector_delay_start_iteration: Optional[int] = None
    """Start the fault delay timer after iteration N completes.
    If unset, fault delay timing starts from the beginning of training."""

    fault_injector_mtti_seconds: Optional[float] = None
    """Mean time to inject (MTTI) in seconds; used when fault_delay is None."""

    fault_injector_offset_seconds: Optional[float] = None
    """Offset seconds added to the sampled fault delay."""

    fault_injector_seed: Optional[int] = None
    """RNG seed for the fault injector."""
