# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Validation utilities for GPU-to-GPU communication.

Provides deterministic data generation and validation for verifying

correctness of communication operations."""

from dataclasses import dataclass
from typing import List

import torch

from .logger import PELogger


@dataclass
class ValidationResult:
    """Result of validating a single task."""

    task_id: int
    size: int
    passed: bool
    src_pe: int = -1
    mismatches: int = 0
    first_mismatch_idx: int = -1
    first_mismatch_expected: int = 0
    first_mismatch_actual: int = 0
    # Scheduling info - which batch/iteration this task was supposed to be handled
    batch_index: int = -1
    iteration: int = -1


@dataclass
class ValidationSummary:
    """Summary of validation across all tasks."""

    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    total_bytes: int
    results: List[ValidationResult]

    @property
    def all_passed(self) -> bool:
        """Check if all validated tasks passed."""
        return self.failed_tasks == 0


def generate_deterministic_data(task_id: int, size: int, device: str = "cuda") -> torch.Tensor:
    """
    Generate deterministic data pattern for a task.

    Pattern: Each byte = (task_id * 31 + position) % 256
    This creates a unique pattern per task that varies along the data.

    Args:
        task_id: Unique task identifier
        size: Number of bytes to generate
        device: Device to create tensor on ('cuda' or 'cpu')

    Returns:
        torch.Tensor of uint8 with deterministic pattern
    """
    positions = torch.arange(size, dtype=torch.int64, device=device)
    pattern = ((task_id * 31 + positions) % 256).to(torch.uint8)
    return pattern


def validate_received_data(
    task_id: int, tensor: torch.Tensor, size: int, src_pe: int = -1
) -> ValidationResult:
    """
    Validate received data against expected deterministic pattern.

    Args:
        task_id: Task identifier to regenerate expected data
        tensor: Received tensor to validate
        size: Number of bytes to validate

    Returns:
        ValidationResult with pass/fail status and details
    """
    # Get the data slice to validate
    recv_data = tensor[:size]

    # Generate expected pattern on same device
    expected = generate_deterministic_data(task_id, size, device=recv_data.device.type)

    # Compare
    mismatches_mask = recv_data != expected
    num_mismatches = mismatches_mask.sum().item()

    result = ValidationResult(
        task_id=task_id,
        size=size,
        passed=(num_mismatches == 0),
        src_pe=src_pe,
        mismatches=num_mismatches,
    )

    if num_mismatches > 0:
        # Find first mismatch for debugging
        first_idx = mismatches_mask.nonzero(as_tuple=True)[0][0].item()
        result.first_mismatch_idx = first_idx
        result.first_mismatch_expected = expected[first_idx].item()
        result.first_mismatch_actual = recv_data[first_idx].item()

    return result


def log_validation_summary(summary: ValidationSummary) -> None:
    """Log validation summary."""
    if summary.all_passed:
        PELogger.info(
            "Validation PASSED: %d/%d tasks, %d bytes validated",
            summary.passed_tasks,
            summary.total_tasks,
            summary.total_bytes,
        )
    else:
        PELogger.error(
            "Validation FAILED: %d/%d tasks passed, %d failed",
            summary.passed_tasks,
            summary.total_tasks,
            summary.failed_tasks,
        )

        # Group failures by source PE
        failures_by_src = {}
        for r in summary.results:
            if not r.passed:
                failures_by_src.setdefault(r.src_pe, []).append(r)

        PELogger.error("  Failures by source PE:")
        for src_pe in sorted(failures_by_src.keys()):
            failed_tasks = failures_by_src[src_pe]
            task_ids = [r.task_id for r in failed_tasks]
            PELogger.error(
                "    PE %d: %d failed tasks: %s",
                src_pe,
                len(failed_tasks),
                task_ids[:15] if len(task_ids) <= 15 else task_ids[:15] + ["..."],
            )
