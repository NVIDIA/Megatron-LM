# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Utilities for detecting and configuring SLURM cluster environments.

This module provides functionality to detect SLURM environments and extract
distributed training configuration from SLURM environment variables.
"""

import os


def is_slurm_job() -> bool:
    """Detect if running in a SLURM environment.

    Returns:
        True if SLURM job detected, False otherwise.
    """
    return "SLURM_NTASKS" in os.environ


def resolve_slurm_rank() -> int | None:
    """Get the global rank from SLURM environment.

    Returns:
        The global rank, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None
    return int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else None


def resolve_slurm_world_size() -> int | None:
    """Get the world size from SLURM environment.

    Returns:
        The world size, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None
    return int(os.environ["SLURM_NTASKS"]) if "SLURM_NTASKS" in os.environ else None


def resolve_slurm_local_rank() -> int | None:
    """Get the local rank from SLURM environment.

    Returns:
        The local rank, or None if not in SLURM environment.
    """
    if not is_slurm_job():
        return None
    return int(os.environ["SLURM_LOCALID"]) if "SLURM_LOCALID" in os.environ else None
