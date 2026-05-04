# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Low-level rank utilities with minimal dependencies to avoid circular imports."""

import logging
import os
import warnings
from typing import Any

import torch

from megatron.core._slurm_utils import resolve_slurm_world_size


def safe_get_rank() -> int:
    """Safely get the rank of the current process.

    Returns the rank from torch.distributed if initialized, otherwise falls back
    to the RANK environment variable, defaulting to 0.

    Returns:
        int: The rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    # If torch.distributed is not initialized, try to read environment variables.
    try:
        return int(os.environ.get("RANK", 0))
    except (ValueError, TypeError):
        return 0


def safe_get_world_size() -> int:
    """Get the distributed world size safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_world_size() (if initialized)
    2. WORLD_SIZE environment variable (torchrun/torchelastic)
    3. SLURM_NTASKS environment variable (SLURM)
    4. Default: 1 (with warning)

    Returns:
        The total number of processes in the distributed job.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    slurm_world_size = resolve_slurm_world_size()
    if slurm_world_size is not None:
        return slurm_world_size

    warnings.warn(
        "Could not determine world size from torch.distributed, WORLD_SIZE, or SLURM_NTASKS. "
        "Defaulting to world size 1."
    )
    return 1


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any) -> None:
    """Log a message only on a single rank.

    If torch distributed is initialized, write log on only one rank.

    Args:
        logger: The logger to write the logs.
        *args: All logging.Logger.log positional arguments.
        rank: The rank to write on. Defaults to 0.
        **kwargs: All logging.Logger.log keyword arguments.
    """
    if safe_get_rank() == rank:
        logger.log(*args, **kwargs)
