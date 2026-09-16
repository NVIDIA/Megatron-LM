# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Low-level rank utilities with minimal dependencies to avoid circular imports."""

import logging
import os
import warnings
from typing import Any

import torch

from megatron.core._slurm_utils import resolve_slurm_rank, resolve_slurm_world_size


def safe_get_rank() -> int:
    """Get the distributed rank safely, even if torch.distributed is not initialized.

    Fallback order:
    1. torch.distributed.get_rank() (if initialized)
    2. RANK environment variable (torchrun/torchelastic)
    3. SLURM_PROCID environment variable (SLURM)
    4. Default: 0 (with warning)

    Returns:
        int: The rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    # If torch.distributed is not initialized, try to read environment variables.
    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    slurm_rank = resolve_slurm_rank()
    if slurm_rank is not None:
        return slurm_rank

    warnings.warn(
        "Could not determine rank from torch.distributed, RANK, or SLURM_PROCID. "
        "Defaulting to rank 0."
    )
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


def log_single_rank(
    logger: logging.Logger, level: int, msg: object, *args: Any, rank: int = 0, **kwargs: Any
) -> None:
    """Log a message only on a single rank.

    If torch distributed is initialized, write log on only one rank.

    Args:
        logger: The logger to write the logs.
        level: Logging level for the message.
        msg: Message format string.
        *args: Message format arguments.
        rank: The rank to write on. Defaults to 0.
        **kwargs: Additional ``logging.Logger.log`` keyword arguments.
    """
    if not logger.isEnabledFor(level):
        return

    if safe_get_rank() == rank:
        logger.log(level, msg, *args, **kwargs)


def warn_single_rank(
    message: str, category: type[Warning] = UserWarning, stacklevel: int = 2, rank: int = 0
) -> None:
    """Issue a warning only on a single rank.

    Use for warnings that describe a property of the job rather than of the calling rank,
    such as deprecated settings and experimental-API notices. Every rank raises those
    identically, so a large job repeats one message thousands of times in a shared log.

    ``safe_get_rank`` reads the RANK or SLURM_PROCID environment variable when torch
    distributed is not initialized, so this also works at import time.

    Args:
        message: The warning message.
        category: Warning category. Defaults to ``UserWarning``.
        stacklevel: Frames to skip when attributing the warning. Defaults to 2, which
            reports the caller of the function that warns.
        rank: The rank to warn on. Defaults to 0.
    """
    with warnings.catch_warnings():
        # safe_get_rank warns when it can find no rank at all, which happens on a plain
        # import outside a launcher. Defaulting to rank 0 is the right answer here, and
        # letting that warning through would just swap it for the one being deduplicated.
        warnings.simplefilter("ignore")
        current_rank = safe_get_rank()

    if current_rank == rank:
        warnings.warn(message, category=category, stacklevel=stacklevel + 1)
