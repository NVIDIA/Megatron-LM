# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from datetime import datetime
import os
from functools import partial
from logging import Filter, LogRecord
from typing import Callable

from megatron.core._rank_utils import safe_get_rank, safe_get_world_size
import torch


logger = logging.getLogger(__name__)


def warning_filter(record: LogRecord) -> bool:
    """Logging filter to exclude WARNING level messages.

    Args:
        record: The logging record to check.

    Returns:
        False if the record level is WARNING, True otherwise.
    """
    if record.levelno == logging.WARNING:
        return False

    return True


def module_filter(record: LogRecord, modules_to_filter: list[str]) -> bool:
    """Logging filter to exclude messages from specific modules.

    Args:
        record: The logging record to check.
        modules_to_filter: A list of module name prefixes to filter out.

    Returns:
        False if the record's logger name starts with any of the specified
        module prefixes, True otherwise.
    """
    for module in modules_to_filter:
        if record.name.startswith(module):
            return False
    return True


def add_filter_to_all_loggers(filter: Filter | Callable[[LogRecord], bool]) -> None:
    """Add a filter to the root logger and all existing loggers.

    Args:
        filter: A logging filter instance or callable to add.
    """
    # Get the root logger
    root = logging.getLogger()
    root.addFilter(filter)

    # Add handler to all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.addFilter(filter)


def setup_logging(
    logging_level: int | None = None,
    filter_warning: bool = True,
    modules_to_filter: list[str] | None = None,
    set_level_for_all_loggers: bool = False,
) -> None:
    """Set up logging level and filters for the application.

    Configures the logging level based on arguments, environment variables,
    or defaults. Optionally adds filters to suppress warnings or messages
    from specific modules.

    Logging Level Precedence:
    1. Env var `MEGATRON_BRIDGE_LOGGING_LEVEL`
    2. `logging_level` argument
    3. Default: `logging.INFO`

    Args:
        logging_level: The desired logging level (e.g., logging.INFO, logging.DEBUG).
        filter_warning: If True, adds a filter to suppress WARNING level messages.
        modules_to_filter: An optional list of module name prefixes to filter out.
        set_level_for_all_loggers: If True, sets the logging level for all existing
                                   loggers. If False (default), only sets the level
                                   for the root logger and loggers starting with 'megatron.bridge'.
    """
    cfg_logging_level = logging_level
    env_logging_level = os.getenv("MEGATRON_BRIDGE_LOGGING_LEVEL", None)

    logging_level = logging.INFO
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if cfg_logging_level is not None:
        logging_level = cfg_logging_level

    logger.info(f"Setting logging level to {logging_level}")
    logging.getLogger().setLevel(logging_level)

    for _logger_name in logging.root.manager.loggerDict:
        if _logger_name.startswith("megatron.bridge") or set_level_for_all_loggers:
            _logger = logging.getLogger(_logger_name)
            _logger.setLevel(logging_level)

    if filter_warning:
        add_filter_to_all_loggers(warning_filter)
    if modules_to_filter:
        add_filter_to_all_loggers(partial(module_filter, modules_to_filter=modules_to_filter))


def append_to_progress_log(save_dir: str, string: str, barrier: bool = True) -> None:
    """Append a formatted string to the progress log file (rank 0 only).

    Includes timestamp, job ID, and number of GPUs in the log entry.

    Args:
        save_dir: The directory where the 'progress.txt' file is located.
        string: The message string to append.
        barrier: If True, performs a distributed barrier before writing (rank 0 only).
    """
    if save_dir is None:
        return
    progress_log_filename = os.path.join(save_dir, "progress.txt")
    if barrier and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if safe_get_rank() == 0:
        os.makedirs(os.path.dirname(progress_log_filename), exist_ok=True)
        with open(progress_log_filename, "a+") as f:
            job_id = os.getenv("SLURM_JOB_ID", "")
            num_gpus = safe_get_world_size()
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t# GPUs: {num_gpus}\t{string}\n"
            )


