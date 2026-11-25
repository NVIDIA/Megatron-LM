# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hf_pretrained/safe_config_loader.py
# Source license: Apache-2.0.

"""
Thread-safe configuration loading utilities.

This module provides utilities for safely loading HuggingFace model configurations
in multi-threaded environments, preventing race conditions that can occur when
multiple threads try to download and cache the same model simultaneously.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Union

import filelock
from huggingface_hub import constants as hf_constants
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

import megatron.bridge.models.conversion.transformers_compat  # noqa: F401  # patches removed HF utils


def safe_load_config_with_retry(
    path: Union[str, Path],
    trust_remote_code: bool = False,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> PretrainedConfig:
    """
    Thread-safe and process-safe configuration loading with retry logic.

    This function prevents race conditions when multiple threads/processes
    try to download and cache the same model configuration simultaneously.
    Uses file locking (if filelock is available) to coordinate access across
    processes.

    Args:
        path: HuggingFace model ID or path to model directory
        trust_remote_code: Whether to trust remote code when loading config
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        **kwargs: Additional arguments passed to AutoConfig.from_pretrained

    Returns:
        PretrainedConfig: The loaded model configuration

    Raises:
        ValueError: If config loading fails after all retries

    Environment Variables:
        MEGATRON_CONFIG_LOCK_DIR: Override the directory where lock files are created.
            Default: the Hugging Face home directory (``HF_HOME``, normally
            ``~/.cache/huggingface``), so the lock lives next to the cache it protects.
            Set it to a shared directory when ``HF_HOME`` is not shared across the ranks
            of a multi-node job.

    Example:
        >>> config = safe_load_config_with_retry("meta-llama/Meta-Llama-3-8B")
        >>> print(config.model_type)

        >>> # With custom retry settings
        >>> config = safe_load_config_with_retry(
        ...     "gpt2",
        ...     max_retries=5,
        ...     base_delay=0.5,
        ...     trust_remote_code=True
        ... )

        >>> # Multi-node setup with shared lock directory
        >>> import os
        >>> os.environ["MEGATRON_CONFIG_LOCK_DIR"] = "/shared/locks"
        >>> config = safe_load_config_with_retry("meta-llama/Meta-Llama-3-8B")
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            # Use file locking for process-safe access
            # Create a lock file based on the path hash to avoid conflicts
            path_hash = hashlib.md5(str(path).encode()).hexdigest()

            # Allow override of lock directory via environment variable
            # (useful for multi-node setups where a shared lock directory is needed).
            # Otherwise follow HF_HOME: that is where the config cache being protected
            # lives, and on containerised clusters $HOME is often an ephemeral,
            # per-container layer that is neither shared nor persistent.
            lock_dir = os.getenv("MEGATRON_CONFIG_LOCK_DIR") or hf_constants.HF_HOME
            lock_file = Path(lock_dir) / f".megatron_config_lock_{path_hash}"

            lock_file.parent.mkdir(parents=True, exist_ok=True)

            with filelock.FileLock(str(lock_file) + ".lock", timeout=60):
                return AutoConfig.from_pretrained(
                    path, trust_remote_code=trust_remote_code, **kwargs
                )

        except Exception as e:
            last_exception = e

            # Don't retry on certain types of errors
            error_msg = str(e).lower()
            if any(
                phrase in error_msg
                for phrase in [
                    "does not appear to have a file named config.json",
                    "repository not found",
                    "entry not found",
                    "401 client error",
                    "403 client error",
                ]
            ):
                # Model doesn't exist or access denied, no point retrying
                raise ValueError(
                    f"Failed to load configuration from {path}. "
                    f"Ensure the path is valid and contains a config.json file. "
                    f"Error: {e}"
                ) from e

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + (time.time() % 1) * 0.1
                time.sleep(delay)
            else:
                # Final attempt failed
                break

    # All retries exhausted
    hint = ""
    if isinstance(last_exception, filelock.Timeout):
        hint = (
            " The config file lock could not be acquired; on multi-node jobs set "
            "MEGATRON_CONFIG_LOCK_DIR to a directory on a shared filesystem."
        )
    raise ValueError(
        f"Failed to load configuration from {path} after {max_retries + 1} attempts. "
        f"This might be due to network issues or concurrent access conflicts.{hint} "
        f"Last error: {last_exception}"
    ) from last_exception
