# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
AMem NCCL Plugin Integration

This module provides integration with AMem NCCL plugin for transparent NCCL memory
offloading and restoration, particularly useful in RL scenarios where memory needs
to be freed between training and inference phases.

AMem NCCL plugin enables saving up to 10GB+ GPU memory per card by offloading
NCCL-allocated memory during rollout/inference phases in RL training.

For more information, see: https://github.com/inclusionAI/asystem-amem
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_AMEM_AVAILABLE = False
_AMEM_NCCL_LIB = None


def _try_load_amem():
    """Attempt to load the AMem NCCL library with the extended API."""
    global _AMEM_AVAILABLE, _AMEM_NCCL_LIB

    if _AMEM_NCCL_LIB is not None:
        return _AMEM_AVAILABLE

    try:
        import ctypes

        # Try to load the NCCL library with AMem extensions
        # The AMem plugin provides libnccl.so.2 with extended APIs
        nccl_lib_path = os.environ.get('AMEM_NCCL_LIB_PATH', 'libnccl.so.2')

        try:
            _AMEM_NCCL_LIB = ctypes.CDLL(nccl_lib_path)
        except OSError:
            logger.debug(f"Could not load NCCL library from {nccl_lib_path}")
            return False

        # Check if AMem APIs are available
        required_funcs = ['ncclPause', 'ncclResume', 'ncclSetGroupID', 'ncclMemStats']
        for func_name in required_funcs:
            if not hasattr(_AMEM_NCCL_LIB, func_name):
                logger.debug(f"AMem API {func_name} not found in NCCL library")
                _AMEM_NCCL_LIB = None
                return False

        # Define function signatures
        ncclResult_t = ctypes.c_int
        ncclComm_t = ctypes.c_void_p

        _AMEM_NCCL_LIB.ncclPause.argtypes = [ctypes.POINTER(ncclComm_t)]
        _AMEM_NCCL_LIB.ncclPause.restype = ncclResult_t

        _AMEM_NCCL_LIB.ncclResume.argtypes = [ctypes.POINTER(ncclComm_t)]
        _AMEM_NCCL_LIB.ncclResume.restype = ncclResult_t

        _AMEM_NCCL_LIB.ncclSetGroupID.argtypes = [ctypes.c_int]
        _AMEM_NCCL_LIB.ncclSetGroupID.restype = ncclResult_t

        _AMEM_NCCL_LIB.ncclMemStats.argtypes = []
        _AMEM_NCCL_LIB.ncclMemStats.restype = ncclResult_t

        _AMEM_AVAILABLE = True
        logger.info("AMem NCCL plugin successfully loaded")
        return True

    except Exception as e:
        logger.debug(f"Failed to load AMem NCCL plugin: {e}")
        _AMEM_NCCL_LIB = None
        return False


def is_amem_available() -> bool:
    """Check if AMem NCCL plugin is available.

    Returns:
        bool: True if AMem is available and enabled, False otherwise.
    """
    global _AMEM_AVAILABLE

    if _AMEM_NCCL_LIB is None:
        _try_load_amem()

    # Check if AMem is enabled via environment variables
    if not _AMEM_AVAILABLE:
        return False

    # Check environment variable settings
    nccl_cumem_enable = os.environ.get('NCCL_CUMEM_ENABLE', '0')
    amem_enable = os.environ.get('AMEM_ENABLE', '0')

    return _AMEM_AVAILABLE and nccl_cumem_enable == '1' and amem_enable == '1'


def nccl_pause() -> bool:
    """Offload NCCL-allocated GPU memory.

    This function releases all NCCL-allocated GPU memory in the current process,
    moving data to CPU pinned buffers. Must be called before inference/rollout
    phases in RL training to free up memory.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not is_amem_available():
        return False

    try:
        # ncclPause accepts NULL for the communicator parameter
        result = _AMEM_NCCL_LIB.ncclPause(None)
        if result == 0:  # ncclSuccess
            logger.debug("Successfully offloaded NCCL memory")
            return True
        else:
            logger.warning(f"ncclPause returned error code: {result}")
            return False
    except Exception as e:
        logger.warning(f"Failed to call ncclPause: {e}")
        return False


def nccl_resume() -> bool:
    """Restore NCCL-allocated GPU memory.

    This function restores all previously offloaded NCCL memory from CPU pinned
    buffers back to GPU. Must be called before training phases in RL training.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not is_amem_available():
        return False

    try:
        # ncclResume accepts NULL for the communicator parameter
        result = _AMEM_NCCL_LIB.ncclResume(None)
        if result == 0:  # ncclSuccess
            logger.debug("Successfully restored NCCL memory")
            return True
        else:
            logger.warning(f"ncclResume returned error code: {result}")
            return False
    except Exception as e:
        logger.warning(f"Failed to call ncclResume: {e}")
        return False


def nccl_set_group_id(group_id: int) -> bool:
    """Set the process group ID for AMem.

    When multiple processes share GPUs (e.g., training and inference processes),
    this function assigns a unique group ID to differentiate them. This must be
    called before the first NCCL memory allocation.

    Args:
        group_id: Unique identifier for the process group (e.g., 100 for training, 200 for inference).

    Returns:
        bool: True if successful, False otherwise.
    """
    if not is_amem_available():
        return False

    try:
        result = _AMEM_NCCL_LIB.ncclSetGroupID(group_id)
        if result == 0:  # ncclSuccess
            logger.info(f"Successfully set NCCL group ID to {group_id}")
            return True
        else:
            logger.warning(f"ncclSetGroupID returned error code: {result}")
            return False
    except Exception as e:
        logger.warning(f"Failed to call ncclSetGroupID: {e}")
        return False


def nccl_mem_stats() -> bool:
    """Print NCCL memory allocation statistics.

    This function reports the total NCCL memory usage and breakdown by allocation source.
    Useful for debugging and monitoring memory usage.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not is_amem_available():
        return False

    try:
        result = _AMEM_NCCL_LIB.ncclMemStats()
        return result == 0  # ncclSuccess
    except Exception as e:
        logger.warning(f"Failed to call ncclMemStats: {e}")
        return False


def setup_amem_environment(enable: bool = True, group_id: Optional[int] = None) -> None:
    """Setup environment variables for AMem NCCL plugin.

    This is a convenience function to set up the required environment variables
    for AMem. Should be called early in the initialization process.

    Args:
        enable: Whether to enable AMem (default: True).
        group_id: Optional group ID to differentiate process groups.
    """
    if enable:
        # Enable NCCL CUMEM (required for AMem)
        os.environ.setdefault('NCCL_CUMEM_ENABLE', '1')

        # Enable AMem plugin
        os.environ.setdefault('AMEM_ENABLE', '1')

        # Set group ID if provided
        if group_id is not None:
            os.environ.setdefault('AMEM_GROUPID', str(group_id))

        # Set log level (3 = INFO)
        os.environ.setdefault('GMM_LOG', '3')

        # TODO: expose AMEM_NCCL_OFFLOAD_FREE_TAG as a CLI option if needed

        logger.info("AMem NCCL plugin environment configured")
    else:
        os.environ['AMEM_ENABLE'] = '0'
        logger.info("AMem NCCL plugin disabled")
