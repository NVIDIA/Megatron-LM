# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Compatibility layer for cuda-core version differences.

cuda-core >=0.5 removed the ``cuda.core.experimental._memory`` and
``cuda.core.experimental._stream`` private submodules, but nvshmem4py
still imports from them.  We register ``sys.modules`` shims so those
imports resolve to the new ``cuda.core._memory`` / ``cuda.core._stream``
paths.

This module should be imported before any nvshmem.core usage.
"""

import importlib
import sys


def _patch_cuda_core_experimental():
    """Register cuda.core._memory / _stream as cuda.core.experimental._memory / _stream."""
    for submod in ("_memory", "_stream"):
        exp_key = f"cuda.core.experimental.{submod}"
        new_key = f"cuda.core.{submod}"
        if exp_key not in sys.modules:
            try:
                sys.modules[exp_key] = importlib.import_module(new_key)
            except ImportError:
                pass  # old cuda-core that still has experimental._memory


def get_cuda_core_device_class():
    """Return the ``Device`` class from whichever cuda-core location is available.

    cuda-core <0.5: ``cuda.core.experimental.Device``
    cuda-core >=0.5: ``cuda.core.Device``
    """
    try:
        from cuda.core import Device

        return Device
    except ImportError:
        from cuda.core.experimental import Device

        return Device


def ensure_nvshmem_compat():
    """Apply all compatibility patches.  Safe to call multiple times."""
    _patch_cuda_core_experimental()
