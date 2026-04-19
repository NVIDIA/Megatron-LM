# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
"""Backward-compatible wrapper for pretrain_hybrid.py.

Deprecated. Use pretrain_hybrid.py instead.
"""
import os
import runpy
import warnings

warnings.warn(
    "pretrain_mamba.py has been deprecated. Use pretrain_hybrid.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Execute pretrain_hybrid.py as if it were invoked directly.
_this_dir = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_this_dir, "pretrain_hybrid.py"), run_name="__main__")
