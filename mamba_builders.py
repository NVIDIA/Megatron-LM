# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
"""Backward-compatible re-export of hybrid_builders.

Deprecated. Use hybrid_builders instead.
"""
import warnings

warnings.warn(
    "mamba_builders has been deprecated. Use hybrid_builders instead.",
    DeprecationWarning,
    stacklevel=2,
)

from hybrid_builders import *  # noqa: F401,F403
from hybrid_builders import hybrid_builder as mamba_builder  # noqa: F401
