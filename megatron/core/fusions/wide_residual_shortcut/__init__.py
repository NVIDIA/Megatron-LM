# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused wide-residual shortcut boundaries with call-local autograd state."""

from .api import PendingWideResidualWrites, check_mxfp8_call, fused_seam
from .residual_replay import ColumnSlot, ReplayInterval, install_columns

__all__ = [
    "ColumnSlot",
    "PendingWideResidualWrites",
    "ReplayInterval",
    "check_mxfp8_call",
    "fused_seam",
    "install_columns",
]
