# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused wide-residual shortcut boundaries with call-local autograd state."""

from .api import (
    PendingShortcutAdd,
    shortcut_add_wide_residual_write_read_rmsnorm_quant,
    shortcut_add_wide_residual_write_two_reader_rmsnorm_quant,
    validate_input,
    wide_residual_write_read_rmsnorm_quant,
)

__all__ = [
    "PendingShortcutAdd",
    "shortcut_add_wide_residual_write_read_rmsnorm_quant",
    "shortcut_add_wide_residual_write_two_reader_rmsnorm_quant",
    "validate_input",
    "wide_residual_write_read_rmsnorm_quant",
]
