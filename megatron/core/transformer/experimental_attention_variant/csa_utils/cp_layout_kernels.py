# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compatibility wrapper for the former CP-only layout-kernel module.

The implementation now lives in :mod:`.thd_layout_kernels` because final-index
lowering is shared by context-parallel and generic packed-THD paths. Keep this
module temporarily for callers, including Megatron Lite, that still import the
old path.
"""

from . import thd_layout_kernels as _impl

CompressorInputCompact = _impl.CompressorInputCompact
build_attention_indices = _impl.build_attention_indices

# Preserve the availability probe used by downstream tests and diagnostics.
_CUTE_AVAILABLE = _impl._CUTE_AVAILABLE

__all__ = ["CompressorInputCompact", "build_attention_indices"]


def __getattr__(name):
    """Forward legacy access to implementation details during the rename."""
    return getattr(_impl, name)


def __dir__():
    """Expose the implementation's names to interactive legacy callers."""
    return sorted(set(globals()) | set(dir(_impl)))
