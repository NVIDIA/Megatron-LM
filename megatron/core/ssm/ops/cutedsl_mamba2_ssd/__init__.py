# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuteDSL Blackwell mamba2 SSD kernel, adapted for THD varlen packing.

This package vendors the upstream Blackwell CuteDSL SSD kernel
(``_mamba2_ssd_kernel_varlen.py``, a per-(sequence, head) varlen tile
scheduler) and wraps it with a THD (token-packed, variable length) front-end
in :mod:`ssd_cutedsl` that mirrors the Triton
``mamba_chunk_scan_combined_varlen`` interface. Batches whose sequence
lengths are not multiples of the kernel chunk size fall back to Triton.
"""

from .ssd_cutedsl import (
    cutedsl_unsupported_reason,
    is_cutedsl_ssd_available,
    mamba_chunk_scan_combined_varlen_cutedsl_thd,
)

__all__ = [
    "mamba_chunk_scan_combined_varlen_cutedsl_thd",
    "is_cutedsl_ssd_available",
    "cutedsl_unsupported_reason",
]
