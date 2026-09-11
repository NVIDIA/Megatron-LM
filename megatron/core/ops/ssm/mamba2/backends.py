# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional mamba-ssm kernels and their construction-time capabilities."""

import inspect

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    HAVE_MAMBA_SSM = True
except ImportError:
    from unittest.mock import MagicMock

    RMSNormGated = MagicMock()
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None
    HAVE_MAMBA_SSM = False

MAMBA_HAS_STATE_DTYPE = (
    HAVE_MAMBA_SSM
    and ("state_dtype" in inspect.signature(mamba_split_conv1d_scan_combined).parameters)
    and ("state_dtype" in inspect.signature(mamba_chunk_scan_combined).parameters)
)

__all__ = [
    "HAVE_MAMBA_SSM",
    "MAMBA_HAS_STATE_DTYPE",
    "RMSNormGated",
    "mamba_chunk_scan_combined",
    "mamba_split_conv1d_scan_combined",
]
