# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional-kernel test skips use the same requirements as construction."""

from megatron.core.ops.kernel_metadata import KernelMetadata, validate_kernels
from megatron.core.ops.ssm.gated_delta.kernel_metadata import (
    FLA_CONV,
    FLA_L2NORM,
    GDN2_FLA,
    GDN_FLA,
)
from megatron.core.ops.ssm.gdp.kernel_metadata import (
    GDP_CONV,
    GDP_CUTEDSL_CP,
    GDP_FLA,
    GDP_FLA_CP,
    GDP_L2NORM,
)
from megatron.core.ops.ssm.mamba2.kernel_metadata import MAMBA_NORM


def kernels_available(*kernels: KernelMetadata) -> bool:
    """Check imports/exports/versions for a test's selection, without choosing a fallback."""
    try:
        validate_kernels(kernels)
    except ImportError:
        return False
    return True


HAVE_FLA = kernels_available(FLA_CONV, FLA_L2NORM, GDN_FLA)
HAVE_FLA_GDN2 = kernels_available(FLA_CONV, FLA_L2NORM, GDN2_FLA)
HAVE_GDP_DEPS = kernels_available(GDP_CONV, GDP_FLA, GDP_L2NORM, MAMBA_NORM)
HAVE_FLA_GDP_CP = kernels_available(GDP_FLA_CP)
HAVE_CUTEDSL_GDP_CP = kernels_available(GDP_CUTEDSL_CP)

if HAVE_FLA_GDN2:
    from fla.ops.gdn2.chunk import chunk_gdn2
else:
    chunk_gdn2 = None
