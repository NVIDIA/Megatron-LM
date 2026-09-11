# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional GDP training kernels; CP engines and checkpoint adaptation stay in SSM."""

from typing import Callable

from megatron.core.ops.kernel_metadata import validate_kernel
from megatron.core.ops.ssm.gdp.kernel_metadata import GDP_CUTEDSL, GDP_FLA

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

    HAVE_MAMBA_SSM = True
except ImportError:
    from unittest.mock import MagicMock

    RMSNormGated = MagicMock()
    HAVE_MAMBA_SSM = False

try:
    from fla.modules.l2norm import l2_norm
    from fla.ops.gated_delta_product import chunk_gated_delta_product

    HAVE_FLA = True
except ImportError:
    l2_norm = None
    chunk_gated_delta_product = None
    HAVE_FLA = False

try:
    from gdp_attn import chunk_gated_delta_product as cutedsl_chunk_gated_delta_product

    HAVE_CUTEDSL_GDP = True
except ImportError:
    cutedsl_chunk_gated_delta_product = None
    HAVE_CUTEDSL_GDP = False

__all__ = [
    "HAVE_MAMBA_SSM",
    "HAVE_FLA",
    "HAVE_CUTEDSL_GDP",
    "RMSNormGated",
    "l2_norm",
    "chunk_gated_delta_product",
    "cutedsl_chunk_gated_delta_product",
    "select_gated_delta_product",
]


def select_gated_delta_product(use_cutedsl: bool = False) -> Callable:
    """Return the selected GDP training callable with its existing signature."""
    if use_cutedsl:
        if cutedsl_chunk_gated_delta_product is None:
            raise ImportError("CuTeDSL GDP requires gdp_attn.")
        validate_kernel(GDP_CUTEDSL)
        return cutedsl_chunk_gated_delta_product
    if chunk_gated_delta_product is None:
        raise ImportError(
            "GDP requires flash-linear-attention with the gated_delta_product kernel."
        )
    validate_kernel(GDP_FLA)
    return chunk_gated_delta_product
