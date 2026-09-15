# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional-kernel test skips use the same ``require`` checks as construction."""

from megatron.core.ops._backends import require


def kernels_available(*requirements: tuple) -> bool:
    """Whether every ``(module, *symbols)`` requirement can be satisfied, without falling back."""
    try:
        for module, *symbols in requirements:
            require(module, *symbols, needed_by="test")
    except ImportError:
        return False
    return True


FLA_CONV = ("fla.modules.convolution", "causal_conv1d")
FLA_L2NORM = ("fla.modules.l2norm", "l2norm")
GDN_FLA = ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
GDN_RECURRENT = ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule")
GDN2_FLA = ("fla.ops.gdn2.chunk", "chunk_gdn2")
GDP_CONV = ("causal_conv1d", "causal_conv1d_fn")
GDP_FLA = ("fla.ops.gated_delta_product", "chunk_gated_delta_product")
GDP_L2NORM = ("fla.modules.l2norm", "l2_norm")
GDP_FLA_CP = ("megatron.core.ops.ssm.context_parallel.gdp", "FLAGatedDeltaProductCPBackend")
GDP_CUTEDSL_CP = (
    "megatron.core.ops.ssm.context_parallel.gdp_cutedsl",
    "CuTeDSLGatedDeltaProductCPBackend",
)
MAMBA_NORM = ("mamba_ssm.ops.triton.layernorm_gated", "RMSNorm")

HAVE_FLA = kernels_available(FLA_CONV, FLA_L2NORM, GDN_FLA)
HAVE_FLA_GDN2 = kernels_available(FLA_CONV, FLA_L2NORM, GDN2_FLA)
HAVE_GDP_DEPS = kernels_available(GDP_CONV, GDP_FLA, GDP_L2NORM, MAMBA_NORM)
HAVE_FLA_GDP_CP = kernels_available(GDP_FLA_CP)
HAVE_CUTEDSL_GDP_CP = kernels_available(GDP_CUTEDSL_CP)

if HAVE_FLA_GDN2:
    from fla.ops.gdn2.chunk import chunk_gdn2
else:
    chunk_gdn2 = None
