# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional FLA entry points shared by the GDN model variants."""

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None
    HAVE_FLA = False

try:
    from fla.modules.convolution import causal_conv1d_update
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
except ImportError:
    causal_conv1d_update = None
    fused_recurrent_gated_delta_rule = None

try:
    from fla.ops.gdn2.chunk import chunk_gdn2

    HAVE_FLA_GDN2 = True
except ImportError:
    chunk_gdn2 = None
    HAVE_FLA_GDN2 = False

__all__ = [
    "HAVE_FLA",
    "HAVE_FLA_GDN2",
    "causal_conv1d",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "chunk_gdn2",
    "fused_recurrent_gated_delta_rule",
    "l2norm",
]
