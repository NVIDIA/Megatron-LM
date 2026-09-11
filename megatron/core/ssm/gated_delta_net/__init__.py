# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility exports for the GDN/GDN2 operation implementations."""

from megatron.core.ops.ssm.gated_delta.modules import (
    HAVE_FLA,
    HAVE_FLA_GDN2,
    GatedDeltaNet,
    GatedDeltaNet2,
    GatedDeltaNetSubmodules,
    causal_conv1d,
    chunk_gated_delta_rule,
    chunk_gdn2,
    get_parameter_local_cp,
    l2norm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
    torch_chunk_gated_delta_rule,
    torch_chunk_gdn2,
)

__all__ = [
    "HAVE_FLA",
    "HAVE_FLA_GDN2",
    "GatedDeltaNet",
    "GatedDeltaNet2",
    "GatedDeltaNetSubmodules",
    "causal_conv1d",
    "chunk_gated_delta_rule",
    "chunk_gdn2",
    "get_parameter_local_cp",
    "l2norm",
    "tensor_a2a_cp2hp",
    "tensor_a2a_hp2cp",
    "torch_chunk_gated_delta_rule",
    "torch_chunk_gdn2",
]
