# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gated Delta Net (GDN) family of layers.

This package replaces the former ``megatron/core/ssm/gated_delta_net.py`` module
at the same import path; the names below preserve that module's public surface.
"""

from megatron.core.ssm.gated_delta_net.common import (
    HAVE_FLA,
    GatedDeltaNet,
    GatedDeltaNetSubmodules,
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    causal_conv1d,
    chunk_gated_delta_rule,
    get_parameter_local_cp_headwise,
    l2norm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
    torch_chunk_gated_delta_rule,
)

__all__ = [
    "HAVE_FLA",
    "GatedDeltaNet",
    "GatedDeltaNetSubmodules",
    "_build_head_perm_for_split_sections",
    "_build_thd_cp_a2a_perm",
    "causal_conv1d",
    "chunk_gated_delta_rule",
    "get_parameter_local_cp_headwise",
    "l2norm",
    "tensor_a2a_cp2hp",
    "tensor_a2a_hp2cp",
    "torch_chunk_gated_delta_rule",
]
