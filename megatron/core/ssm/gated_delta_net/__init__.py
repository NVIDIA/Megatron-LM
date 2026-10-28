# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path; use ``megatron.core.ops.ssm.gated_delta``.

The historical package-level names resolve from the modules that define them.
"""

from megatron.core.ops._compat import deprecated_module

__getattr__, __dir__ = deprecated_module(
    __name__,
    "megatron.core.ops.ssm.gated_delta",
    exports={
        "HAVE_FLA": "megatron.core.ops.ssm.gated_delta.common",
        "HAVE_FLA_GDN2": "megatron.core.ops.ssm.gated_delta.gdn2",
        "HAVE_TE_GDN": "megatron.core.extensions.transformer_engine",
        "GatedDeltaNet": "megatron.core.ops.ssm.gated_delta.gdn",
        "GatedDeltaNet2": "megatron.core.ops.ssm.gated_delta.gdn2",
        "GatedDeltaNetSubmodules": "megatron.core.ops.ssm.gated_delta.common",
        "causal_conv1d": "megatron.core.ops.ssm.gated_delta.common",
        "chunk_gated_delta_rule": "megatron.core.ops.ssm.gated_delta.common",
        "chunk_gdn2": "megatron.core.ops.ssm.gated_delta.gdn2",
        "get_parameter_local_cp": "megatron.core.ops.ssm.gated_delta.common",
        "l2norm": "megatron.core.ops.ssm.gated_delta.common",
        "tensor_a2a_cp2hp": "megatron.core.ops.ssm.gated_delta.common",
        "tensor_a2a_hp2cp": "megatron.core.ops.ssm.gated_delta.common",
        "torch_chunk_gated_delta_rule": "megatron.core.ops.ssm.gated_delta.gdn",
        "torch_chunk_gdn2": "megatron.core.ops.ssm.gated_delta.gdn2",
    },
)
