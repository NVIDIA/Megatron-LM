# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec


def get_linear_attention_module_spec_for_backend(
    backend: BackendSpecProvider, linear_attention_type: str, normalization: Optional[str] = None
) -> ModuleSpec:
    """Helper function to get module spec for Linear Attention"""
    rms_norm = normalization == "RMSNorm"
    if linear_attention_type == "gated_delta_net":
        attention = ModuleSpec(
            module=GatedDeltaNet,
            submodules=GatedDeltaNetSubmodules(
                in_proj=backend.column_parallel_layer_norm_linear(),
                out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
                out_proj=backend.row_parallel_linear(),
            ),
            metainfo={"fuse_input_layernorm": True},
        )
    else:
        raise ValueError(f"Invalid linear attention type: {linear_attention_type}")
    return attention
