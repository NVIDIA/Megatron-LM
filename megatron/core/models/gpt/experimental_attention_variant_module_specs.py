# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec


def is_linear_attention_variant(experimental_attention_variant: str) -> bool:
    """Check if the experimental attention variant is a linear attention variant."""
    linear_attention_variants = ["gated_delta_net"]
    return experimental_attention_variant in linear_attention_variants


def get_gated_delta_net_module_spec_for_backend(
    backend: BackendSpecProvider, normalization: Optional[str] = None
) -> ModuleSpec:
    """Helper function to get module spec for Linear Attention"""
    rms_norm = normalization == "RMSNorm"
    attention = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_layer_norm_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": True},
    )
    return attention


def get_experimental_attention_variant_module_spec_for_backend(
    backend: BackendSpecProvider,
    sharded_state_dict_keys_map: dict,
    experimental_attention_variant: Optional[str] = None,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    mla_down_proj_use_column_parallel: Optional[bool] = False,
    normalization: Optional[str] = None,
    fallback_to_eager_attn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for Attention"""
    if experimental_attention_variant == "gated_delta_net":
        return get_gated_delta_net_module_spec_for_backend(
            backend=backend, normalization=normalization
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {experimental_attention_variant}"
        )
