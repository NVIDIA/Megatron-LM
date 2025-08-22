# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from functools import partial

import torch

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from examples.multimodal.layer_scaling import LayerScalingTransformerLayer, get_bias_dropout_add_layer_scaling

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )


def get_norm_mlp_module_spec_te() -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )


def get_radio_g_layer_spec(normalization) -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask
    if normalization == "LayerNorm":
        norm = LNImpl
    elif normalization == "RMSNorm":
        if HAVE_TE:
            norm = TENorm
        else:
            assert is_torch_min_version("2.4.0"), "Torch version >= 2.4.0 is required for RMSNorm"
            if HAVE_APEX:
                warnings.warn(f'Apex does not support RMSNorm. Falling back to Torch Norm')
            norm = WrappedTorchNorm
    else:
        raise RuntimeError("unknown normalization", normalization)

    mlp = get_mlp_module_spec(use_te=False)  # doesn't include norm.

    return ModuleSpec(
        module=LayerScalingTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add_layer_scaling,
            pre_mlp_layernorm=norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add_layer_scaling,
        ),
    )


def get_radio_g_layer_spec_te() -> ModuleSpec:
    attn_mask_type = AttnMaskType.no_mask

    mlp = get_norm_mlp_module_spec_te()
    return ModuleSpec(
        module=LayerScalingTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add_layer_scaling,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add_layer_scaling,
        ),
    )
