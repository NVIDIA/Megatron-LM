# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_vit_layer_with_transformer_engine_spec() -> ModuleSpec:
    '''
    Returns ViT layer spec with Transformer Engine layers
    '''
    mlp = _get_mlp_module_spec(use_te=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_vit_layer_with_local_spec() -> ModuleSpec:
    '''
    Returns ViT layer spec with Mcore local layers
    '''
    mlp = _get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )
