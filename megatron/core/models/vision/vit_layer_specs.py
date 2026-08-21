# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from functools import partial

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import get_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

# One provider per backend; every choice below comes from the provider, not from a flag.
_te = get_backend("transformer_engine")
_local = get_backend("local")


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_vit_layer_with_transformer_engine_spec() -> ModuleSpec:
    """
    Returns ViT layer spec with Transformer Engine layers
    """
    mlp = _get_mlp_module_spec(use_te=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=_te.column_parallel_layer_norm_linear(),
                    core_attention=_te.core_attention(),
                    linear_proj=_te.row_parallel_linear(),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_vit_layer_with_local_spec() -> ModuleSpec:
    """
    Returns ViT layer spec with Mcore local layers
    """
    mlp = _get_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=_local.layer_norm(),
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=_local.column_parallel_linear(),
                    core_attention=_local.core_attention(),
                    linear_proj=_local.row_parallel_linear(),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=_local.layer_norm(),
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(use_te: bool = True):
    # Dense MLP w/ or w/o TE modules.
    return partial(
        MLP.as_mlp_submodule,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )
