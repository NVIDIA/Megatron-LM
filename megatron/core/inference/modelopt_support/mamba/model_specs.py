# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention, TENorm
    HAVE_TE = True
except ModuleNotFoundError:
    TEDotProductAttention = None
    TENorm = None
    HAVE_TE = False

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.torch_norm import WrappedTorchNorm


# Use this spec for ModelOpt PTQ and TensorRT-LLM export
def get_mamba_stack_modelopt_spec(
    local_core_attention: bool = False, remap_te_layernorm: bool = False
) -> ModuleSpec:
    """Mix the native spec with TENorm, if HAVE_TE

    This is essentially the native local spec except for the layernorm implementation
    is using TENorm from Transformer-Engine.
    """
    assert local_core_attention or HAVE_TE

    core_attention = DotProductAttention if local_core_attention else TEDotProductAttention
    layer_norm = TENorm if HAVE_TE else WrappedTorchNorm

    mamba_state_dict_keys_map = {}
    transformer_state_dict_keys_map = {}
    if remap_te_layernorm:
        mamba_state_dict_keys_map = {'norm.': 'mixer.in_proj.layer_norm_'}
        transformer_state_dict_keys_map = {
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        }

    mamba_layer = ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            norm=layer_norm,
            mixer=ModuleSpec(
                module=MambaMixer,
                submodules=MambaMixerSubmodules(
                    in_proj=ColumnParallelLinear, out_proj=RowParallelLinear
                ),
            ),
            mamba_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map=mamba_state_dict_keys_map,
        ),
    )

    core_attention = DotProductAttention if local_core_attention else TEDotProductAttention
    attention_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=layer_norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=core_attention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map=transformer_state_dict_keys_map,
        ),
    )

    mlp_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=layer_norm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map=transformer_state_dict_keys_map,
        ),
    )

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=mamba_layer, attention_layer=attention_layer, mlp_layer=mlp_layer
        ),
    )
