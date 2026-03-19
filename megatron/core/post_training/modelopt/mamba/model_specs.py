# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.post_training.modelopt.layers import Norm
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


# Use this spec for ModelOpt PTQ and TensorRT-LLM export
def get_mamba_stack_modelopt_spec(
    local_core_attention: bool = False,
    remap_te_layernorm: bool = False,
    use_default_te_spec: bool = False,
) -> ModuleSpec:
    """Get the Mamba stack spec for ModelOpt PTQ and TensorRT-LLM export.

    When use_default_te_spec=False (default), this is the native local spec with TENorm
    from Transformer-Engine for the layernorm implementation (since FusedLayerNorm from
    apex has stopped supporting RMSNorm needed by llama). The remap_te_layernorm flag
    can be used to add sharded state_dict key remapping for TE-compatible checkpoint
    saving/loading.

    When use_default_te_spec=True, this returns the standard mamba_stack_spec from
    mamba_layer_specs.py which uses full TE modules (TELayerNormColumnParallelLinear,
    TERowParallelLinear, TEDotProductAttention, TENorm, moe_grouped_gemm=True).


    Args:
        local_core_attention: whether to use local DotProductAttention
            (only for use_default_te_spec=False)
        remap_te_layernorm: whether to perform sharded state_dict prefix mapping
            on layernorm (only for use_default_te_spec=False)
        use_default_te_spec: whether to use the default Transformer-Engine spec
    """
    if use_default_te_spec:
        from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec

        return mamba_stack_spec

    return _get_mamba_stack_local_spec(
        local_core_attention=local_core_attention, remap_te_layernorm=remap_te_layernorm
    )


def _get_mamba_stack_local_spec(
    local_core_attention: bool = False, remap_te_layernorm: bool = False
) -> ModuleSpec:
    """Get the Mamba stack spec with local (non-TE) modules.

    This is essentially the native local spec except for the layernorm implementation
    is using TENorm from Transformer-Engine.
    """
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
            norm=Norm,
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

    attn_mask_type = AttnMaskType.causal
    core_attention = DotProductAttention if local_core_attention else TEDotProductAttention
    attention_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
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
            pre_mlp_layernorm=Norm,
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

    moe_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=Norm,
            mlp=get_moe_module_spec(
                use_te=False, num_experts=8, moe_grouped_gemm=False  # Can be anything non None
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=mamba_layer,
            attention_layer=attention_layer,
            mlp_layer=mlp_layer,
            moe_layer=moe_layer,
        ),
    )
