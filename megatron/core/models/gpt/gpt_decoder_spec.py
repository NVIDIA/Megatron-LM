# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELayerNormMLP,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import (
    get_num_layers_to_build,
    TransformerBlockSpec,
)
from megatron.core.transformer.transformer_layer import TransformerLayerSpec


def get_gpt_layer_spec() -> TransformerLayerSpec:
    return TransformerLayerSpec(
        self_attention=SelfAttentionSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            layernorm_linear_qkv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        self_attn_bda=get_bias_dropout_add,
        ln_mlp=TELayerNormMLP,
        mlp_bda=get_bias_dropout_add,
    )


def get_gpt_block_spec() -> TransformerBlockSpec:
    num_layers = get_num_layers_to_build()
    layer_spec = get_gpt_layer_spec()
    block_spec = TransformerBlockSpec([layer_spec] * num_layers)
    return block_spec
