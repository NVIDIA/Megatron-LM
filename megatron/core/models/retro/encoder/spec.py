# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

from megatron.core.models.gpt.gpt_decoder_spec import get_gpt_layer_spec
from megatron.core.models.retro.attn import BaseRetroCrossAttention
from megatron.core.transformer import (
    ModuleSpec,
    TransformerBlockSpec,
    TransformerConfig,
    TransformerLayerSpec,
)
from megatron.core.transformer.attention import CrossAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP

from .attn import (
    RetroEncoderCrossAttention,
    RetroEncoderBiasDropoutAdd,
    RetroEncoderLayerNorm,
)


def get_retro_encoder_layer_spec() -> TransformerLayerSpec:
    spec = get_gpt_layer_spec()
    spec.cross_attention=CrossAttentionSpec(
        module=RetroEncoderCrossAttention,
        params={
            "attn_mask_type" : AttnMaskType.padding,
        },
        layernorm_linear_q=TELayerNormColumnParallelLinear,
        layernorm_linear_kv=TELayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
    )
    spec.cross_attn_bda=ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.post_cross_attn_layernorm=ModuleSpec(module=RetroEncoderLayerNorm)
    spec.ln_mlp=ModuleSpec(module=MLP)
    return spec

def get_retro_encoder_block_spec(config: TransformerConfig) -> TransformerBlockSpec:

    # Num layers.
    num_layers = config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    # Layer specs.
    gpt_layer_spec = get_gpt_layer_spec()
    retro_layer_spec = get_retro_encoder_layer_spec()
    gpt_layer_spec.self_attention.params["attn_mask_type"] = AttnMaskType.padding
    retro_layer_spec.self_attention.params["attn_mask_type"] = AttnMaskType.padding

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number in retro_layer_numbers:
            layer_specs.append(retro_layer_spec)
        else:
            layer_specs.append(gpt_layer_spec)

    # Block spec.
    block_spec = TransformerBlockSpec(layers=layer_specs)

    return block_spec
