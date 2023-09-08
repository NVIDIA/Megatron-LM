# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

# from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
# from megatron.core.models.gpt.gpt_decoder_spec import get_gpt_decoder_spec as get_gpt_layer_spec
# from megatron.core.transformer.attention import CrossAttention, CrossAttentionSpec
# from megatron.core.transformer.custom_layers.transformer_engine import (
#     TEDotProductAttention,
#     TELayerNormColumnParallelLinear,
#     TELayerNormMLP,
#     TERowParallelLinear,
# )
# from megatron.core.transformer.enums import AttnMaskType
# from megatron.core.transformer.mlp import MLP
# from megatron.core.transformer.spec_utils import ModuleSpec
# from megatron.core.transformer.transformer_layer import TransformerLayerSpec

# from .attn import (
#     RetroDecoderCrossAttention,
#     RetroDecoderBiasDropoutAdd,
#     RetroDecoderLayerNorm,
#     RetroEncoderCrossAttention,
#     RetroEncoderBiasDropoutAdd,
#     RetroEncoderLayerNorm,
# )

# >>>
from lutil import pax
# <<<


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
    # spec.cross_attn_bda=get_bias_dropout_add
    spec.cross_attn_bda=ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.post_cross_attn_layernorm=ModuleSpec(module=RetroEncoderLayerNorm)
    spec.ln_mlp=ModuleSpec(module=MLP)
    # pax("spec")
    return spec

# def get_encoder_layer_specs(config, spec):
def get_retro_encoder_block_spec(config)

    num_layers = self.config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number in retro_layer_numbers:
            layer_specs.append(self.spec.retro_encoder_layer_spec)
        else:
            layer_specs.append(self.spec.gpt_layer_spec)

    pax({
        "config" : config,
        "spec" : spec,
        "num_layers" : num_layers,
        "retro_layer_numbers" : retro_layer_numbers,
        # "layer_specs" : layer_specs,
        "attn specs" : [ s.cross_attention for s in layer_specs ],
    })

    return layer_specs


# @dataclass
# class RetroEncoderModelSpec:
#     gpt_layer_spec: TransformerLayerSpec = None
#     retro_encoder_layer_spec: TransformerLayerSpec = None


# def get_encoder_model_spec() -> RetroEncoderModelSpec:
#     spec = RetroEncoderModelSpec(
#         gpt_layer_spec = get_gpt_layer_spec(),
#         retro_encoder_layer_spec = get_encoder_layer_spec(),
#     )
#     # pax("spec")
#     return spec


