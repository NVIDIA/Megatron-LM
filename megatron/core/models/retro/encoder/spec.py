# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.retro.attn import BaseRetroCrossAttention
from megatron.core.transformer import (
    ModuleSpec,
    TransformerBlock,
    TransformerBlockSubmodules,
    TransformerConfig,
)
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from .attn import (
    RetroEncoderCrossAttention,
    RetroEncoderBiasDropoutAdd,
    RetroEncoderLayerNorm,
)

# >>>
from lutil import pax
# <<<


def get_retro_encoder_layer_spec() -> ModuleSpec:
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.cross_attention=ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={
            "attn_mask_type" : AttnMaskType.padding,
        },
        submodules=CrossAttentionSubmodules(
            linear_q=TELayerNormColumnParallelLinear,
            linear_kv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        )
    )
    spec.submodules.cross_attn_bda=ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm=ModuleSpec(module=RetroEncoderLayerNorm)
    spec.submodules.mlp=ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    # >>>
    # pax({
    #     "spec" : spec,
    #     "spec / submodules" : spec.submodules,
    #     "ca subs" : spec.submodules.cross_attention.submodules,
    #     "mlp subs" : spec.submodules.mlp.submodules,
    # })
    # <<<
    return spec


def get_retro_encoder_block_spec(config: TransformerConfig) -> ModuleSpec:

    # Num layers.
    num_layers = config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    # Layer specs.
    gpt_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    retro_layer_spec = get_retro_encoder_layer_spec()
    for spec in (gpt_layer_spec, retro_layer_spec):
        spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.padding

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number in retro_layer_numbers:
            layer_specs.append(retro_layer_spec)
        else:
            layer_specs.append(gpt_layer_spec)

    # Block spec.
    block_spec = ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(layer_specs=layer_specs),
    )

    # >>>
    # pax({
    #     "block_spec" : block_spec,
    #     "cross attns" : [ s.submodules.cross_attention
    #                       for s in block_spec.submodules.layer_specs ],
    # })
    # <<<

    return block_spec
