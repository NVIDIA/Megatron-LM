# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.models.retro.attn import BaseRetroCrossAttention
from megatron.core.models.retro.encoder import get_retro_encoder_block_spec
from megatron.core.transformer import (
    get_num_layers_to_build,
    ModuleSpec,
    TransformerBlock,
    TransformerBlockSubmodules,
    TransformerConfig,
)

from .attn import (
    RetroDecoderBiasDropoutAdd,
    RetroDecoderCrossAttention,
    RetroDecoderLayerNorm,
)


def get_retro_decoder_layer_spec(encoder_block_submodules=None) -> ModuleSpec:
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.cross_attention=ModuleSpec(
        module=RetroDecoderCrossAttention,
        params={
            "encoder_block_submodules" : encoder_block_submodules,
        },
        submodules=CrossAttentionSubmodules(
            linear_q=TELayerNormColumnParallelLinear,
            linear_kv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda=ModuleSpec(module=RetroDecoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm=ModuleSpec(module=RetroDecoderLayerNorm)
    spec.submodules.mlp=ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    return spec


def get_retro_decoder_block_spec(config: TransformerConfig) -> TransformerBlockSubmodules:

    # Num layers.
    assert parallel_state.get_pipeline_model_parallel_world_size() == 1, \
        "retro does not currently support pipeline parallelism."
    assert parallel_state.get_virtual_pipeline_model_parallel_world_size() is None, \
        "retro does not currently support virtual pipeline parallelism."
    num_layers = get_num_layers_to_build(config)

    # Retro layer numbers.
    retro_layer_start = 6 if num_layers <= 15 else 9
    retro_layer_numbers = list(range(retro_layer_start, num_layers + 1, 3))

    # Layer specs.
    gpt_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    retro_layer_spec = get_retro_decoder_layer_spec()
    retro_layer_spec_with_retriever = \
        get_retro_decoder_layer_spec(get_retro_encoder_block_spec(config))

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number == retro_layer_numbers[0]:
            layer_specs.append(retro_layer_spec_with_retriever)
        elif layer_number in retro_layer_numbers:
            layer_specs.append(retro_layer_spec)
        else:
            layer_specs.append(gpt_layer_spec)

    # Block spec.
    block_spec = ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(layer_specs=layer_specs),
    )

    return block_spec
