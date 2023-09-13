# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_decoder_spec import get_gpt_layer_spec
from megatron.core.transformer.attention import CrossAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP
from megatron.core.models.retro.attn import BaseRetroCrossAttention
from megatron.core.models.retro.encoder import get_retro_encoder_block_spec
from megatron.core.transformer import (
    get_num_layers_to_build,
    ModuleSpec,
    TransformerBlockSpec,
    TransformerConfig,
    TransformerLayerSpec,
)

from .attn import (
    RetroDecoderBiasDropoutAdd,
    RetroDecoderCrossAttention,
    RetroDecoderLayerNorm,
)


def get_retro_decoder_layer_spec(encoder_block_spec=None) -> TransformerLayerSpec:
    spec = get_gpt_layer_spec()
    spec.cross_attention=CrossAttentionSpec(
        module=RetroDecoderCrossAttention,
        params={
            "encoder_block_spec" : encoder_block_spec,
        },
        layernorm_linear_q=TELayerNormColumnParallelLinear,
        layernorm_linear_kv=TELayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
    )
    spec.cross_attn_bda=ModuleSpec(module=RetroDecoderBiasDropoutAdd)
    spec.post_cross_attn_layernorm=ModuleSpec(module=RetroDecoderLayerNorm)
    spec.ln_mlp=ModuleSpec(module=MLP)
    return spec


def get_retro_decoder_block_spec(config: TransformerConfig) -> TransformerBlockSpec:

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
    gpt_layer_spec = get_gpt_layer_spec()
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
    block_spec = TransformerBlockSpec(layers=layer_specs)

    return block_spec
