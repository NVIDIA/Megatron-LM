# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core import parallel_state
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.retro.config import RetroConfig
from megatron.core.models.retro.decoder_attention import (
    RetroDecoderBiasDropoutAdd,
    RetroDecoderCrossAttention,
)
from megatron.core.models.retro.encoder_spec import get_retro_encoder_block_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import (
    ModuleSpec,
    TransformerBlock,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention


def get_retro_decoder_layer_te_spec(encoder_block_spec: ModuleSpec = None) -> ModuleSpec:
    """Retro decoder TE spec (uses Transformer Engine components).

    A Retro decoder layer uses custom attention and bias-dropout-add operators
    to perform chunked-cross attention. Additionally, the first Retro decoder
    layer instantiates an entire encoder transformer block. As such, the decoder
    cross attention module takes an optional encoder block spec, which is only
    provided for the first Retro decoder layer.
    """
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.pre_cross_attn_layernorm = TENorm
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroDecoderCrossAttention,
        params={"encoder_block_spec": encoder_block_spec,},
        submodules=CrossAttentionSubmodules(
            linear_q=TEColumnParallelLinear,
            linear_kv=TEColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroDecoderBiasDropoutAdd)
    return spec


def get_retro_decoder_layer_local_spec(encoder_block_spec: ModuleSpec = None) -> ModuleSpec:
    """Retro decoder local spec (uses Megatron-Core components).

    A Retro decoder layer uses custom attention and bias-dropout-add operators
    to perform chunked-cross attention. Additionally, the first Retro decoder
    layer instantiates an entire encoder transformer block. As such, the decoder
    cross attention module takes an optional encoder block spec, which is only
    provided for the first Retro decoder layer.
    """
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.pre_cross_attn_layernorm = FusedLayerNorm
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroDecoderCrossAttention,
        params={"encoder_block_spec": encoder_block_spec,},
        submodules=CrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            linear_kv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroDecoderBiasDropoutAdd)
    return spec


def get_retro_decoder_block_spec(
    config: RetroConfig, use_transformer_engine: bool,
) -> TransformerBlockSubmodules:

    """
    Retro decoder block implementation details:
    - The retro decoder block consists of interleaved GPT layers and customized
      Retro decoder layers.
    - The Retro decoder layers are spaced three layers apart, and start on layer
      6 or 9 (depending on the total number of layers).
    - The first decoder layer instantiates an encoder block, and it therefore
      passes in an encoder_block_spec.
    """

    # Num layers.
    assert (
        parallel_state.get_pipeline_model_parallel_world_size() == 1
    ), "retro does not currently support pipeline parallelism."
    assert (
        parallel_state.get_virtual_pipeline_model_parallel_world_size() is None
    ), "retro does not currently support virtual pipeline parallelism."
    num_layers = get_num_layers_to_build(config)

    # Retro layer numbers.
    retro_layer_start = 6 if num_layers <= 15 else 9
    retro_layer_numbers = list(range(retro_layer_start, num_layers + 1, 3))

    # Layer specs.
    gpt_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    get_retro_decoder_layer_spec = (
        get_retro_decoder_layer_te_spec
        if use_transformer_engine
        else get_retro_decoder_layer_local_spec
    )
    retro_layer_spec = get_retro_decoder_layer_spec()
    retro_layer_spec_with_retriever = get_retro_decoder_layer_spec(
        get_retro_encoder_block_spec(config, use_transformer_engine)
    )

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
        module=TransformerBlock, submodules=TransformerBlockSubmodules(layer_specs=layer_specs),
    )

    return block_spec
