# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.retro.config import RetroConfig
from megatron.core.models.retro.encoder_attention import (
    RetroEncoderBiasDropoutAdd,
    RetroEncoderCrossAttention,
    RetroEncoderLayerNorm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import CrossAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules


def get_retro_encoder_layer_te_spec() -> ModuleSpec:
    """Retro encoder TE spec (uses Transformer Engine components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.
    """
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.pre_cross_attn_layernorm = TENorm
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding,},
        submodules=CrossAttentionSubmodules(
            linear_q=TEColumnParallelLinear,
            linear_kv=TEColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm = ModuleSpec(module=RetroEncoderLayerNorm, submodules=TENorm,)
    spec.submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear,
        ),
    )
    return spec


def get_retro_encoder_layer_local_spec() -> ModuleSpec:
    """Retro encoder local spec (uses Megatron-Core components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.
    """
    spec = get_gpt_layer_local_spec()
    spec.submodules.pre_cross_attn_layernorm = FusedLayerNorm
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding,},
        submodules=CrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            linear_kv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm = ModuleSpec(
        module=RetroEncoderLayerNorm, submodules=FusedLayerNorm,
    )
    spec.submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,),
    )
    return spec


def get_retro_encoder_block_spec(
    config: RetroConfig, use_transformer_engine: bool
) -> TransformerBlockSubmodules:

    """Retro encoder block spec.

    The retro encoder block consists of one customized Retro encoder layer
    (layer 1), and all of the following layers are standard GPT layers.

    Arguments:
      config (RetroConfig): Retro config.

      use_transformer_engine (bool): If True, use Transformer Engine (instead
      of local modules.
    """

    # Num layers.
    num_layers = config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    # Layer specs.
    gpt_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec()
        if use_transformer_engine
        else get_gpt_layer_local_spec()
    )
    get_retro_encoder_layer_spec = (
        get_retro_encoder_layer_te_spec
        if use_transformer_engine
        else get_retro_encoder_layer_local_spec
    )
    retro_layer_spec = get_retro_encoder_layer_spec()
    for spec in (gpt_layer_spec, retro_layer_spec):
        spec.params["hidden_dropout"] = config.retro_encoder_hidden_dropout
        spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.padding
        spec.submodules.self_attention.submodules.core_attention = ModuleSpec(
            module=TEDotProductAttention if use_transformer_engine else DotProductAttention,
            params={"attention_dropout": config.retro_encoder_attention_dropout,},
        )

    layer_specs = []
    for layer_number in range(1, num_layers + 1):
        if layer_number in retro_layer_numbers:
            layer_specs.append(retro_layer_spec)
        else:
            layer_specs.append(gpt_layer_spec)

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs)

    return block_spec
