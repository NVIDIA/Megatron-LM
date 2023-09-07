# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass

# from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    # TELayerNormMLP,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_decoder_spec import get_gpt_decoder_spec as get_gpt_layer_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSpec

from .attn import (
    RetroDecoderCrossAttention,
    RetroDecoderBiasDropoutAdd,
    RetroDecoderLayerNorm,
    RetroEncoderCrossAttention,
    RetroEncoderBiasDropoutAdd,
    RetroEncoderLayerNorm,
)

# >>>
from lutil import pax
# <<<


def get_encoder_layer_spec() -> TransformerLayerSpec:
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
    # pax("spec")
    return spec


# def get_decoder_layer_spec(add_retriever) -> TransformerLayerSpec:
def get_decoder_layer_spec(encoder) -> TransformerLayerSpec:
    spec = get_gpt_layer_spec()
    spec.cross_attention=CrossAttentionSpec(
        module=RetroDecoderCrossAttention,
        params={
            "attn_mask_type" : AttnMaskType.causal,
            # "add_retriever" : add_retriever,
            "encoder" : encoder,
        },
        layernorm_linear_q=TELayerNormColumnParallelLinear,
        layernorm_linear_kv=TELayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
    )
    # spec.cross_attn_bda=get_bias_dropout_add
    spec.cross_attn_bda=ModuleSpec(module=RetroDecoderBiasDropoutAdd)
    spec.post_cross_attn_layernorm=ModuleSpec(module=RetroDecoderLayerNorm)
    return spec


@dataclass
class RetroEncoderModelSpec:
    gpt_layer_spec: TransformerLayerSpec = None
    retro_encoder_layer_spec: TransformerLayerSpec = None


@dataclass
class RetroDecoderModelSpec:
    gpt_layer_spec: TransformerLayerSpec = None
    retro_decoder_with_retriever_layer_spec: TransformerLayerSpec = None
    retro_decoder_layer_spec: TransformerLayerSpec = None


# def class RetroModelSpec(ModuleSpec):
#     decoder_with_retriever: RetroDeocderWithRetrieverSpec = 
# def get_retro_model_spec() -> RetroModelSpec:
# def get_model_spec(encoder) -> RetroModelSpec:
#     spec = RetroModelSpec(
#         gpt_layer_spec = get_gpt_layer_spec(),
#         retro_decoder_with_retriever_layer_spec = get_decoder_layer_spec(True),
#         retro_decoder_layer_spec = get_decoder_layer_spec(False),
#         retro_encoder_layer_spec = get_encoder_layer_spec(),
#     )
#     # pax("spec")
#     return spec


def get_encoder_model_spec() -> RetroEncoderModelSpec:
    spec = RetroEncoderModelSpec(
        gpt_layer_spec = get_gpt_layer_spec(),
        retro_encoder_layer_spec = get_encoder_layer_spec(),
    )
    # pax("spec")
    return spec


def get_decoder_model_spec(encoder) -> RetroDecoderModelSpec:
    spec = RetroDecoderModelSpec(
        gpt_layer_spec = get_gpt_layer_spec(),
        retro_decoder_with_retriever_layer_spec = get_decoder_layer_spec(encoder),
        retro_decoder_layer_spec = get_decoder_layer_spec(None),
    )
    # pax("spec")
    return spec


# >>>
# eof
# <<<
