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
from megatron.core.transformer.spec_utils import ModuleSpec #, build_module
from megatron.core.transformer.transformer_layer import TransformerLayerSpec

from .attn import (
    RetroDecoderWithRetrieverCrossAttention,
    RetroDecoderWithRetrieverBiasDropoutAdd,
    RetroDecoderWithRetrieverLayernorm,
)

# >>>
from lutil import pax
# <<<


# def get_decoder_with_retriever_spec() -> TransformerLayerSpec:
#     layer_spec = TransformerLayerSpec(
#         self_attention=SelfAttentionSpec(
#             module=SelfAttention,
#             params={"attn_mask_type": AttnMaskType.causal},
#             layernorm_linear_qkv=TELayerNormColumnParallelLinear,
#             dot_product_attention=TEDotProductAttention,
#             linear_proj=TERowParallelLinear,
#         ),
#         self_attn_bda=get_bias_dropout_add,
#         ln_mlp=TELayerNormMLP,
#         mlp_bda=get_bias_dropout_add,
#     )
#     return layer_spec
# class RetroDecoderWithRetrieverSpec(GPTSpec):
#     add_retriever = True
#     cross_attention=CrossAttentionSpec(
#         module=RetroDecoderWithRetrieverCrossAttention,
#         params={"attn_mask_type": AttnMaskType.causal},
#         layernorm_linear_qkv=TELayerNormColumnParallelLinear,
#         dot_product_attention=TEDotProductAttention,
#         linear_proj=TERowParallelLinear,
#     )

def get_decoder_layer_spec(add_retriever=False) -> TransformerLayerSpec:
    spec = get_gpt_layer_spec()
    # spec.add_retriever = True
    # self_attention=SelfAttentionSpec(
    #     module=SelfAttention,
    #     params={"attn_mask_type": AttnMaskType.causal},
    #     layernorm_linear_qkv=TELayerNormColumnParallelLinear,
    #     dot_product_attention=TEDotProductAttention,
    #     linear_proj=TERowParallelLinear,
    # ),
    spec.cross_attention=CrossAttentionSpec(
        module=RetroDecoderWithRetrieverCrossAttention,
        params={
            "attn_mask_type" : AttnMaskType.causal,
            "add_retriever" : add_retriever,
        },
        layernorm_linear_q=TELayerNormColumnParallelLinear,
        layernorm_linear_kv=TELayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
    )
    # spec.cross_attn_bda=get_bias_dropout_add
    spec.cross_attn_bda=ModuleSpec(
        module=RetroDecoderWithRetrieverBiasDropoutAdd,
        params=None,
    )
    spec.post_cross_attn_layernorm=ModuleSpec(
        module=RetroDecoderWithRetrieverLayernorm,
        params=None,
    )
    # pax("spec")
    return spec


def get_decoder_with_retriever_layer_spec() -> TransformerLayerSpec:
    return get_decoder_layer_spec(add_retriever=True)


@dataclass
class RetroModelSpec:
    gpt_layer_spec: TransformerLayerSpec = None
    retro_decoder_with_retriever_layer_spec: TransformerLayerSpec = None
    retro_decoder_layer_spec: TransformerLayerSpec = None
    retro_encoder_layer_spec: TransformerLayerSpec = None

# def class RetroModelSpec(ModuleSpec):
#     decoder_with_retriever: RetroDeocderWithRetrieverSpec = 
# def get_retro_model_spec() -> RetroModelSpec:
def get_model_spec() -> RetroModelSpec:
    spec = RetroModelSpec(
        gpt_layer_spec = get_gpt_layer_spec(),
        retro_decoder_with_retriever_layer_spec = get_decoder_with_retriever_layer_spec(),
        retro_decoder_layer_spec = get_decoder_layer_spec(),
        retro_encoder_layer_spec = get_encoder_layer_spec(),
    )
    pax("spec")
    return spec
