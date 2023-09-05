from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELayerNormMLP,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_layer import TransformerLayerSpec


def get_gpt_decoder_spec() -> TransformerLayerSpec:
    layer_spec = TransformerLayerSpec(
        self_attention=SelfAttentionSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            layernorm_linear_qkv=TELayerNormColumnParallelLinear,
            dot_product_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        self_attn_bda=get_bias_dropout_add,
        ln_mlp=TELayerNormMLP,
        mlp_bda=get_bias_dropout_add,
    )
    # >>>
    # from lutil import pax
    # pax("layer_spec", {
    #     # "layer_spec / self_attn_bda" : self_attn_bda,
    #     # "get_bias_dropout_add" : get_bias_dropout_add,
    #     # "tls" : TransformerLayerSpec(),
    # })
    # <<<
    return layer_spec
