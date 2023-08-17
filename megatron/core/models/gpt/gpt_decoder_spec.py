from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELayernormMLP,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import SelfAttentionSpec, TransformerLayerSpec


def get_gpt_decoder_spec() -> TransformerLayerSpec:
    layer_spec = TransformerLayerSpec(
        self_attention=SelfAttentionSpec(
            module_path_or_module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            layernorm_linear_qkv=TELayerNormColumnParallelLinear,
            dot_product_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        self_attn_bda=get_bias_dropout_add,
        ln_mlp=TELayernormMLP,
        mlp_bda=get_bias_dropout_add,
    )
    return layer_spec
