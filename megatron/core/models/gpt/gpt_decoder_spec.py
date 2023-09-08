from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSpec
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.layernorm_mlp import LayerNormMLP
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELayerNormMLP,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_layer import TransformerLayerSpec


gpt_model_with_transformer_engine_default_spec = TransformerLayerSpec(
    self_attention=SelfAttentionSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        linear_qkv=TELayerNormColumnParallelLinear,
        dot_product_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
    ),
    self_attn_bda=get_bias_dropout_add,
    mlp=TELayerNormMLP,
    mlp_bda=get_bias_dropout_add,
)

gpt_model_vanilla_spec = TransformerLayerSpec(
    self_attention=SelfAttentionSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        linear_qkv=ColumnParallelLinear,
        dot_product_attention=DotProductAttention,
        linear_proj=RowParallelLinear,
    ),
    self_attn_bda=get_bias_dropout_add,
    mlp=LayerNormMLP,
    mlp_bda=get_bias_dropout_add,
)