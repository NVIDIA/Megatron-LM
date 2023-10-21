from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSpec,
    SelfAttention,
    SelfAttentionSpec,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELayerNormMLP,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import (
    TransformerBlockSpec,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import TransformerLayerSpec


def encoder_model_with_transformer_engine_default_spec() -> TransformerLayerSpec:
    return TransformerLayerSpec(
        self_attention=SelfAttentionSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            layernorm_linear_qkv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        self_attn_bda=get_bias_dropout_add,
        ln_mlp=TELayerNormMLP,
        mlp_bda=get_bias_dropout_add,
    )


def decoder_model_with_transformer_engine_default_spec() -> TransformerLayerSpec:
    return TransformerLayerSpec(
        self_attention=SelfAttentionSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            layernorm_linear_qkv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        self_attn_bda=get_bias_dropout_add,
        cross_attention=CrossAttentionSpec(
            module=CrossAttention,
            layernorm_linear_q=TELayerNormColumnParallelLinear,
            layernorm_linear_kv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
        cross_attn_bda=get_bias_dropout_add,
        ln_mlp=TELayerNormMLP,
        mlp_bda=get_bias_dropout_add,
        # post_mlp_layernorm = TENorm,
    )


def get_t5_encoder_block_spec(config) -> TransformerBlockSpec:
    num_layers = get_num_layers_to_build(config)
    layer_spec = encoder_model_with_transformer_engine_default_spec()
    block_spec = TransformerBlockSpec([layer_spec] * num_layers)
    return block_spec


def get_t5_decoder_block_spec(config) -> TransformerBlockSpec:
    num_layers = get_num_layers_to_build(config)
    layer_spec = decoder_model_with_transformer_engine_default_spec()
    block_spec = TransformerBlockSpec([layer_spec] * num_layers)
    return block_spec
