from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def encoder_model_with_transformer_engine_default_spec() -> ModuleSpec:
    """T5 encoder TE spec (uses Transformer Engine components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def decoder_model_with_transformer_engine_default_spec() -> ModuleSpec:
    """T5 decoder TE spec (uses Transformer Engine components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=TENorm,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def encoder_model_with_local_spec() -> ModuleSpec:
    """T5 encoder local spec (uses Megatron-Core components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def decoder_model_with_local_spec() -> ModuleSpec:
    """T5 decoder local spec (uses Megatron-Core components)."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=FusedLayerNorm,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                submodules=CrossAttentionSubmodules(
                    linear_q=ColumnParallelLinear,
                    linear_kv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def get_t5_encoder_with_transformer_engine_block_spec(
    num_layers: int,
) -> TransformerBlockSubmodules:
    """T5 encoder block spec for Transformer Engine

    Args:
      config (TransformerConfig): config, containing number of layers for encoder
    """

    layer_spec = encoder_model_with_transformer_engine_default_spec()
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers)
    return block_spec


def get_t5_decoder_with_transformer_engine_block_spec(
    num_layers: int,
) -> TransformerBlockSubmodules:
    """T5 decoder block spec for Transformer Engine

    Args:
      config (TransformerConfig): config, containing number of layers for decoder
    """

    layer_spec = decoder_model_with_transformer_engine_default_spec()
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers)
    return block_spec


def get_t5_encoder_with_local_block_spec(num_layers: int) -> TransformerBlockSubmodules:
    """T5 encoder block spec for local (uses Megatron-Core components)

    Args:
      num_layers (int): number of encoder layers
    """

    layer_spec = encoder_model_with_local_spec()
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers)
    return block_spec


def get_t5_decoder_with_local_block_spec(num_layers: int) -> TransformerBlockSubmodules:
    """T5 decoder block spec for local (uses Megatron-Core components)

    Args:
      num_layers (int): number of decoder layers
    """

    layer_spec = decoder_model_with_local_spec()
    block_spec = TransformerBlockSubmodules([layer_spec] * num_layers)
    return block_spec
