# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    AttentionConfig,
    HeterogeneousTransformerConfig,
    MLPConfig,
    TransformerBlockConfig,
)
from megatron.core.transformer.heterogeneous.linear_replacements import ColumnParallelLinearGathered
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import is_te_min_version

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
    from megatron.core.transformer.heterogeneous.linear_replacements import (
        TELayerNormColumnParallelLinearGathered,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def _get_layer_norm(config: AttentionConfig | MLPConfig, use_te: bool, normalization: str):
    # RMSNorm is not supported in FusedLayerNorm
    ln_impl = LNImpl if normalization == "LayerNorm" else WrappedTorchNorm

    # We don't use layernorm when the attention/mlp is no-op or
    # when we are using TE (the layernorm is fused with the first linear).
    return IdentityOp if use_te or config.no_op else ln_impl


def _get_qk_layernorm(use_te: bool, normalization: str):
    # RMSNorm is not supported in FusedLayerNorm
    ln_impl = LNImpl if normalization == "LayerNorm" else WrappedTorchNorm

    if use_te:
        if is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            qk_norm = TENorm
        else:
            qk_norm = ln_impl
    else:
        qk_norm = ln_impl

    return qk_norm


def _get_heterogenous_attention_spec(
    attn_config: AttentionConfig, use_te: bool, qk_layernorm: bool, normalization: str
):
    if attn_config.no_op:
        self_attention = ModuleSpec(module=IdentityOp)
    elif attn_config.replace_with_linear:
        self_attention = ModuleSpec(
            module=(
                TELayerNormColumnParallelLinearGathered if use_te else ColumnParallelLinearGathered
            ),
            params={"tp_comm_buffer_name": "linear_attn"},
        )
    else:
        ln = _get_qk_layernorm(use_te, normalization) if qk_layernorm else IdentityOp
        self_attention = ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                core_attention=TEDotProductAttention if use_te else DotProductAttention,
                linear_proj=TERowParallelLinear if use_te else RowParallelLinear,
                q_layernorm=ln,
                k_layernorm=ln,
            ),
        )
    return self_attention


def _get_heterogenous_mlp_spec(mlp_config: MLPConfig, use_te: bool):
    if mlp_config.no_op:
        mlp = ModuleSpec(module=IdentityOp)
    elif mlp_config.replace_with_linear:
        mlp = ModuleSpec(
            module=(
                TELayerNormColumnParallelLinearGathered if use_te else ColumnParallelLinearGathered
            ),
            params={"tp_comm_buffer_name": "linear_mlp"},
        )
    else:
        mlp = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    return mlp


def _get_sharded_state_dict_keys_map(block_config: TransformerBlockConfig, use_te: bool):
    """
    Generate a mapping of sharded state dictionary keys.
    Mapping in case of not using Transformer Engine with regular attention and mlp.
    Args:
        block_config (TransformerBlockConfig): The configuration of the transformer block.
        use_te (bool): Flag indicating whether to use Transformer Engine.

    Returns:
        dict: A dictionary mapping sharded state dictionary keys.
    """
    mapping = {}
    if not use_te:
        if block_config.attention.num_query_groups is not None:
            mapping.update({'input_layernorm.': 'self_attention.linear_qkv.layer_norm_'})
        if block_config.attention.replace_with_linear:
            mapping.update({'input_layernorm.': 'self_attention.layer_norm_'})
        if block_config.mlp.ffn_hidden_size is not None:
            mapping.update({'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_'})
        if block_config.mlp.replace_with_linear:
            mapping.update({'pre_mlp_layernorm.': 'mlp.layer_norm_'})
    return mapping


def get_gpt_heterogeneous_layer_spec(config: HeterogeneousTransformerConfig, use_te: bool = False):
    """
    Returns a list of ModuleSpec objects for the transformer layers in the heterogeneous model.

    Args:
        config (HeterogeneousTransformerConfig): Heterogeneous Transformer configuration.
        use_te (bool, optional): To use Transformer-Engine. Defaults to False.

    Returns:
        ModuleSpec: Module specification for the transformer layers
    """
    qk_layernorm = config.qk_layernorm
    layer_specs = [
        ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=_get_layer_norm(
                    block_params.attention, use_te, config.normalization
                ),
                self_attention=_get_heterogenous_attention_spec(
                    block_params.attention, use_te, qk_layernorm, config.normalization
                ),
                self_attn_bda=(
                    get_bias_dropout_add if not block_params.attention.no_op else IdentityFuncOp
                ),
                pre_mlp_layernorm=_get_layer_norm(block_params.mlp, use_te, config.normalization),
                mlp=_get_heterogenous_mlp_spec(block_params.mlp, use_te),
                mlp_bda=get_bias_dropout_add if not block_params.mlp.no_op else IdentityFuncOp,
                sharded_state_dict_keys_map=_get_sharded_state_dict_keys_map(block_params, use_te),
            ),
        )
        for block_params in config.per_block_parameters
    ]

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Submodules layer_norm determines the type of layernorm used in the last layernorm
    if use_te:
        layer_norm = TENorm
    else:
        layer_norm = LNImpl if config.normalization == "LayerNorm" else WrappedTorchNorm
    return TransformerBlockSubmodules(layer_specs, layer_norm=layer_norm)
