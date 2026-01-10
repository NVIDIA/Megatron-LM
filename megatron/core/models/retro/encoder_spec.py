# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Specs for Retro encoder."""

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.retro.config import RetroConfig
from megatron.core.models.retro.encoder_attention import (
    RetroEncoderBiasDropoutAdd,
    RetroEncoderCrossAttention,
    RetroEncoderLayerNorm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import CrossAttentionSubmodules, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_retro_encoder_layer_te_submodules() -> TransformerLayerSubmodules:
    """Retro encoder TE spec (uses Transformer Engine components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.

    Returns:
        The submodules of TransformerLayer using Transformer Engine modules.
    """
    submodules = get_gpt_layer_with_transformer_engine_submodules()
    submodules.pre_cross_attn_layernorm = TENorm
    submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding},
        submodules=CrossAttentionSubmodules(
            linear_q=TEColumnParallelLinear,
            linear_kv=TEColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
    )
    submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    submodules.pre_mlp_layernorm = ModuleSpec(module=RetroEncoderLayerNorm, submodules=TENorm)
    submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )
    return submodules


def get_retro_encoder_layer_te_spec() -> ModuleSpec:
    """Retro encoder TE spec (uses Transformer Engine components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.

    Returns:
        A module spec of Transformer Engine modules.
    """

    return ModuleSpec(module=TransformerLayer, submodules=get_retro_encoder_layer_te_submodules())


def get_retro_encoder_layer_local_submodules() -> TransformerLayerSubmodules:
    """Retro encoder local spec (uses Megatron-Core components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.

    Returns:
        The submodules of TransformerLayer using Megatron-Core local modules.
    """
    submodules = get_gpt_layer_local_submodules()
    submodules.pre_cross_attn_layernorm = LNImpl
    submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding},
        submodules=CrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            linear_kv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
        ),
    )
    submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    submodules.pre_mlp_layernorm = ModuleSpec(module=RetroEncoderLayerNorm, submodules=LNImpl)
    submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
    )
    submodules.sharded_state_dict_keys_map = {
        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_'
    }  # pre_mlp_layernorm doesn't need remapping


def get_retro_encoder_layer_local_spec() -> ModuleSpec:
    """Retro encoder local spec (uses Megatron-Core components).

    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.

    Returns:
        A module spec of Megatron-Core local modules.
    """

    return ModuleSpec(
        module=TransformerLayer, submodules=get_retro_encoder_layer_local_submodules()
    )


def get_retro_encoder_block_spec(
    config: RetroConfig, use_transformer_engine: bool
) -> TransformerBlockSubmodules:
    """Retro encoder block spec.

    The retro encoder block consists of one customized Retro encoder layer
    (layer 1), and all of the following layers are standard GPT layers.

    Args:
      config (RetroConfig): Retro config.
      use_transformer_engine (bool): If True, use Transformer Engine (instead of local modules).

    Returns:
        Transformer block submodules for the given spec.
    """

    # Num layers.
    num_layers = config.retro_encoder_num_layers
    retro_layer_numbers = [1]

    # Layer specs.
    gpt_layer_submodules = (
        get_gpt_layer_with_transformer_engine_submodules()
        if use_transformer_engine
        else get_gpt_layer_local_submodules()
    )
    retro_layer_submodules = (
        get_retro_encoder_layer_te_submodules()
        if use_transformer_engine
        else get_retro_encoder_layer_local_submodules()
    )
    for submodules in (gpt_layer_submodules, retro_layer_submodules):
        submodules.self_attention.params["attn_mask_type"] = AttnMaskType.padding
        assert isinstance(submodules.self_attention.submodules, SelfAttentionSubmodules)
        submodules.self_attention.submodules.core_attention = ModuleSpec(
            module=TEDotProductAttention if use_transformer_engine else DotProductAttention,
            params={"attention_dropout": config.retro_encoder_attention_dropout},
        )
    gpt_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=gpt_layer_submodules,
        params={"hidden_dropout": config.retro_encoder_hidden_dropout},
    )
    retro_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=retro_layer_submodules,
        params={"hidden_dropout": config.retro_encoder_hidden_dropout},
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
