# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import List

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gemma4.gemma4_block import Gemma4TransformerBlock
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.gemma4_attention import (
    Gemma4SelfAttention,
    Gemma4SelfAttentionSubmodules,
)
from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig
from megatron.core.transformer.gemma4_layer import (
    Gemma4TransformerLayer,
    Gemma4TransformerLayerSubmodules,
)
from megatron.core.transformer.gemma4_norm import (
    gemma4_rms_norm_builder,
    gemma4_rms_norm_scaleless_builder,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _gemma4_layer_spec(
    self_attn_cls: type,
    core_attention: type,
    linear_qkv: type,
    linear_proj: type,
    linear_fc1: type,
    linear_fc2: type,
    norm_builder,
    scaleless_norm_builder,
) -> ModuleSpec:
    """Assemble one Gemma4 decoder-layer :class:`ModuleSpec` for the given backend.

    ``norm_builder`` is the weighted RMSNorm builder (``Gemma4RMSNorm`` local / ``TENorm``
    TE); ``scaleless_norm_builder`` is the weightless v_norm (always ``Gemma4RMSNorm``
    with_scale=False so the per-head V normalization is bitwise-faithful and weightless).
    """
    return ModuleSpec(
        module=Gemma4TransformerLayer,
        submodules=Gemma4TransformerLayerSubmodules(
            input_layernorm=norm_builder,
            self_attention=ModuleSpec(
                module=self_attn_cls,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=Gemma4SelfAttentionSubmodules(
                    linear_qkv=linear_qkv,
                    core_attention=core_attention,
                    linear_proj=linear_proj,
                    q_layernorm=norm_builder,
                    k_layernorm=norm_builder,
                    v_layernorm=scaleless_norm_builder,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_self_attn_layernorm=norm_builder,
            pre_mlp_layernorm=norm_builder,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=norm_builder,
            per_layer_input_gate=ColumnParallelLinear,
            per_layer_projection=RowParallelLinear,
            post_per_layer_input_norm=norm_builder,
        ),
    )


def _build_block_spec(layer_spec: ModuleSpec, final_norm_builder, num_layers: int):
    """Wrap a single per-layer spec into a heterogeneous block spec (list per layer).

    Each layer carries the SAME submodule spec; the per-layer numeric config (kv_channels
    256/512, rope base, window) comes from ``get_config_for_layer`` at build time when
    ``heterogeneous_block_specs=True``. The layer-type role (KV bus producer/borrower,
    rope/mask selection) is resolved from the per-layer config inside the modules.
    """
    return ModuleSpec(
        module=Gemma4TransformerBlock,
        submodules=TransformerBlockSubmodules(
            layer_specs=[layer_spec] * num_layers, layer_norm=final_norm_builder
        ),
    )


def get_gemma4_layer_local_spec(config: Gemma4TransformerConfig) -> ModuleSpec:
    """Local (unfused) Gemma4 block spec — the BITWISE target.

    ``Gemma4RMSNorm`` everywhere (NOT ``WrappedTorchNorm``: HF uses ``pow(-0.5)`` not
    ``rsqrt``), Column/RowParallelLinear, and the unfused ``DotProductAttention`` core
    (bypassed by ``Gemma4SelfAttention``'s clean eager forward).
    """
    layer_spec = _gemma4_layer_spec(
        self_attn_cls=Gemma4SelfAttention,
        core_attention=DotProductAttention,
        linear_qkv=ColumnParallelLinear,
        linear_proj=RowParallelLinear,
        linear_fc1=ColumnParallelLinear,
        linear_fc2=RowParallelLinear,
        norm_builder=gemma4_rms_norm_builder,
        scaleless_norm_builder=gemma4_rms_norm_scaleless_builder,
    )
    return _build_block_spec(layer_spec, gemma4_rms_norm_builder, config.num_layers)


def get_gemma4_layer_with_transformer_engine_spec(config: Gemma4TransformerConfig) -> ModuleSpec:
    """Transformer-Engine Gemma4 block spec (V4-close).

    TE linears + ``TENorm`` for the layer norms (q/k norms also TENorm), TE attention
    core. The scaleless v_norm stays ``Gemma4RMSNorm`` (TE has no weightless RMSNorm).
    """
    assert HAVE_TE, "get_gemma4_layer_with_transformer_engine_spec requires Transformer Engine."
    layer_spec = _gemma4_layer_spec(
        self_attn_cls=Gemma4SelfAttention,
        core_attention=TEDotProductAttention,
        linear_qkv=TEColumnParallelLinear,
        linear_proj=TERowParallelLinear,
        linear_fc1=TEColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
        norm_builder=TENorm,
        scaleless_norm_builder=gemma4_rms_norm_scaleless_builder,
    )
    return _build_block_spec(layer_spec, TENorm, config.num_layers)
