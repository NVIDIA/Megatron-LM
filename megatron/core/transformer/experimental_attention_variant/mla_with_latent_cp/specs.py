# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Non-mutating model-spec integration for MLA latent CP."""

from __future__ import annotations

from dataclasses import replace

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from .mla_with_latent_cp import (
    HAVE_TE,
    MLAWithLatentCP,
    _build_local_latent_norm,
    _validate_supported_submodules,
)
from .utils import LatentCPError, _require


def make_mla_with_latent_cp_spec(base_mla_spec: ModuleSpec) -> ModuleSpec:
    """Replace only core attention while preserving a supported old-path projection stack."""

    from megatron.core.transformer.dot_product_attention import DotProductAttention

    _require(
        isinstance(base_mla_spec, ModuleSpec), "base_mla_spec must be a ModuleSpec"
    )
    _require(
        base_mla_spec.module is MLASelfAttention,
        "base_mla_spec must be an ordinary MLA self-attention spec",
    )
    _require(
        isinstance(base_mla_spec.submodules, MLASelfAttentionSubmodules),
        "base_mla_spec has incompatible submodules",
    )
    original = base_mla_spec.submodules
    _require(
        original.linear_qkv_down_proj is None,
        "fused MLA down projection is unsupported",
    )
    _require(
        base_mla_spec.params.get("attn_mask_type") is AttnMaskType.causal,
        "base MLA spec must be causal",
    )

    local_norms = (
        original.q_layernorm is WrappedTorchNorm
        and original.kv_layernorm is WrappedTorchNorm
    )
    if local_norms:
        _require(
            original.core_attention is DotProductAttention,
            "local MLA spec must use DotProductAttention",
        )
        latent_submodules = replace(
            original,
            q_layernorm=_build_local_latent_norm,
            kv_layernorm=_build_local_latent_norm,
            core_attention=IdentityOp,
        )
    else:
        _require(
            HAVE_TE
            and original.q_layernorm is IdentityOp
            and original.kv_layernorm is IdentityOp,
            "TE MLA spec must fuse Q/KV norms into its up projections",
        )
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        _require(
            original.core_attention is TEDotProductAttention,
            "TE MLA spec must use TEDotProductAttention",
        )
        latent_submodules = replace(original, core_attention=IdentityOp)
    _validate_supported_submodules(latent_submodules)
    return replace(
        base_mla_spec,
        module=MLAWithLatentCP,
        params=dict(base_mla_spec.params),
        submodules=latent_submodules,
        metainfo=dict(base_mla_spec.metainfo),
    )


def get_mla_with_latent_cp_spec() -> ModuleSpec:
    """Build the feature-owned local MLA attention spec used by model integration."""

    return ModuleSpec(
        module=MLAWithLatentCP,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=ColumnParallelLinear,
            linear_q_down_proj=ColumnParallelLinear,
            linear_q_up_proj=ColumnParallelLinear,
            linear_kv_down_proj=ColumnParallelLinear,
            linear_kv_up_proj=ColumnParallelLinear,
            core_attention=IdentityOp,
            linear_gate=ColumnParallelLinear,
            linear_proj=RowParallelLinear,
            q_layernorm=_build_local_latent_norm,
            kv_layernorm=_build_local_latent_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )


def _replace_transformer_layer_attention(
    layer_spec: ModuleSpec,
) -> tuple[ModuleSpec, bool]:
    """Replace one ordinary MLA attention slot without mutating its layer spec."""

    _require(isinstance(layer_spec, ModuleSpec), "decoder layers must use ModuleSpec")
    layer_submodules = layer_spec.submodules
    _require(layer_submodules is not None, "decoder layer spec must define submodules")
    attention_spec = getattr(layer_submodules, "self_attention", None)
    if (
        not isinstance(attention_spec, ModuleSpec)
        or attention_spec.module is not MLASelfAttention
    ):
        return layer_spec, False
    return (
        replace(
            layer_spec,
            params=dict(layer_spec.params),
            metainfo=dict(layer_spec.metainfo),
            submodules=replace(
                layer_submodules,
                self_attention=make_mla_with_latent_cp_spec(attention_spec),
                sharded_state_dict_keys_map=dict(
                    layer_submodules.sharded_state_dict_keys_map
                ),
            ),
        ),
        True,
    )


def configure_mla_latent_cp_decoder(
    decoder_spec: ModuleSpec | TransformerBlockSubmodules,
) -> ModuleSpec | TransformerBlockSubmodules:
    """Return a non-mutating GPT decoder spec with latent CP attention."""

    replaced = 0
    if isinstance(decoder_spec, ModuleSpec):
        configured_spec, changed = _replace_transformer_layer_attention(decoder_spec)
        replaced = int(changed)
    elif isinstance(decoder_spec, TransformerBlockSubmodules):
        _require(
            decoder_spec.layer_specs is not None,
            "decoder block must define layer specs",
        )
        configured_layers = []
        for layer_spec in decoder_spec.layer_specs:
            configured_layer, changed = _replace_transformer_layer_attention(layer_spec)
            configured_layers.append(configured_layer)
            replaced += int(changed)
        configured_spec = replace(decoder_spec, layer_specs=configured_layers)
    else:
        raise LatentCPError(
            "latent CP requires a ModuleSpec or TransformerBlockSubmodules decoder spec"
        )
    _require(replaced > 0, "decoder spec contains no ordinary MLA attention slot")
    return configured_spec


def configure_mla_latent_cp_hybrid_stack(stack_spec: ModuleSpec) -> ModuleSpec:
    """Return a non-mutating Hybrid stack with latent CP in its ordinary attention slot."""

    _require(isinstance(stack_spec, ModuleSpec), "hybrid stack must use ModuleSpec")
    stack_submodules = stack_spec.submodules
    _require(stack_submodules is not None, "hybrid stack spec must define submodules")
    mla_layer = getattr(stack_submodules, "mla_layer", None)
    _require(
        isinstance(mla_layer, ModuleSpec),
        "hybrid stack must provide its ordinary MLA layer template",
    )
    latent_layer, replaced = _replace_transformer_layer_attention(mla_layer)
    _require(replaced, "hybrid MLA layer template has no ordinary MLA attention slot")
    return replace(
        stack_spec,
        params=dict(stack_spec.params),
        metainfo=dict(stack_spec.metainfo),
        submodules=replace(stack_submodules, attention_layer=latent_layer),
    )
