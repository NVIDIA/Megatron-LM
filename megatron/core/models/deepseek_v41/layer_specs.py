# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer specs and pattern helpers for DeepSeek-V4.1 on HybridModel.

The model is expressed as a hybrid pattern in which every DeepSeek layer is an attention
symbol followed by the MoE symbol ``E``: ``W`` for window-only layers (compress ratio 0)
and ``D`` for layers that attend to compressed positions (ratio read from
``csa_compress_ratios``). Pipeline boundaries (``|``) may only sit between model layers;
for the released 40-layer model the natural split is after layer 20 (the causal-encoder /
decoder boundary), which keeps every CSA2 source together with its consumers.
"""

import dataclasses
from typing import Dict, Iterable, List, Sequence

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.deepseek_v41.hybrid_stack import DSv41HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa2.attention import (
    CSA2Attention,
    CSA2AttentionSubmodules,
    DSv41SelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.csa2.compressor import (
    CSA2Compressor,
    CSA2CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2.indexer import (
    CSA2Indexer,
    CSA2IndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2.roles import (
    pattern_ratios_from_model_layer_ratios,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def build_dsv41_hybrid_layer_pattern(
    model_layer_ratios: Sequence[int], pipeline_split_after_layers: Iterable[int] = ()
) -> str:
    """Hybrid pattern string for the given per-model-layer compress ratios.

    Args:
        model_layer_ratios: ``compress_ratios`` of the text config, one per model layer.
        pipeline_split_after_layers: model layer ids after which a pipeline boundary is
            inserted (e.g. ``[19]`` puts layers 0-19 on stage 0 and 20-39 on stage 1).

    Returns:
        e.g. ``"WEWEDEDE...|DEDE..."``.
    """
    splits = set(int(x) for x in pipeline_split_after_layers)
    num_layers = len(model_layer_ratios)
    for split in splits:
        if not 0 <= split < num_layers - 1:
            raise ValueError(
                f"pipeline split after layer {split} is not inside [0, {num_layers - 1})"
            )
    parts: List[str] = []
    for layer_id, ratio in enumerate(model_layer_ratios):
        symbol = Symbols.WINDOW if int(ratio) == 0 else Symbols.DS_ATTENTION
        parts.append(symbol + Symbols.MOE)
        if layer_id in splits:
            parts.append(Symbols.PIPE)
    return "".join(parts)


def dsv41_config_kwargs_from_model_layers(model_layer_ratios: Sequence[int]) -> Dict:
    """``TransformerConfig`` fields derived from the per-model-layer ratios.

    Returns ``num_layers`` (pattern symbols) and the pattern-position ``csa_compress_ratios``.
    """
    pattern_ratios = pattern_ratios_from_model_layer_ratios(model_layer_ratios)
    return {"num_layers": len(pattern_ratios), "csa_compress_ratios": pattern_ratios}


def get_dsv41_attention_spec(config, backend=None) -> ModuleSpec:
    """Self-attention spec: V4 projections around the CSA2 core attention."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        _get_backend_spec_provider,
    )

    if backend is None:
        backend = _get_backend_spec_provider(config)
    assert config.multi_latent_attention, "DeepSeek-V4.1 attention builds on MLA projections"

    rms_norm = config.normalization == "RMSNorm"
    qk_norm = (
        backend.layer_norm(rms_norm=rms_norm, for_qk=True) if config.qk_layernorm else IdentityOp
    )

    compressor_spec = ModuleSpec(
        module=CSA2Compressor,
        submodules=CSA2CompressorSubmodules(norm=backend.layer_norm(rms_norm=True, for_qk=False)),
    )
    indexer_spec = ModuleSpec(
        module=CSA2Indexer,
        submodules=CSA2IndexerSubmodules(
            linear_wq_b=backend.linear(),
            linear_weights_proj=backend.linear(),
            linear_wk=backend.linear(),
            k_norm=backend.layer_norm(rms_norm=True, for_qk=True),
        ),
    )
    core_attention = ModuleSpec(
        module=CSA2Attention,
        submodules=CSA2AttentionSubmodules(compressor=compressor_spec, indexer=indexer_spec),
    )
    return ModuleSpec(
        module=DSv41SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSv4HybridSelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_proj=backend.column_parallel_linear(),
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )


def hybrid_dsv41_stack_spec(config) -> ModuleSpec:
    """Config-aware stack spec for DeepSeek-V4.1.

    Selected via ``--spec megatron.core.models.deepseek_v41.layer_specs hybrid_dsv41_stack_spec``.
    Both attention symbols (``W`` and ``D``) map to the same layer spec; the window-only
    behaviour follows from the ratio 0 at that pattern position.
    """
    if getattr(config, "dsv4_version", "v4") != "v4.1":
        raise ValueError("hybrid_dsv41_stack_spec requires dsv4_version='v4.1'")

    attention = get_dsv41_attention_spec(config)
    attention_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm, self_attention=attention, self_attn_bda=get_bias_dropout_add
        ),
    )
    submodules = dataclasses.replace(
        hybrid_stack_spec.submodules,
        dsa_layer=attention_layer,
        window_layer=attention_layer,
        mtp_block_spec=None,
    )
    return ModuleSpec(module=DSv41HybridStack, submodules=submodules)
