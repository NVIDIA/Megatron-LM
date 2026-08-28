# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from functools import partial

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionBuilder,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorBuilder,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerBuilder,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def get_dsv4_hybrid_module_spec_for_backend(
    config: TransformerConfig, backend: BackendSpecProvider
) -> ModuleSpec:
    """Build the native SBHD DSv4 HybridModel attention spec."""
    assert config.multi_latent_attention, "Currently only MLA supports sparse attention."
    assert config.qk_l2_norm is False, "qk_l2_norm is not supported with MLA."

    rms_norm = config.normalization == "RMSNorm"
    qk_norm = (
        backend.layer_norm(rms_norm=rms_norm, for_qk=True) if config.qk_layernorm else IdentityOp
    )

    compressor_builder: CompressorBuilder = partial(
        Compressor,
        submodules=CompressorSubmodules(
            linear_wkv=backend.linear(),
            linear_wgate=backend.linear(),
            norm=backend.layer_norm(rms_norm=True, for_qk=False),
        ),
    )
    indexer_builder: CSAIndexerBuilder = partial(
        CSAIndexer,
        submodules=CSAIndexerSubmodules(
            linear_wq_b=backend.linear(),
            linear_weights_proj=backend.linear(),
            compressor=compressor_builder,
        ),
    )
    core_attention_builder: CompressedSparseAttentionBuilder = partial(
        CompressedSparseAttention,
        submodules=CompressedSparseAttentionSubmodules(
            compressor=compressor_builder, indexer=indexer_builder
        ),
    )

    return ModuleSpec(
        module=DSv4HybridSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSv4HybridSelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_proj=backend.column_parallel_linear(),
            core_attention=core_attention_builder,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )
