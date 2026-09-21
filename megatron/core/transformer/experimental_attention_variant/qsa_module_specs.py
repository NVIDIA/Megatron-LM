# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider, get_backend_from_config
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSAIndexer,
    QSAIndexerSubmodules,
    QSASelfAttention,
    QSASelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def get_qsa_module_spec_for_backend(
    config: TransformerConfig, backend: Optional[BackendSpecProvider] = None
) -> ModuleSpec:
    """Helper function to get module spec for QSA (Qwen Sparse Attention).

    QSA is GQA-based: the main attention is a stock ``SelfAttention`` (output gate,
    partial RoPE, qk layernorm come from existing config knobs); only the indexer that
    produces the block-sparse selection mask is new.
    """
    if backend is None:
        backend = get_backend_from_config(config)
    assert not config.multi_latent_attention, "QSA is GQA-based, not MLA."

    rms_norm = config.normalization == "RMSNorm"
    qk_norm = (
        backend.layer_norm(rms_norm=rms_norm, for_qk=True) if config.qk_layernorm else IdentityOp
    )

    indexer = ModuleSpec(
        module=QSAIndexer,
        submodules=QSAIndexerSubmodules(
            index_qk_proj=backend.linear(),
            q_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=True),
            k_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=True),
        ),
    )

    # The indexer consumes the same post-input-layernorm hidden states as the main
    # q/k/v projections (HF semantics), so the input layernorm must stay unfused.
    attention = ModuleSpec(
        module=QSASelfAttention,
        params={"attn_mask_type": AttnMaskType.arbitrary},
        submodules=QSASelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            k_layernorm=qk_norm,
            indexer=indexer,
        ),
        metainfo={"fuse_input_layernorm": False},
    )
    return attention
