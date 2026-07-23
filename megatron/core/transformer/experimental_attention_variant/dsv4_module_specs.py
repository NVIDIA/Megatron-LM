# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Backend-neutral module specs for DeepSeek-V4 compressed sparse attention."""

from typing import Protocol

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class DSv4BackendSpecProvider(Protocol):
    """Minimal backend interface needed to build the DSv4 attention spec."""

    def linear(self) -> type:
        """Return the backend's non-parallel linear module."""
        ...

    def column_parallel_linear(self) -> type:
        """Return the backend's column-parallel linear module."""
        ...

    def row_parallel_linear(self) -> type:
        """Return the backend's row-parallel linear module."""
        ...

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> type:
        """Return the backend's normalization module."""
        ...


def get_dsv4_hybrid_module_spec_for_backend(
    config: TransformerConfig, backend: DSv4BackendSpecProvider
) -> ModuleSpec:
    """Build a DSv4 compressed sparse-attention spec for an explicit backend."""
    assert config.multi_latent_attention, "Currently only MLA supports sparse attention."
    assert config.qk_l2_norm is False, "qk_l2_norm is not supported with MLA."

    rms_norm = config.normalization == "RMSNorm"
    qk_norm = (
        backend.layer_norm(rms_norm=rms_norm, for_qk=True) if config.qk_layernorm else IdentityOp
    )

    compressor_spec = ModuleSpec(
        module=Compressor,
        submodules=CompressorSubmodules(
            linear_wkv=backend.linear(),
            linear_wgate=backend.linear(),
            norm=backend.layer_norm(rms_norm=True, for_qk=False),
        ),
    )
    indexer_spec = ModuleSpec(
        module=CSAIndexer,
        submodules=CSAIndexerSubmodules(
            linear_wq_b=backend.linear(),
            linear_weights_proj=backend.linear(),
            compressor=compressor_spec,
        ),
    )
    core_attention = ModuleSpec(
        module=CompressedSparseAttention,
        submodules=CompressedSparseAttentionSubmodules(
            compressor=compressor_spec, indexer=indexer_spec
        ),
    )

    return ModuleSpec(
        module=DSv4HybridSelfAttention,
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
