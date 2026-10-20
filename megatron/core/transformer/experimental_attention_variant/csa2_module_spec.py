# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 projections composed with the existing DeepSeek attention wrapper."""

from functools import partial
from typing import Protocol

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


class CSA2BackendSpecProvider(BackendSpecProvider, Protocol):
    """Backend builders required by CSA2's duplicated compression/index projections."""

    def linear(self) -> type:
        """Return the backend's duplicated linear projection."""
        ...


def get_csa2_module_spec_for_backend(backend: CSA2BackendSpecProvider) -> ModuleSpec:
    """Compose CSA2 with the caller's projection and normalization implementations."""
    return ModuleSpec(
        module=DSv4HybridSelfAttention,
        metainfo={"fuse_input_layernorm": False},
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSv4HybridSelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_proj=backend.column_parallel_linear(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
            kv_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
            core_attention=partial(
                CompressedSparseAttention2,
                submodules=CompressedSparseAttentionSubmodules(
                    compressor=partial(
                        CSA2Compressor,
                        submodules=CompressorSubmodules(
                            linear_wkv=backend.linear(),
                            linear_wgate=backend.linear(),
                            norm=backend.layer_norm(rms_norm=True, for_qk=False),
                        ),
                    ),
                    indexer=partial(
                        CSA2Indexer,
                        submodules=CSA2IndexerSubmodules(
                            linear_wq_b=backend.linear(),
                            linear_wk=backend.linear(),
                            k_norm=backend.layer_norm(rms_norm=True, for_qk=True),
                            linear_weights_proj=backend.linear(),
                        ),
                    ),
                ),
            ),
        ),
    )


# A precise Transformer Engine composition for static HybridModel specs.
csa2_attention_spec = get_csa2_module_spec_for_backend(TESpecProvider())


csa2_layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=csa2_attention_spec,
        self_attn_bda=get_bias_dropout_add,
    ),
)
