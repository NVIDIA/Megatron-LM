# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 projections composed with the existing DeepSeek attention wrapper."""

from functools import partial

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
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

csa2_attention_spec = ModuleSpec(
    module=DSv4HybridSelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=DSv4HybridSelfAttentionSubmodules(
        linear_q_down_proj=TELinear,
        linear_q_up_proj=TEColumnParallelLinear,
        linear_kv_proj=TEColumnParallelLinear,
        linear_proj=TERowParallelLinear,
        q_layernorm=TENorm,
        kv_layernorm=TENorm,
        core_attention=partial(
            CompressedSparseAttention2,
            submodules=CompressedSparseAttentionSubmodules(
                compressor=partial(
                    CSA2Compressor,
                    submodules=CompressorSubmodules(
                        linear_wkv=TELinear, linear_wgate=TELinear, norm=TENorm
                    ),
                ),
                indexer=partial(
                    CSA2Indexer,
                    submodules=CSA2IndexerSubmodules(
                        linear_wq_b=TELinear,
                        linear_wk=TELinear,
                        k_norm=TENorm,
                        linear_weights_proj=TELinear,
                    ),
                ),
            ),
        ),
    ),
)
