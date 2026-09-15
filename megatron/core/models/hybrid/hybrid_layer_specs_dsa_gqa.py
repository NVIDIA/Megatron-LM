# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Hybrid stack spec with a GQA-based DSA layer.

Main's shipped `dsa_layer` pairs the DSA core with AbsorbedMLASelfAttention (MLA).
This module provides an alternative `dsa_layer` that pairs the DSA core with a
plain GQA SelfAttention, so the same cuDNN/TileLang DSA kernels run over GQA Q/K/V
instead of MLA. The DSAttention + DSAIndexer core is byte-identical to main's; only
the attention module that produces Q/K/V changes.

Reference it via::

    --spec megatron.core.models.hybrid.hybrid_layer_specs_dsa_gqa \
        hybrid_stack_spec_dsa_gqa
and place DSA layers with `D` in --hybrid-layer-pattern.
"""

import copy

from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec as _base_stack_spec
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttentionSubmodules
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGQAIndexer,
    DSGQAIndexerSubmodules,
    DSGQASelfAttention,
    DSGQAttention,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

# GQA DSA layer: DSGQASelfAttention (GQA QKV + DSA-input injection) with a plain
# DSAttention core whose indexer is the hidden-based DSGQAIndexer. Because value is
# a real GQA value tensor, DSAttention.forward takes its non-absorbed path: cuDNN
# indexer top-k + PyTorch reference loss + PyTorch unfused_dsa_fn output.
dsa_gqa_layer = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=DSGQASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,  # folds input-LN + GQA QKV
                core_attention=ModuleSpec(
                    module=DSGQAttention,
                    submodules=DSAttentionSubmodules(
                        indexer=ModuleSpec(
                            module=DSGQAIndexer,
                            submodules=DSGQAIndexerSubmodules(
                                linear_q=TELinear,
                                linear_wk=TELinear,
                                k_norm=TENorm,
                                linear_weights_proj=TELinear,
                            ),
                        )
                    ),
                ),
                linear_proj=TERowParallelLinear,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
    ),
)

# Full stack spec = base hybrid stack with the dsa_layer swapped to the GQA variant.
# copy.deepcopy is safe here: class-type fields are copied by reference.
hybrid_stack_spec_dsa_gqa = copy.deepcopy(_base_stack_spec)
hybrid_stack_spec_dsa_gqa.submodules.dsa_layer = dsa_gqa_layer
