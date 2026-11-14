# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid-stack specs selecting simplified sparse attention for attention layers.

Use with ``experimental_attention_variant='ssa'``, for example::

    --experimental-attention-variant ssa \\
    --spec megatron.core.transformer.experimental_attention_variant.\\
simplified_sparse_attention_layer_specs ssa_stack_spec
"""

import copy

from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention import (
    SimplifiedSparseAttention,
)
from megatron.core.transformer.spec_utils import ModuleSpec


def _with_sparse_attention(stack_spec: ModuleSpec) -> ModuleSpec:
    spec = copy.deepcopy(stack_spec)
    spec.submodules.attention_layer.submodules.self_attention.module = SimplifiedSparseAttention
    return spec


ssa_stack_spec = _with_sparse_attention(hybrid_stack_spec)
gdp_ssa_stack_spec = _with_sparse_attention(gated_delta_product_stack_spec)
