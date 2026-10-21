# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

"""Layer specs that wire DSA-over-GQA into the hybrid (Mamba) stack.

Derived from ``hybrid_stack_spec`` rather than copied from it, so that upstream
changes to the Mamba mixer, MLP, MoE, or MTP submodules are picked up
automatically. The DSA-specific changes are confined to the attention layer:
``DSGroupedSelfAttention`` replaces ``SelfAttention``, and ``DSGQACoreAttention``
(wrapping the indexer and the stack's own dense attention) replaces the plain
core attention.

Note that this is distinct from the ``dsa_layer`` already present in
``hybrid_stack_spec``, which wires DeepSeek's DSA over *MLA*
(``AbsorbedMLASelfAttention``). This spec applies DSA over *GQA*, reusing the
stack's existing attention layer slot.

Two stacks are provided, differing only in their Mamba mixer: ``dsa_stack_spec`` derives
from ``hybrid_stack_spec``, ``gdp_dsa_stack_spec`` from ``gated_delta_product_stack_spec``.
Both share the same attention layer, so DSA over GQA applies to either unchanged.

Usage::

    --spec megatron.core.transformer.experimental_attention_variant.dsa_layer_specs \
        dsa_stack_spec
"""

import copy

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGQAAttentionSubmodules,
    DSGQACoreAttention,
    DSGroupedSelfAttention,
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

dsa_gqa_indexer_spec = ModuleSpec(
    module=SimplifiedDSGQAIndexer,
    submodules=SimplifiedDSGQAIndexerSubmodules(
        linear_q=ModuleSpec(module=TELinear), linear_k=ModuleSpec(module=TELinear)
    ),
)


def _with_dsa_gqa_attention(stack_spec: ModuleSpec) -> ModuleSpec:
    """Return a copy of ``stack_spec`` whose attention layer uses DSA over GQA.

    Args:
        stack_spec: A Mamba/hybrid stack spec to derive from. Not modified.

    Returns:
        A deep copy whose ``attention_layer`` builds :class:`DSGroupedSelfAttention`
        over a :class:`DSGQACoreAttention`.
    """
    spec = copy.deepcopy(stack_spec)
    self_attention = spec.submodules.attention_layer.submodules.self_attention
    self_attention.module = DSGroupedSelfAttention
    # The stack's own dense attention becomes DSA's warmup core (dsa_fwd_use_dense_attn),
    # captured rather than named, so upstream changes to it are inherited.
    self_attention.submodules.core_attention = ModuleSpec(
        module=DSGQACoreAttention,
        submodules=DSGQAAttentionSubmodules(
            # Copied, not shared: the two stacks must not alias one indexer spec, for the
            # same reason this function deep copies the stack.
            indexer=copy.deepcopy(dsa_gqa_indexer_spec),
            dense_core_attention=self_attention.submodules.core_attention,
        ),
    )
    return spec


dsa_stack_spec = _with_dsa_gqa_attention(hybrid_stack_spec)
gdp_dsa_stack_spec = _with_dsa_gqa_attention(gated_delta_product_stack_spec)
