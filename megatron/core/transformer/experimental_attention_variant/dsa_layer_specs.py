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

from dataclasses import replace

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


def _get_dsa_gqa_indexer_spec() -> ModuleSpec:
    """Build an independent indexer spec for each derived stack."""
    return ModuleSpec(
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
        A spec whose ``attention_layer`` builds :class:`DSGroupedSelfAttention`
        over a :class:`DSGQACoreAttention`. Unchanged layers and builders are shared.
    """
    # Copy only the dataclasses on the modified path. Optional-dependency builders
    # can be mocks containing closure cells, which cannot be deep-copied.
    attention_layer = stack_spec.submodules.attention_layer
    self_attention = attention_layer.submodules.self_attention
    # The stack's own dense attention becomes DSA's warmup core (dsa_fwd_use_dense_attn),
    # captured rather than named, so upstream changes to it are inherited.
    core_attention = ModuleSpec(
        module=DSGQACoreAttention,
        submodules=DSGQAAttentionSubmodules(
            indexer=_get_dsa_gqa_indexer_spec(),
            dense_core_attention=self_attention.submodules.core_attention,
        ),
    )
    self_attention = replace(
        self_attention,
        module=DSGroupedSelfAttention,
        submodules=replace(self_attention.submodules, core_attention=core_attention),
    )
    attention_layer = replace(
        attention_layer,
        submodules=replace(attention_layer.submodules, self_attention=self_attention),
    )
    return replace(
        stack_spec, submodules=replace(stack_spec.submodules, attention_layer=attention_layer)
    )


dsa_stack_spec = _with_dsa_gqa_attention(hybrid_stack_spec)
gdp_dsa_stack_spec = _with_dsa_gqa_attention(gated_delta_product_stack_spec)
