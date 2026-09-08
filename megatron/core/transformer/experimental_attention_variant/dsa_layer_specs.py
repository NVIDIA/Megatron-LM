# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

"""Layer specs that wire DSA-over-GQA into the hybrid (Mamba) stack.

Derived from ``hybrid_stack_spec`` rather than copied from it, so that upstream
changes to the Mamba mixer, MLP, MoE, or MTP submodules are picked up
automatically. The only DSA-specific change is the attention layer's
``self_attention`` module: ``DSGroupedSelfAttention`` replaces ``SelfAttention``.

Note that this is distinct from the ``dsa_layer`` already present in
``hybrid_stack_spec``, which wires DeepSeek's DSA over *MLA*
(``AbsorbedMLASelfAttention``). This spec applies DSA over *GQA*, reusing the
stack's existing attention layer slot.

Usage::

    --spec megatron.core.transformer.experimental_attention_variant.dsa_layer_specs \
        dsa_stack_spec
"""

import copy

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import DSGroupedSelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec


def _with_dsa_gqa_attention(stack_spec: ModuleSpec) -> ModuleSpec:
    """Return a copy of ``stack_spec`` whose attention layer uses DSA over GQA.

    Args:
        stack_spec: A Mamba/hybrid stack spec to derive from. Not modified.

    Returns:
        A deep copy with ``attention_layer``'s self-attention module replaced by
        :class:`DSGroupedSelfAttention`.
    """
    spec = copy.deepcopy(stack_spec)
    spec.submodules.attention_layer.submodules.self_attention.module = DSGroupedSelfAttention
    return spec


dsa_stack_spec = _with_dsa_gqa_attention(hybrid_stack_spec)
