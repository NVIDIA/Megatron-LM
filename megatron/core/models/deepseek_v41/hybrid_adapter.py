# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Conditional-memory integration with the standard HybridStack layer loop."""

from dataclasses import dataclass, replace

from torch import Tensor

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_csa2_stack_spec
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)


@dataclass
class DeepSeekV41ForwardContext(HybridStackForwardContext):
    """Conditional inputs for one forward, kept out of parameter-bearing modules."""

    engram_hashes: Tensor | None = None
    token_mask: Tensor | None = None


class DeepSeekV41Adapter(CSA2HybridAdapter):
    """Apply registered memory modules without introducing another stack implementation."""

    def prepare_forward(self, hidden_states, packed_seq_params, inference_context, context):
        """Carry the caller's conditional inputs alongside fresh CSA2 state."""
        if not isinstance(context, DeepSeekV41ForwardContext):
            context = DeepSeekV41ForwardContext(
                cross_layer_state=context.cross_layer_state, mhc_state=context.mhc_state
            )
        return super().prepare_forward(hidden_states, packed_seq_params, inference_context, context)

    def before_layer(self, layer, hidden_states, context):
        """Add conditional memory before the selected attention layer's mHC projection."""
        engram = getattr(layer, "engram", None)
        if engram is not None:
            if context.engram_hashes is None:
                raise ValueError("Engram layers require hash addresses in the forward context")
            hidden_states = engram(hidden_states, context.engram_hashes, context.token_mask)
        return hidden_states


deepseek_v41_stack_spec = replace(
    hybrid_csa2_stack_spec,
    submodules=replace(hybrid_csa2_stack_spec.submodules, forward_adapter=DeepSeekV41Adapter),
)
