# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Conditional-memory integration with the standard HybridStack layer loop."""

from dataclasses import dataclass, replace
from functools import partial

from torch import Tensor

from megatron.core.models.deepseek_v41.moe import ModalityMoELayer, ModalityRouter
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_csa2_stack_spec
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


@dataclass
class DeepSeekV41ForwardContext(HybridStackForwardContext):
    """Conditional inputs for one forward, kept out of parameter-bearing modules."""

    engram_hashes: Tensor | None = None
    token_mask: Tensor | None = None
    image_mask: Tensor | None = None


@dataclass
class MultimodalCSA2State(CSA2State):
    """Shared attention state plus read-only modality metadata for MoE layers."""

    image_mask: Tensor | None = None
    moe_layer_ids: frozenset[int] = frozenset()

    def mlp_kwargs(self, layer_number: int) -> dict:
        """Only modality-aware expert layers receive the mask."""
        if self.image_mask is None or layer_number - 1 not in self.moe_layer_ids:
            return {}
        return {"image_mask": self.image_mask}


class DeepSeekV41Adapter(CSA2HybridAdapter):
    """Apply registered memory modules without introducing another stack implementation."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.moe_layer_ids = frozenset(
            i for i, symbol in enumerate(kwargs["layer_type_list"]) if symbol == "E"
        )

    def prepare_forward(self, hidden_states, packed_seq_params, inference_context, context):
        """Carry the caller's conditional inputs alongside fresh CSA2 state."""
        if not isinstance(context, DeepSeekV41ForwardContext):
            context = DeepSeekV41ForwardContext(
                cross_layer_state=context.cross_layer_state, mhc_state=context.mhc_state
            )
        if context.cross_layer_state is None:
            context.cross_layer_state = MultimodalCSA2State(
                attention_layer_ids=self.attention_layer_ids,
                image_mask=context.image_mask,
                moe_layer_ids=self.moe_layer_ids,
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


# Eager modality routing uses the same dispatcher and expert implementations.
_moe_layer = deepseek_v41_stack_spec.submodules.moe_layer
_modality_mlp = partial(
    ModalityMoELayer,
    submodules=replace(_moe_layer.submodules.mlp.keywords["submodules"], router=ModalityRouter),
)
deepseek_v41_multimodal_stack_spec = replace(
    deepseek_v41_stack_spec,
    submodules=replace(
        deepseek_v41_stack_spec.submodules,
        moe_layer=replace(
            _moe_layer,
            module=TransformerLayer,
            submodules=replace(_moe_layer.submodules, mlp=_modality_mlp),
        ),
    ),
)
