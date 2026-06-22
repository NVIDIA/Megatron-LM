# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from torch import Tensor

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


class Gemma4TransformerBlock(TransformerBlock):
    """Gemma 4 decoder stack: a :class:`TransformerBlock` that owns the KV bus.

    Overrides the eager layer loop so that, per forward, it (a) instantiates a fresh
    cross-layer KV bus (a plain dict keyed by layer type), (b) hands each layer its
    ``per_layer_inputs[:, :, i, :]`` slice, and (c) selects the per-layer-type rotary
    ``cos/sin`` and additive attention mask. This keeps all KV-share / PLE / mask
    plumbing out of the base block (R1: no base-block kwarg churn).

    Recompute=full and inference paths are out of scope for this training-only port.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        per_layer_inputs: Tensor = None,
        rotary_cos_sin_by_type: dict = None,
        attention_mask_by_type: dict = None,
        **kwargs,
    ):
        """Run the gemma4 decoder layers. ``hidden_states`` is [s, b, h]."""
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        # Fresh KV bus per forward (plain dict; single device, no CP/TP interaction).
        kv_bus: dict = {}

        for i, layer in enumerate(self.layers):
            layer_type = layer.self_attention.layer_type
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask_by_type[layer_type],
                per_layer_input=per_layer_inputs[:, :, i, :],
                rotary_cos_sin=rotary_cos_sin_by_type[layer_type],
                kv_bus=kv_bus,
            )

        if self.final_layernorm is not None:
            hidden_states = apply_module(self.final_layernorm)(hidden_states)

        return hidden_states
