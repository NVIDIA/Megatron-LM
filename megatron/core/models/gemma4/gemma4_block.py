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

    Recompute=full is out of scope as of now. Inference is supported via the optional
    ``inference_context`` kwarg (:class:`Gemma4InferenceContext`), which replaces
    the per-forward ``kv_bus`` with a persistent per-layer KV cache.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        per_layer_inputs: Tensor = None,
        rotary_cos_sin_by_type: dict = None,
        attention_mask_by_type: dict = None,
        inference_context=None,
        **kwargs,
    ):
        """Run the gemma4 decoder layers. ``hidden_states`` is [s, b, h]."""
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        # Training path uses a fresh per-forward KV bus (cross-layer K/V sharing
        # within one forward). Inference path uses the persistent
        # ``inference_context`` cache instead; kv_bus stays None and is ignored.
        kv_bus: Optional[dict] = None if inference_context is not None else {}

        for i, layer in enumerate(self.layers):
            layer_type = layer.self_attention.layer_type
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask_by_type[layer_type],
                per_layer_input=per_layer_inputs[:, :, i, :],
                rotary_cos_sin=rotary_cos_sin_by_type[layer_type],
                kv_bus=kv_bus,
                inference_context=inference_context,
            )

        if self.final_layernorm is not None:
            hidden_states = apply_module(self.final_layernorm)(hidden_states)

        return hidden_states
