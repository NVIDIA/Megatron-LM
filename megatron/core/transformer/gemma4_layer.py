# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from megatron.core.transformer.gemma4_norm import Gemma4RMSNorm
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


@dataclass
class Gemma4TransformerLayerSubmodules(TransformerLayerSubmodules):
    """:class:`TransformerLayerSubmodules` plus Gemma4 sandwich + PLE submodules."""

    # Sandwich norms applied to a sublayer output BEFORE its residual add.
    post_self_attn_layernorm: LayerNormBuilder = None
    post_mlp_layernorm: LayerNormBuilder = None

    # Per-Layer-Embedding (PLE) injection sub-block.
    per_layer_input_gate: type = None
    per_layer_projection: type = None
    post_per_layer_input_norm: LayerNormBuilder = None


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma 4 decoder layer (HF ``Gemma4TextDecoderLayer``, modeling_gemma4.py:1409-1454).

    Adds three things to the base pre-norm :class:`TransformerLayer`:

    * **Sandwich norms** — ``post_self_attn_layernorm`` applied to the attention output
      and ``post_mlp_layernorm`` to the MLP output, each BEFORE its residual add
      (``h = residual + post_norm(sublayer(pre_norm(h)))``).
    * **PLE sub-block** — a third residual sub-block injecting the per-layer embedding:
      ``h += post_per_layer_input_norm(per_layer_projection(gelu_tanh(
      per_layer_input_gate(h)) * p_i))``.
    * **layer_scalar** — a ``ones(1)`` buffer multiplied into the layer output last.

    The per-layer input ``p_i``, the per-layer-type rotary ``cos/sin``, and the
    cross-layer ``kv_bus`` are threaded in as forward kwargs by
    :class:`~megatron.core.models.gemma4.gemma4_block.Gemma4TransformerBlock`.
    """

    def __init__(self, config: TransformerConfig, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)

        self.post_self_attn_layernorm = submodules.post_self_attn_layernorm(
            config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
        )
        self.post_mlp_layernorm = submodules.post_mlp_layernorm(
            config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
        )

        ple_dim = self.config.hidden_size_per_layer_input
        # The PLE sub-block is a standard Column->Row pair, so it inherits the SP
        # all-gather/reduce-scatter dance and is correct at TP=1, TP>1 (no SP), and TP>1+SP:
        #   gate = per_layer_input_gate (Column, gather_output=False) -> [s, b, ple_dim/TP]
        #          (under SP the Column linear all-gathers the sequence internally, so the
        #           gate is full-sequence with the ple_dim partitioned across TP);
        #   the threaded ``per_layer_input`` is full-sequence and ple_dim-sharded
        #          ([s, b, ple_dim/TP]) -- Gemma4Model.forward scatters per_layer_inputs on
        #          the ple_dim with scatter_to_tensor_model_parallel_region -- so
        #          ``gate * per_layer_input`` is an elementwise [s, b, ple_dim/TP] product;
        #   per_layer_projection (Row, input_is_parallel=True) consumes the partitioned
        #          input and reduce-scatters (SP) / all-reduces (no SP) back to [s(/TP), b, h].
        # All of this is a no-op at TP=1 (the scatter is identity) and does not affect weight
        # sharding / ckpt load.
        self.per_layer_input_gate = submodules.per_layer_input_gate(
            self.config.hidden_size,
            ple_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="per_layer_input_gate",
        )
        self.per_layer_projection = submodules.per_layer_projection(
            ple_dim,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="per_layer_projection",
        )
        self.post_per_layer_input_norm = submodules.post_per_layer_input_norm(
            config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        per_layer_input: Optional[Tensor] = None,
        rotary_cos_sin: Optional[tuple] = None,
        kv_bus: Optional[dict] = None,
        **kwargs,
    ):
        """Decoder-layer forward. ``hidden_states`` is [s, b, h] (MLM seq-first).

        Mirrors the base attention/MLP residual structure but threads the gemma4
        side-inputs (per-layer-type rope cos/sin and the cross-layer kv_bus) into the
        attention, then appends the PLE sub-block and the ``layer_scalar`` multiply.
        """
        # HF gemma4 uses explicit sandwich residuals: ``h = residual + post_norm(
        # sublayer(pre_norm(h)))`` (modeling_gemma4.py:1409-1443). We add the residual
        # explicitly (rather than via bias_dropout_add) because the gemma4 sublayers have
        # no bias, dropout is 0, and the in-place bda path aliases the post-norm tensor.
        # Attention sub-block.
        residual = hidden_states
        attn_input = apply_module(self.input_layernorm)(hidden_states)
        attn_output, _ = self.self_attention(
            attn_input,
            attention_mask=attention_mask,
            rotary_cos_sin=rotary_cos_sin,
            kv_bus=kv_bus,
        )
        attn_output = apply_module(self.post_self_attn_layernorm)(attn_output)
        hidden_states = residual + attn_output

        # MLP sub-block.
        residual = hidden_states
        mlp_input = apply_module(self.pre_mlp_layernorm)(hidden_states)
        mlp_output, _ = self.mlp(mlp_input)
        mlp_output = apply_module(self.post_mlp_layernorm)(mlp_output)
        hidden_states = residual + mlp_output

        # PLE injection sub-block + layer_scalar.
        hidden_states = self._forward_per_layer_input(hidden_states, per_layer_input)
        hidden_states = hidden_states * self.layer_scalar
        return make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        ), None

    def _forward_per_layer_input(self, hidden_states: Tensor, per_layer_input: Tensor) -> Tensor:
        """PLE injection sub-block (HF modeling_gemma4.py:1445-1452)."""
        residual = hidden_states
        gate, _ = apply_module(self.per_layer_input_gate)(hidden_states)
        gate = torch.nn.functional.gelu(gate, approximate="tanh")
        hidden_states = gate * per_layer_input
        hidden_states, _ = apply_module(self.per_layer_projection)(hidden_states)
        hidden_states = apply_module(self.post_per_layer_input_norm)(hidden_states)
        hidden_states = residual + hidden_states
        return make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
