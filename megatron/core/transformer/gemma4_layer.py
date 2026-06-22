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

        # Sandwich: post-norm the sublayer output BEFORE its residual add. The base
        # layer adds the raw sublayer output to the residual inside *_bda; we wrap the
        # bda factories so they normalize ``x_with_bias[0]`` first (HF applies the norm
        # before the add, modeling_gemma4.py:1421-1422,1442-1443). Bias is None for
        # gemma4 (no attention/MLP bias), so wrapping the [0] element is sufficient.
        self.self_attn_bda = _wrap_bda_with_post_norm(self.self_attn_bda, self.post_self_attn_layernorm)
        self.mlp_bda = _wrap_bda_with_post_norm(self.mlp_bda, self.post_mlp_layernorm)

        ple_dim = self.config.hidden_size_per_layer_input
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
            input_is_parallel=False,
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
        # Attention sub-block (input_ln -> attn -> post_self_attn_layernorm -> +residual).
        residual = hidden_states
        attn_input = apply_module(self.input_layernorm)(hidden_states)
        attention_output_with_bias = self.self_attention(
            attn_input,
            attention_mask=attention_mask,
            rotary_cos_sin=rotary_cos_sin,
            kv_bus=kv_bus,
        )
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # MLP sub-block (pre_mlp_ln -> mlp -> post_mlp_layernorm -> +residual).
        hidden_states = self._forward_mlp(hidden_states)

        # PLE injection sub-block + layer_scalar.
        hidden_states = self._forward_per_layer_input(hidden_states, per_layer_input)
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None

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


def _wrap_bda_with_post_norm(bda_factory, post_norm):
    """Wrap a bias-dropout-add factory so the sublayer output is post-normed first.

    The base layer calls ``bda_factory(training, fused)(x_with_bias, residual, prob)``
    and adds ``x_with_bias[0]`` to ``residual``. Gemma4's sandwich norm must normalize
    that output before the add, so we replace it with
    ``(post_norm(x_with_bias[0]), x_with_bias[1])`` and defer to the original bda.
    """

    def factory(training, fused):
        inner = bda_factory(training, fused)

        def bda(x_with_bias, residual, prob):
            x, bias = x_with_bias
            return inner((apply_module(post_norm)(x), bias), residual, prob)

        return bda

    return factory
