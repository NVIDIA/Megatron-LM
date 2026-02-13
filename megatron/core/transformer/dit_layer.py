# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from typing import Optional, Union, Tuple

from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.adaptive_layernorm import AdaLayerNormZero


class DiTTransformerLayer(TransformerLayer):
    """
    Diffusion Transformer Layer (DiT).

    Differences vs standard TransformerLayer:
    - Uses AdaLayerNormZero instead of LayerNorm
    - Accepts a conditioning tensor
    - Applies gating AFTER Attention and MLP
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        layer_number: int = 1,
        cond_dim: Optional[int] = None,
    ) -> None:
        if cond_dim is None:
            raise ValueError("cond_dim must be provided for DiTTransformerLayer")

        # Initialize standard Megatron layer (creates attention, mlp, etc.)
        super().__init__(config, submodules, layer_number)

        # Replace LayerNorms with AdaLN-Zero
        self.input_layernorm = AdaLayerNormZero(
            config=config,
            hidden_size=config.hidden_size,
            cond_dim=cond_dim,
            eps=config.layernorm_epsilon,
        )

        self.post_attention_layernorm = AdaLayerNormZero(
            config=config,
            hidden_size=config.hidden_size,
            cond_dim=cond_dim,
            eps=config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conditioning: torch.Tensor,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, S, H]
            attention_mask: standard Megatron attention mask
            conditioning: [B, C] or [B, S, C]

        Returns:
            hidden_states: [B, S, H]
        """

        # === Attention block ===
        normed_states, gate_attn = self.input_layernorm(
            hidden_states, conditioning
        )

        attention_output, _ = self.self_attention(
            normed_states,
            attention_mask,
            **kwargs,
        )

        # Apply DiT gate AFTER attention
        attention_output = gate_attn * attention_output

        # Residual
        hidden_states = hidden_states + attention_output

        # === MLP block ===
        normed_states, gate_mlp = self.post_attention_layernorm(
            hidden_states, conditioning
        )

        mlp_output, _ = self.mlp(normed_states)

        # Apply DiT gate AFTER MLP
        mlp_output = gate_mlp * mlp_output

        # Residual
        hidden_states = hidden_states + mlp_output

        return hidden_states
