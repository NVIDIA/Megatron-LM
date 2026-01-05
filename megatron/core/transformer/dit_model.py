# Copyright (c) NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn

from megatron.core.transformer.dit_layer import DiTTransformerLayer

class DiTModel(nn.Module):
    """
    Minimal Diffusion Transformer (DiT) reference model.

    This is a lightweight, forward-only reference implementation intended
    to demonstrate correct wiring of:
      - DiTTransformerLayer
      - adaLN-Zero conditioning
      - timestep conditioning

    This is NOT a training or inference-ready model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        conditioning_dim: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Timestep embedding (simple MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Conditioning projection
        self.cond_proj = nn.Linear(conditioning_dim, hidden_size)

        # Transformer stack
        self.layers = nn.ModuleList(
            [
                DiTTransformerLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_hidden_size=mlp_hidden_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, hidden_size]
            timesteps: Tensor of shape [batch] or [batch, 1]
            conditioning: Tensor of shape [batch, conditioning_dim]
        """

        # Ensure timestep shape
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)

        # Build conditioning embedding
        t_emb = self.time_embed(timesteps)
        c_emb = self.cond_proj(conditioning)
        cond = t_emb + c_emb

        # Transformer forward
        for layer in self.layers:
            x = layer(x, cond)

        x = self.final_norm(x)
        return x
