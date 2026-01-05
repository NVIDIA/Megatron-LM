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
    ) -> None:
        super().__init__()

        # Input validation
        for name, val in [
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
            ("num_attention_heads", num_attention_heads),
            ("mlp_hidden_size", mlp_hidden_size),
            ("conditioning_dim", conditioning_dim),
        ]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

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
        # Input validation and device checks
        if x.dim() != 3:
            raise ValueError(f"x must be 3D [batch, seq_len, hidden_size], got {x.shape}")
        if x.size(2) != self.hidden_size:
            raise ValueError(
                f"x.size(2)={x.size(2)} does not match model "
                f"hidden_size={self.hidden_size}"
            )
        batch = x.size(0)
        if timesteps.size(0) != batch:
            raise ValueError(f"timesteps batch {timesteps.size(0)} != x batch {batch}")
        if conditioning.size(0) != batch:
            raise ValueError(f"conditioning batch {conditioning.size(0)} != x batch {batch}")
        # Allow timesteps to be [batch] or [batch, 1]
        if timesteps.dim() not in (1, 2):
            raise ValueError(f"timesteps must be [batch] or [batch, 1], got {timesteps.shape}")
        # Device checks
        device = x.device
        if timesteps.device != device or conditioning.device != device:
            raise ValueError("x, timesteps, and conditioning must be on the same device")
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
