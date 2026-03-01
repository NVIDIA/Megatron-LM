# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
from typing import Tuple
from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
from megatron.core.transformer import TransformerConfig

class AdaLayerNormZero(nn.Module):
    """
    Adaptive LayerNorm Zero (adaLN-Zero) used in Diffusion Transformers (DiT).

    This module:
    - Applies standard LayerNorm
    - Generates scale (gamma), shift (beta), and gate (alpha)
    - Returns (x_modulated, gate)
    - Does NOT apply residuals or attention/MLP
    """

    def __init__(self, config: TransformerConfig, hidden_size: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm: WrappedTorchLayerNorm = WrappedTorchLayerNorm(config, hidden_size, eps=eps)
        self.cond_proj: nn.Linear = nn.Linear(cond_dim, 3 * hidden_size, bias=True)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, S, D]
            cond: Conditioning tensor [B, C] or [B, S, C]

        Returns:
            x_modulated: Normalized and modulated input (for Attn/MLP)
            gate: Gating tensor applied AFTER Attn/MLP
        """
        x_norm = self.norm(x)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        gamma, beta, gate = self.cond_proj(cond).chunk(3, dim=-1)
        x_modulated = x_norm * (1 + gamma) + beta
        return x_modulated, gate
