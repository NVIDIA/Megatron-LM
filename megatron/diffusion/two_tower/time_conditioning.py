# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Timestep conditioning for the denoiser tower (PixArt-alpha adaLN-single).

Provides sinusoidal timestep embedding, an MLP encoder, and per-layer
adaptive layer-norm modulation (shift, scale, gate).  Only the denoiser
tower uses these; the context tower is never conditioned on *t*.

Public API:

    :class:`TimestepEmbedder`
        Maps scalar timesteps ``t in [0, 1]`` to dense vectors.

    :func:`modulate`
        Applies ``x * (1 + scale) + shift`` to hidden states.

    :func:`get_modulation_params`
        Splits a global time embedding into per-layer ``(shift, scale, gate)``
        via a learned bias table.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TimestepEmbedder(nn.Module):
    """Encode scalar timesteps as dense vectors via sinusoidal features + MLP.

    Timesteps are expected in ``[0, 1]`` and internally scaled to
    ``[0, max_period]`` before computing sinusoidal frequencies.

    Args:
        hidden_size (int): Output embedding dimension.
        frequency_embedding_size (int): Intermediate sinusoidal dimension.
        max_period (int): Upper bound for internal timestep scaling.
    """

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 1000
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal positional embeddings for timestep values.

        Args:
            t (Tensor): 1-D tensor of *N* timestep values (may be fractional).
            dim (int): Output dimension.
            max_period (int): Controls the minimum frequency of the sinusoids.

        Returns:
            Tensor: Embeddings ``(N, dim)``.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding.to(t.dtype)

    def forward(self, t: Tensor) -> Tensor:
        """Embed a batch of timesteps.

        Args:
            t (Tensor): Timesteps ``(B,)`` in ``[0, 1]``.

        Returns:
            Tensor: Dense embeddings ``(B, hidden_size)``.
        """
        t_scaled = t * self.max_period
        t_freq = self.timestep_embedding(t_scaled, self.frequency_embedding_size)
        return self.mlp(t_freq)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Adaptive layer-norm affine transform: ``x * (1 + scale) + shift``.

    Supports both sequence-first ``(S, B, D)`` (Megatron convention) and
    batch-first ``(B, L, D)`` layouts.

    Args:
        x (Tensor): Hidden states, 3-D.
        shift (Tensor): Shift vector ``(B, D)``.
        scale (Tensor): Scale vector ``(B, D)``.

    Returns:
        Tensor: Modulated hidden states with the same shape as *x*.
    """
    if x.dim() == 3 and shift.dim() == 2:
        return x * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_modulation_params(
    t_emb: Tensor, scale_shift_table: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Derive per-layer ``(shift, scale, gate)`` from a global time embedding.

    The global ``t_emb`` vector (produced by ``t_block``) is reshaped to
    ``(B, 3, D)`` and added to a learned per-layer bias ``(3, D)``, then
    split into three ``(B, D)`` vectors.

    Args:
        t_emb (Tensor): Global time embedding ``(B, 3*D)``.
        scale_shift_table (Tensor): Per-layer bias ``(3, D)``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: ``(shift, scale, gate)`` each ``(B, D)``.
    """
    B = t_emb.shape[0]
    D = scale_shift_table.shape[1]

    t_emb_reshaped = t_emb.reshape(B, 3, D)
    combined = scale_shift_table[None] + t_emb_reshaped

    shift, scale, gate = combined.chunk(3, dim=1)
    return shift.squeeze(1), scale.squeeze(1), gate.squeeze(1)
