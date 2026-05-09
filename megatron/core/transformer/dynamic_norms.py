# Copyright (c) 2026, ischlag. Apache-2.0.
"""DyT and Derf: element-wise drop-in replacements for Layer/RMSNorm.

* DyT  (Zhu et al., 2025, "Transformers without Normalization")
      gamma * tanh(alpha * x) + beta
* Derf (Chen et al., 2025, "Stronger Normalization-Free Transformers")
      gamma * erf(alpha * x + s) + beta

Both expose the standard Megatron norm signature
``__init__(config, hidden_size, eps=...)`` and a ``forward(x)`` that returns a
tensor with the same shape as the input. They have no token-axis reduction,
so they are sequence-parallel-safe (each rank's slice is processed
independently; gamma/beta/alpha grads sync via standard DDP since the
parameters are replicated, not TP-sharded).
"""

from __future__ import annotations

import torch
from torch import nn


class DyT(nn.Module):
    """DyT(x) = gamma * tanh(alpha * x) + beta.

    `alpha` is a single learnable scalar (initialised to 0.5 per the paper).
    `gamma` and `beta` are per-channel learnable vectors of size `hidden_size`,
    matching the affine parameters in LayerNorm/RMSNorm so checkpoints can be
    swapped one-for-one (with the obvious caveat that `alpha` is new state).
    `eps` is accepted but unused (DyT has no division by a norm).
    """

    def __init__(
        self,
        config=None,
        hidden_size: int | None = None,
        eps: float = 1e-5,
        alpha_init: float = 0.5,
        **_unused,
    ):
        super().__init__()
        if hidden_size is None:
            raise ValueError("DyT requires hidden_size")
        self.hidden_size = hidden_size
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.tanh(self.alpha * x) + self.bias


class Derf(nn.Module):
    """Derf(x) = gamma * erf(alpha * x + s) + beta.

    Followup to DyT; `alpha` and `s` are learnable scalars (init 0.5 / 0.0).
    Same per-channel `gamma`/`beta`. `eps` accepted but unused.
    """

    def __init__(
        self,
        config=None,
        hidden_size: int | None = None,
        eps: float = 1e-5,
        alpha_init: float = 0.5,
        s_init: float = 0.0,
        **_unused,
    ):
        super().__init__()
        if hidden_size is None:
            raise ValueError("Derf requires hidden_size")
        self.hidden_size = hidden_size
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.s = nn.Parameter(torch.tensor(float(s_init)))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.erf(self.alpha * x + self.s) + self.bias
