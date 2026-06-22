# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch


class Gemma4RMSNorm(torch.nn.Module):
    """Bitwise-faithful port of HF ``Gemma4RMSNorm`` (modeling_gemma4.py:193-211).

    The compute is intentionally an exact mirror of HF: cast to fp32, normalize
    with ``pow(2).mean(-1) + eps`` and ``torch.pow(., -0.5)`` (NOT ``rsqrt`` — HF
    uses ``pow`` to match JAX), multiply by ``weight.float()`` (plain weight, no
    ``+1``), then cast back to the input dtype only at the end.

    ``with_scale=False`` gives the weightless variant used for the scaleless
    v_norm; it has no parameter and is a pure RMS normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)
