# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP8 expert weights for the decode grouped GEMMs (weight-only, W8A16).

The expert GEMMs are the largest item in the decode budget and they are pinned
to memory bandwidth, not math: at BS256/EP4 every step streams all 32 local
experts' weights — 302 MB per layer — while the activations are a rounding error
next to that. Measured on GB200 the kernel already runs at ~93% of HBM peak, so
the only lever left is reading fewer bytes.

Quantizing the weights to fp8 e4m3 with one scale per output channel halves the
weight traffic and leaves activations and accumulation in their original
precision: each tile is converted back to bf16 in registers before the MMA, and
because the scale is per output channel it factors out of the K sum exactly
(``W[n,k] = q[n,k] * s[n]``) and is applied once to the fp32 accumulator.

This changes numerics, so it is off by default and must be reported separately
from any iso-precision result. Enable with ``MCORE_MOE_FP8_WEIGHTS=1``.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch

ENABLED: bool = os.environ.get("MCORE_MOE_FP8_WEIGHTS", "0") == "1"

# Largest finite magnitude of e4m3: the amax of each output channel maps here.
_E4M3_MAX: float = 448.0


@dataclass
class Fp8ExpertWeights:
    """An fp8 view of one grouped weight plus its per-output-channel scales."""

    weight: torch.Tensor  # [E, N, K] float8_e4m3fn
    scale: torch.Tensor  # [E, N] fp32


def quantize_expert_weights(weight: torch.Tensor) -> Fp8ExpertWeights:
    """Quantize ``[E, N, K]`` weights to e4m3 with one fp32 scale per (E, N).

    Quantizes expert by expert: the fp32 intermediate for all experts at once
    would be four times the size of the weight it is replacing.
    """
    num_experts = weight.size(0)
    q = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty(weight.shape[:2], dtype=torch.float32, device=weight.device)
    for e in range(num_experts):
        w = weight[e].float()
        # clamp: an all-zero output channel would otherwise divide by zero.
        amax = w.abs().amax(dim=1).clamp_(min=1e-12)
        s = amax / _E4M3_MAX
        q[e] = (w / s[:, None]).to(torch.float8_e4m3fn)
        scale[e] = s
    return Fp8ExpertWeights(weight=q, scale=scale)


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` activations to e4m3 with one fp32 scale per token.

    Needed only for the w8a8 path: with bf16 activations the weight tile has to
    be widened in registers, and that convert costs more than the halved weight
    traffic saves. Feeding the tensor cores fp8 on both sides removes it.
    """
    amax = x.abs().amax(dim=1).float().clamp_(min=1e-12)
    scale = amax / _E4M3_MAX
    q = (x.float() / scale[:, None]).to(torch.float8_e4m3fn)
    return q, scale


def maybe_quantize(weight: Optional[torch.Tensor]) -> Optional[Fp8ExpertWeights]:
    """Quantize when the gate is on and the source is a supported dtype."""
    if not ENABLED or weight is None:
        return None
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    return quantize_expert_weights(weight)
