# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RMSNorm for an FP32 inference residual stream and model-dtype weights."""

import torch

from megatron.core.jit import jit_fuser


@jit_fuser
def fp32_residual_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Normalize before rounding, returning the dtype required by the next GEMM.

    Both the reduction and weight multiplication retain FP32. In particular,
    do not cast the residual input or the normalized activation to BF16 before
    the weight multiplication, even when executing without compiler fusion.
    """
    x = x.float()
    scale = weight.float() + 1.0 if zero_centered_gamma else weight.float()
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * scale).to(weight.dtype if output_dtype is None else output_dtype)
