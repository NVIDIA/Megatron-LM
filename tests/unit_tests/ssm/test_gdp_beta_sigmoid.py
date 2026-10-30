# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer


@pytest.mark.parametrize("cutedsl", [False, True])
def test_beta_sigmoid_forward_and_backward(cutedsl: bool) -> None:
    """Preserve BF16 beta values without losing the sigmoid gradient at saturation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.tensor(
        [-8, -6, -1, 0, 1, 5, 6, 6.25, 7, 8, 8.5, 9], device=device, dtype=torch.bfloat16
    ).reshape(2, 1, 6)
    ba = torch.cat((logits, torch.zeros_like(logits[..., :2])), dim=-1).requires_grad_()
    mixer = SimpleNamespace(
        nheads_local_cp=2,
        num_householder=3,
        chunkwise_context_parallel=True,
        A_log=torch.zeros(2, device=device),
        dt_bias=torch.zeros(2, device=device),
        training=False,
        config=SimpleNamespace(gdp_cutedsl_kernel=cutedsl),
    )
    beta, _ = GatedDeltaProductMixer._compute_gating(cast(GatedDeltaProductMixer, mixer), ba)
    expected_shape = (2, 6) if cutedsl else (2, 3, 2)
    torch.testing.assert_close(beta, logits.sigmoid().reshape(expected_shape), rtol=0, atol=0)
    assert beta.dtype == torch.bfloat16
    assert beta.is_contiguous()

    upstream = torch.linspace(-2, 3, beta.numel(), device=device).to(beta.dtype).reshape_as(beta)
    (grad,) = torch.autograd.grad(beta, ba, upstream)
    # Independent FP64 derivative; allow one BF16 relative rounding unit.
    s = logits.double().sigmoid()
    expected = (upstream.reshape_as(logits).double() * s * (1 - s)).to(logits.dtype)
    torch.testing.assert_close(grad[..., :6], expected, rtol=torch.finfo(logits.dtype).eps, atol=0)
    saturated = logits >= 6.25
    assert torch.all(logits.sigmoid()[saturated] == 1)
    assert torch.all(grad[..., :6][saturated] != 0)
