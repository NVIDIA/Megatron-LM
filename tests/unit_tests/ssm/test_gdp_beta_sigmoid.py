# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer


@pytest.mark.parametrize("cudnn", [False, True])
def test_beta_sigmoid_forward_and_backward(cudnn: bool) -> None:
    """Preserve BF16 sigmoid values and gradients across the FLA and cuDNN layouts."""
    logits = (
        torch.tensor([-8, -6, -1, 0, 1, 5, 6, 6.25, 7, 8, 8.5, 9], dtype=torch.bfloat16)
        .reshape(2, 1, 6)
        .expand(2, 2, 6)
    )
    ba = torch.cat((logits, torch.zeros_like(logits[..., :2])), dim=-1).requires_grad_()
    mixer = SimpleNamespace(
        nheads_local_cp=2,
        num_householder=3,
        chunkwise_context_parallel=True,
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2),
        config=SimpleNamespace(gdp_cudnn_kernel=cudnn),
    )
    beta, _ = GatedDeltaProductMixer._compute_gating(cast(GatedDeltaProductMixer, mixer), ba)
    expected_shape = (4, 3, 2) if cudnn else (2, 6, 2)
    torch.testing.assert_close(beta, logits.sigmoid().reshape(expected_shape), rtol=0, atol=0)
    assert beta.dtype == torch.bfloat16
    assert beta.is_contiguous()

    upstream = torch.linspace(-2, 3, beta.numel()).to(beta.dtype).reshape_as(beta)
    (grad,) = torch.autograd.grad(beta, ba, upstream)
    reference_logits = logits.detach().requires_grad_()
    (expected,) = torch.autograd.grad(
        reference_logits.sigmoid(), reference_logits, upstream.reshape_as(logits)
    )
    torch.testing.assert_close(grad[..., :6], expected, rtol=0, atol=0)
