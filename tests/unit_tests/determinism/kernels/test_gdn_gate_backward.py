# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the pre-GDR gate gradients before casting away FP32 reduction differences."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_pre_gated_delta_rule import _triton_g_beta_backward
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.mark.parametrize(
    "batch,length,heads", [(1, 33, 8), (2, 2048, 16), (1, 4096, 16), (3, 4103, 24)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_gate_backward_replays(
    batch: int, length: int, heads: int, dtype: torch.dtype, strided: bool
) -> None:
    """Compare all gradients bit-exactly under contention, and against Torch autograd."""
    seeded()
    key_heads, dim = 4, 32
    beta_offset = 2 * key_heads * dim + 2 * heads * dim
    channels = beta_offset + 2 * heads
    qkvzba = torch.randn(length, batch, channels, dtype=dtype, device="cuda")
    A_log = torch.randn(heads, dtype=dtype, device="cuda") * 0.2
    dt_bias = torch.randn(heads, dtype=dtype, device="cuda") * 0.2
    d_g = torch.randn(batch, length, heads * (2 if strided else 1), device="cuda")
    d_beta = torch.randn_like(d_g)
    d_g, d_beta = d_g[..., :heads], d_beta[..., :heads]
    stream = torch.cuda.Stream() if strided else None

    def backward(
        qkvzba: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        d_g: torch.Tensor,
        d_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = _triton_g_beta_backward(
            qkvzba,
            A_log,
            dt_bias,
            d_g,
            d_beta,
            num_value_heads=heads,
            key_head_dim=dim,
            value_head_dim=dim,
            num_key_heads=key_heads,
            d_qkvzba_out=torch.zeros_like(qkvzba) if strided else None,
            stream=stream,
        )
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        return result

    outputs, _ = assert_replays_bit_exact(
        backward,
        (qkvzba, A_log, dt_bias, d_g, d_beta),
        backward=False,
        replays=5,
        contention=True,
        what="pre-GDR gate backward",
    )

    projection = qkvzba.float().detach().requires_grad_()
    a = A_log.float().detach().requires_grad_()
    bias = dt_bias.float().detach().requires_grad_()
    beta = projection[..., beta_offset : beta_offset + heads].permute(1, 0, 2).sigmoid()
    alpha = projection[..., beta_offset + heads :].permute(1, 0, 2)
    g = -a.exp() * F.softplus(alpha + bias)
    reference = torch.autograd.grad((g, beta), (projection, a, bias), (d_g, d_beta))
    reference = (reference[0].to(dtype), reference[1], reference[2])
    torch.testing.assert_close(outputs["out[0]"], reference[0])
    for index in (1, 2):
        # Torch's exp/sigmoid approximations and reduction tree differ from
        # Triton's. This numerical reference is separate from the byte replay.
        torch.testing.assert_close(outputs[f"out[{index}]"], reference[index], atol=1e-4, rtol=2e-5)
