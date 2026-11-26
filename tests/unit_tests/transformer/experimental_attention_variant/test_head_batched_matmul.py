# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``head_batched_matmul`` equals the per-head einsum of absorbed MLA, forward and backward,
and reads and writes the token-major layout without copy kernels."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.head_batched_matmul import (
    head_batched_matmul,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(transposed_weight: bool):
    generator = torch.Generator(device="cuda").manual_seed(7)
    tokens, heads, width, nope, rank = 96, 8, 48, 32, 64
    # The no-PE slice of a wider projection, as the query absorption reads it.
    q = torch.randn(
        tokens, 1, heads, width, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    x = q[..., :nope].detach().requires_grad_(True)
    if transposed_weight:
        base = torch.randn(
            heads, rank, nope, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        weight = base.detach().requires_grad_(True)
        return x, weight, weight.transpose(1, 2)
    weight = torch.randn(
        heads, nope, rank, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = weight.detach().requires_grad_(True)
    return x, weight, weight


@pytest.mark.parametrize("transposed_weight", [False, True])
def test_matches_einsum_forward_and_backward(transposed_weight):
    x, leaf, weight = _inputs(transposed_weight)
    out = head_batched_matmul(x, weight)
    assert out.is_contiguous() and out.shape == (*x.shape[:-1], weight.size(-1))
    grad = torch.randn_like(out)
    out.backward(grad)
    got = (out.detach(), x.grad.clone(), leaf.grad.clone())

    x2 = x.detach().clone().requires_grad_(True)
    leaf2 = leaf.detach().clone().requires_grad_(True)
    weight2 = leaf2.transpose(1, 2) if transposed_weight else leaf2
    ref = torch.einsum("...nd,ndk->...nk", x2, weight2).contiguous()
    ref.backward(grad)
    want = (ref.detach(), x2.grad, leaf2.grad)
    for name, a, b in zip(("out", "dx", "dw"), got, want):
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2, msg=name)


def test_forward_reads_and_writes_token_major_without_copies():
    """The forward is one GEMM; einsum adds a copy of the strided input and of the output."""
    x, _leaf, weight = _inputs(transposed_weight=True)

    def kernels(fn):
        fn()  # warm up cuBLAS outside the profile
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]

    ours = kernels(lambda: head_batched_matmul(x, weight))
    theirs = kernels(lambda: torch.einsum("...nd,ndk->...nk", x, weight).contiguous())
    assert len(ours) == 1, ours
    assert len(theirs) > len(ours), theirs


def test_falls_back_on_cpu():
    x = torch.randn(4, 2, 3, 5)
    weight = torch.randn(3, 5, 6)
    torch.testing.assert_close(
        head_batched_matmul(x, weight), torch.einsum("...nd,ndk->...nk", x, weight)
    )
