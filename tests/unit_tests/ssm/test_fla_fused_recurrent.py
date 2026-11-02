# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLA's fused recurrent KDA and GDN2 kernels must read only the gate rows they are given.

The kernels tile the head dim K to BK = next_power_of_2(K) and read each (token, head) row of the
per-channel gate ``g`` with stride K. Before FLA 0.6.0 that load had no mask, so the last row read
BK - K elements past the end of ``g``, and a NaN pattern there turned that head's output NaN.
Megatron runs these kernels on packed single-token sequences, and KDA and GDN2 accept head dims
that are not a power of two (96, 160, 192, ...).

Each case runs one step from a zero state, whose output does not depend on the gate, with ``g`` a
contiguous view at the start of a NaN-filled buffer.
"""

import pytest
import torch

try:
    from fla.ops.gdn2.fused_recurrent import fused_recurrent_gdn2
    from fla.ops.kda import fused_recurrent_kda

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

pytestmark = [
    pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

NUM_TOKENS, NUM_HEADS = 6, 4


def _inputs(head_dim):
    """BF16 [N, 1, H, K] q/k/v and a gate followed by NaN, as single-token sequences pass them."""
    gen = torch.Generator(device="cuda").manual_seed(head_dim)
    shape = (NUM_TOKENS, 1, NUM_HEADS, head_dim)

    def randn(*size):
        return torch.randn(*size, generator=gen, device="cuda")

    q, k, v = (randn(*shape).bfloat16() for _ in range(3))
    numel = q.numel()
    buffer = torch.full((numel + 1024,), float("nan"), dtype=torch.bfloat16, device="cuda")
    buffer[:numel] = randn(numel)
    g = buffer[:numel].view(shape)
    a_log = randn(NUM_HEADS)
    dt_bias = randn(NUM_HEADS * head_dim)
    return q, k, v, g, a_log, dt_bias, randn


def _one_step(q, k, value, l2norm):
    """Output of one step from a zero state: ``value * (q . k) / sqrt(K)``, in FP32."""
    q, k = q.float(), k.float()
    if l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    return value * (q * k).sum(-1, keepdim=True) * q.shape[-1] ** -0.5


@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("l2norm", [False, True])
@pytest.mark.parametrize("head_dim", [64, 96, 192])
def test_fused_recurrent_kda_reads_only_g(head_dim, l2norm, lower_bound):
    q, k, v, g, a_log, dt_bias, randn = _inputs(head_dim)
    beta = torch.sigmoid(randn(NUM_TOKENS, 1, NUM_HEADS))
    out, _ = fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=a_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=l2norm,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
    )
    expected = _one_step(q, k, beta[..., None] * v.float(), l2norm)
    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("l2norm", [False, True])
@pytest.mark.parametrize("head_dim", [64, 96, 192])
def test_fused_recurrent_gdn2_reads_only_g(head_dim, l2norm):
    q, k, v, g, a_log, dt_bias, randn = _inputs(head_dim)
    erase = torch.sigmoid(randn(*q.shape)).bfloat16()
    write = torch.sigmoid(randn(*v.shape)).bfloat16()
    out, _ = fused_recurrent_gdn2(
        q,
        k,
        v,
        g,
        erase,
        write,
        A_log=a_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=l2norm,
        use_gate_in_kernel=True,
    )
    expected = _one_step(q, k, write.float() * v.float(), l2norm)
    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=2e-2)
