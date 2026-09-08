# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replays of the two chunked gated-delta-rule implementations GDN dispatches to.

``deterministic_mode`` selects ``torch_chunk_gated_delta_rule``; the default is FLA's Triton
``chunk_gated_delta_rule``. Each is replayed on identical inputs and compared bit for bit, forward
and backward, under side-stream contention where the process allows it (the unit-test bucket pins
``CUDA_DEVICE_MAX_CONNECTIONS=1``; see ``tests/unit_tests/stream_contention.py``). FLA's kernel has
no atomics in its call graph and on fla 0.4.2 / GB300 four independent processes agreed bit for bit
as well; pinning its result here means a regression to an order-dependent reduction shows up before
anyone relies on it.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.ssm.gated_delta_net.gdn import torch_chunk_gated_delta_rule
from megatron.core.ssm.gated_delta_net.gdn2 import torch_chunk_gdn2
from tests.unit_tests.stream_contention import SideStreamContention, bit_equal, replay_count

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_FLA), reason="needs a GPU and flash-linear-attention"
)

OUTPUT_NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")


def _inputs(B=2, T=2048, H=4, K=128, V=128, seed=0):
    """GDN-shaped inputs as the mixer hands them to the kernel: q/k L2-normalised (without it the
    recurrence overflows to NaN for K=128) and gates from A_log = log U(1, 16)."""
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, V, generator=g).to(torch.bfloat16)
    a_log = torch.log(torch.empty(H).uniform_(1, 16, generator=g))
    dt = torch.empty(H).uniform_(1e-3, 1e-1, generator=g)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    gate = (-a_log.exp() * F.softplus(torch.randn(B, T, H, generator=g) + dt_bias)).float()
    beta = torch.randn(B, T, H, generator=g).sigmoid().to(torch.bfloat16)
    grad_out = torch.randn(B, T, H, V, generator=g).to(torch.bfloat16)
    return [t.cuda() for t in (q, k, v, gate, beta, grad_out)]


def _forward_backward(kernel, inputs, **kwargs):
    q, k, v, gate, beta, grad_out = inputs
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, gate, beta)]
    out, _ = kernel(*leaves, initial_state=None, output_final_state=False, **kwargs)
    grads = torch.autograd.grad(out, leaves, grad_out)
    torch.cuda.synchronize()
    return [out.detach().clone(), *[grad.detach().clone() for grad in grads]]


def _assert_replays_bit_exact(kernel, inputs, replays, **kwargs):
    reference = _forward_backward(kernel, inputs, **kwargs)
    # Guard against a vacuous pass: NaN patterns also compare equal.
    for name, t in zip(OUTPUT_NAMES, reference):
        assert torch.isfinite(t).all(), f"{kernel.__name__}: {name} is not finite"
    for _ in range(replay_count(with_contention=replays) - 1):
        with SideStreamContention():
            replay = _forward_backward(kernel, inputs, **kwargs)
        for name, a, b in zip(OUTPUT_NAMES, reference, replay):
            assert bit_equal(a, b), f"{kernel.__name__}: {name} differs between replays"
    return reference


def test_fla_chunk_gated_delta_rule_replays_bit_exact():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _assert_replays_bit_exact(
        chunk_gated_delta_rule, _inputs(), replays=4, use_qk_l2norm_in_kernel=False
    )


def test_torch_chunk_gated_delta_rule_replays_bit_exact():
    _assert_replays_bit_exact(
        torch_chunk_gated_delta_rule, _inputs(T=1024), replays=3, use_qk_l2norm_in_kernel=False
    )


def test_torch_path_in_kernel_l2norm_matches_normalising_outside():
    """``use_qk_l2norm_in_kernel=True`` calls FLA's l2norm with its current signature and equals
    normalising q and k before the call."""
    from fla.modules.l2norm import l2norm

    q, k, v, gate, beta, _ = _inputs(T=512)
    q, k = q * 3.0, k * 0.5  # un-normalised on purpose so the in-kernel l2norm has work to do
    fused, _ = torch_chunk_gated_delta_rule(q, k, v, gate, beta, use_qk_l2norm_in_kernel=True)
    outside, _ = torch_chunk_gated_delta_rule(
        l2norm(q), l2norm(k), v, gate, beta, use_qk_l2norm_in_kernel=False
    )
    assert torch.isfinite(fused).all()
    assert torch.equal(fused, outside)


def test_gdn2_torch_path_in_kernel_l2norm_matches_normalising_outside():
    """Same for the GDN2 torch path: ``use_qk_l2norm_in_kernel=True`` must call FLA's l2norm with
    its current signature and equal normalising q and k beforehand."""
    from fla.modules.l2norm import l2norm

    B, T, H, K, V = 2, 512, 4, 128, 128
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(B, T, H, K, generator=g) * 3.0).to(torch.bfloat16).cuda()
    k = (torch.randn(B, T, H, K, generator=g) * 0.5).to(torch.bfloat16).cuda()
    v = torch.randn(B, T, H, V, generator=g).to(torch.bfloat16).cuda()
    log_decay = (-F.softplus(torch.randn(B, T, H, K, generator=g)) * 0.1).cuda()
    erase = torch.rand(B, T, H, K, generator=g).cuda()
    write = torch.rand(B, T, H, V, generator=g).cuda()
    fused, _ = torch_chunk_gdn2(q, k, v, log_decay, erase, write, use_qk_l2norm_in_kernel=True)
    outside, _ = torch_chunk_gdn2(
        l2norm(q), l2norm(k), v, log_decay, erase, write, use_qk_l2norm_in_kernel=False
    )
    assert torch.isfinite(fused).all()
    assert torch.equal(fused, outside)
