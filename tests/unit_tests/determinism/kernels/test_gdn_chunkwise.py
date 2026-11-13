# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay TE GDN state operations and MCore's right-oriented summary composition."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE_GDN, TEGatedDeltaNetAttention
from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardUnpackedSummary,
    CPForwardUnpackedSummary,
)
from megatron.core.ssm.gated_delta_net.chunkwise_core import _Backend, _Inputs
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.mark.skipif(not HAVE_TE_GDN, reason="Transformer Engine GDN is unavailable")
def test_gdn_chunkwise_local_state_replays():
    """Include explicit local backward results in the bitwise replay outputs."""
    torch.manual_seed(72)
    tokens, heads, dim = 4096, 4, 128
    adapter = TEGatedDeltaNetAttention(heads, dim, dim, 1)
    backend = _Backend(adapter.get_context_parallel_backend())
    q = F.normalize(torch.randn(tokens, heads, dim, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn_like(q)
    g = -0.05 * torch.rand(tokens, heads, device="cuda")
    beta = torch.rand_like(g)
    cu = torch.tensor([0, 1024, tokens], device="cuda", dtype=torch.int32)
    output_grad = torch.randn_like(v)

    def run(q, k, v, g, beta, output_grad):
        summary, inputs = backend.cp_forward_prepare(_Inputs(q, k, v, g, beta, cu, dim**-0.5))
        # Exercise nonempty prefix/suffix composition and the state-gradient path.
        preceding = CPForwardUnpackedSummary(
            summary.transition.unsqueeze(0), summary.state_update.unsqueeze(0)
        )
        output, saved = backend.cp_forward_apply(inputs, preceding)
        reverse, backward_context = backend.cp_backward_prepare(output_grad, saved)
        following = CPBackwardUnpackedSummary(
            reverse.transition.unsqueeze(0), reverse.local_state_grad.unsqueeze(0)
        )
        gradients = backend.cp_backward_apply(backward_context, following)
        return (
            output,
            summary.transition,
            summary.state_update,
            reverse.transition,
            reverse.local_state_grad,
            *gradients,
        )

    assert_replays_bit_exact(
        run, (q, k, v, g, beta, output_grad), replays=3, backward=False, contention=True
    )
