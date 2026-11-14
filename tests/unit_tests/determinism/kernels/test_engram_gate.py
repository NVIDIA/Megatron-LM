# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the fused Engram gate: output, input and q/k-weight gradients."""

import pytest
import torch

from megatron.core.models.deepseek_v41.engram_gate import HAVE_TRITON, fused_engram_gate
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    seeded,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON, reason="requires CUDA and Triton"
)


@pytest.mark.parametrize(
    "batch,masked,padded",
    [(1, False, False), (2, True, False), (1, True, True)],
    ids=["B1", "B2-masked", "B1-masked-padded-rows"],
)
def test_engram_fused_gate_replays(batch, masked, padded):
    """Proxy shape (4096 tokens, 4 streams of 5120): every q/k-weight element sums 32
    token-chunk partials written by different programs, and every token's value gradient
    sums its 4 streams; replays under side-stream contention must match bit for bit.
    ``padded`` passes kv as the MXFP8 lookup's projection output: [4096, width] rows holding
    4072 tokens, whose gradient has pad rows the gate zeroes.
    """
    seeded()
    streams, hidden_size = 4, 5120
    width = (streams + 1) * hidden_size
    length = (CONTENTION_TOKENS - 24 * padded) // batch
    device = torch.device("cuda", torch.cuda.current_device())
    hidden = torch.randn(
        length, batch, streams * hidden_size, device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    if padded:
        kv = torch.randn(CONTENTION_TOKENS, width, device=device, dtype=torch.bfloat16)
    else:
        # Engram's kv: a [batch, sequence] projection output seen as [sequence, batch].
        kv = torch.randn(batch, length, width, device=device, dtype=torch.bfloat16).transpose(0, 1)
    kv.requires_grad_(True)
    q_weight = (1 + 0.1 * torch.randn(streams * hidden_size, device=device)).bfloat16()
    k_weight = (1 + 0.1 * torch.randn(streams * hidden_size, device=device)).bfloat16()
    mask = (torch.rand(batch, length, device=device) > 0.2) if masked else None
    grad = torch.randn(length, batch, streams * hidden_size, device=device, dtype=torch.bfloat16)

    def gate(hidden, kv, q_weight, k_weight, mask):
        return fused_engram_gate(hidden, kv, q_weight, k_weight, mask, streams, hidden_size, 1e-6)

    _, grads = assert_replays_bit_exact(
        gate,
        (hidden, kv, q_weight.requires_grad_(True), k_weight.requires_grad_(True), mask),
        replays=3,
        contention=True,
        grad_outputs={"out": grad},
        what="fused Engram gate",
    )
    assert len(grads) == 4  # hidden, kv, q_weight and k_weight gradients were all compared
    assert grads["in[1]"].shape == kv.shape
    assert torch.count_nonzero(grads["in[1]"][length * batch :]) == 0  # pad rows (if any)
