# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the fused Sinkhorn step: momentum, row norms, column sums and update."""

import pytest
import torch

from megatron.core.optimizer.sinkhorn_fused import HAVE_TRITON, fused_sinkhorn_step
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON, reason="requires CUDA and Triton"
)


@pytest.mark.parametrize(
    "rows,local_rows,cols",
    [(65543, 65536, 256), (2048, 2048, 5120)],
    ids=["engram-shard-padded", "embedding-width"],
)
def test_fused_sinkhorn_step_replays(rows, local_rows, cols):
    """Two consecutive steps; every column sum adds the partials of hundreds to thousands of
    programs (2048 at 64K x 256, 512 at 2048 x 5120) in a two-level fixed-order sum, and the
    masking threshold depends on the summed row norms of as many programs. Replays under
    side-stream contention must match bit for bit, also in the padding rows' momentum.
    """
    seeded()
    param = torch.randn(rows, cols, device="cuda")
    momentum = torch.zeros(rows, cols, device="cuda")
    grads = [torch.randn(rows, cols, device="cuda") for _ in range(2)]
    for grad in grads:
        grad[:100].zero_()
        grad[100:200].mul_(1e-6)
        grad[local_rows:] = 1e5

    def step(param, momentum, *grads):
        for grad in grads:
            fused_sinkhorn_step(
                param,
                grad,
                momentum,
                lr=0.02,
                momentum=0.95,
                eps=1e-20,
                num_normalization_steps=11,
                zero_row_threshold=1e-3,
                extra_scale_factor=0.18,
                local_rows=local_rows,
                mean_rows=local_rows,
            )
        return param, momentum

    outputs, _ = assert_replays_bit_exact(
        step,
        (param, momentum, *grads),
        replays=3,
        backward=False,
        contention=True,
        what="fused Sinkhorn step",
    )
    updated, _ = outputs["out[0]"], outputs["out[1]"]
    assert torch.equal(updated[local_rows:], param[local_rows:])
    assert torch.equal(updated[:100], param[:100])
    assert not torch.equal(updated[200:local_rows], param[200:local_rows])
