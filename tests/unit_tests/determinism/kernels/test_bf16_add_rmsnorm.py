# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay under contention for the BF16 residual/RMSNorm fusion."""

import os

import pytest
import torch

pytest.importorskip("triton")

from megatron.core.ops.normalization.bf16_add_rmsnorm import fused_bf16_add_rmsnorm
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "rows,hidden", [(257, 128), (256, 768), (257, 2688), (256, 4096), (256, 8192), (257, 8192)]
)
def test_bf16_add_rmsnorm_replays(zero_centered_gamma, inplace, rows, hidden):
    """Stress independent rows and return both outputs to the byte comparator."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    seeded()
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    def run(x, residual, weight):
        return fused_bf16_add_rmsnorm(
            x, residual, weight, zero_centered_gamma=zero_centered_gamma, inplace=inplace
        )

    assert_replays_bit_exact(
        run,
        (x, residual, weight),
        replays=3,
        backward=False,
        contention=True,
        what="BF16 add/RMSNorm with TE native reduction",
    )
