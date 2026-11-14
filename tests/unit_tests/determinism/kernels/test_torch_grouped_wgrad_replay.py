# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Byte-exact grouped-linear replay, including all weight gradients."""

import os

import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.extensions.torch_grouped_wgrad import _grouped_mm, torch_grouped_linear
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.mark.parametrize("counts", [[0, 1, 17, 129], [128] * 4, [0] * 4])
@pytest.mark.parametrize("contention", [False, True])
def test_grouped_linear_replays(counts, contention):
    """Replay forward, dgrad and dW while a side stream competes for SMs."""
    if not torch.cuda.is_available() or _grouped_mm() is None:
        pytest.skip("requires CUDA and grouped matmul")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 grouped weight-gradient path")
    torch.manual_seed(943)
    x = torch.randn(sum(counts), 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = [
        torch.randn(128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in counts
    ]
    dy = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16)

    def run(inp, *params):
        return torch_grouped_linear(inp, counts, params)

    assert_replays_bit_exact(
        run,
        (x, *weights),
        grad_outputs={"out": dy},
        contention=contention,
        what="SM90 BF16 grouped linear with torch dW",
    )
