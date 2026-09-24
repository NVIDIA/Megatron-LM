# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic replay of cuDNN Engram forward and all gate gradients."""

import pytest
import torch

from megatron.core.fusions.cudnn_engram import CudnnEngramGate
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)


class Gate(torch.nn.Module):
    """Expose original normalization parameters to the generic replay harness."""

    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.ones(4, 5120, device="cuda"))
        self.k = torch.nn.Parameter(torch.full_like(self.q, 0.75))
        self.gate = CudnnEngramGate(1e-20)

    def forward(self, x, kv):
        """Use a fresh forward state for each replay."""
        return self.gate(x, kv, self.q, self.k)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="cuDNN gate requires SM100"
)
def test_cudnn_engram_gate_replays_bit_exact():
    from tests.unit_tests.fusions.test_cudnn_engram import require_cudnn_engram

    require_cudnn_engram()
    seeded()
    module = Gate()
    inputs = (
        torch.randn(64, 1, 20480, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(64, 1, 25600, device="cuda", dtype=torch.bfloat16, requires_grad=True),
    )
    with deterministic_algorithms(True):
        assert_module_replays_bit_exact(module, inputs)
