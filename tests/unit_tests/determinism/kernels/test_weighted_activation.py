# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay chunked router weighting under concurrent GPU work."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.ops.moe import weighted_activation
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("activation", [None, F.silu, squared_relu])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunked_weighting_replays_bit_exact(monkeypatch, activation, dtype):
    """Replay outputs and both input gradients across full and partial chunks."""
    seeded()
    monkeypatch.setattr(weighted_activation, "_MAX_TEMP_ELEMENTS", 256 * 1024)
    features = 4096
    inputs = torch.randn(
        513, features * (2 if activation else 1), device="cuda", dtype=dtype, requires_grad=True
    )
    probs = torch.rand(513, 1, device="cuda", dtype=torch.float32, requires_grad=True)

    def glu(values):
        gate, linear = values.chunk(2, dim=-1)
        return activation(gate) * linear

    def apply(values, weights):
        with torch.autocast("cuda", dtype=dtype, enabled=activation is squared_relu):
            if activation is not None:
                return weighted_activation.apply_weighted_activation(glu, values, weights)
            return weighted_activation.apply_router_probs(values, weights)

    assert_replays_bit_exact(
        apply, (inputs, probs), replays=3, contention=True, what="chunked router weighting"
    )
