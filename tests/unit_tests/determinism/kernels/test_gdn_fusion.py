# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replay of GDN preparation and output gating."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ssm import gdn_fusion, gdn_gated_norm
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.mark.parametrize("boundaries", [None, [0, 4099], [0, 1, 1031, 2048, 4099]])
@pytest.mark.parametrize("has_bias", [False, True])
def test_preparation_replays(boundaries, has_bias):
    if gdn_fusion._LINEAR_BWD is None:
        pytest.skip("FLA convolution backward is unavailable")
    seeded()
    x = torch.randn((1, 4099, 5184), device="cuda", dtype=torch.bfloat16)
    x = x[..., :5152].detach().requires_grad_()
    weight = (torch.randn((3072, 4), device="cuda", dtype=x.dtype) * 0.1).requires_grad_()
    bias = torch.randn(3072, device="cuda", dtype=x.dtype).requires_grad_() if has_bias else None
    alog = torch.rand(16, device="cuda").requires_grad_()
    dtbias = torch.randn(16, device="cuda").requires_grad_()
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int64) if boundaries else None
    outputs, grads = assert_replays_bit_exact(
        gdn_fusion.fused_prepare,
        (x, weight, bias, alog, dtbias, cu),
        replays=3,
        contention=True,
        what="GDN preparation",
    )
    assert len(outputs) == 6 and len(grads) == (5 if has_bias else 4)


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("strided_gate", [False, True])
def test_output_norm_replays(monkeypatch, zero_centered, strided_gate):
    te = pytest.importorskip("transformer_engine.pytorch")
    seeded()
    monkeypatch.setenv("MCORE_GDN_FUSION", "1")
    norm = te.RMSNorm(
        128, eps=1e-6, params_dtype=torch.bfloat16, zero_centered_gamma=zero_centered, device="cuda"
    )

    class GatedNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(deterministic_mode=False)
            self.cp_size = 1
            self.activation = "silu"
            self.out_norm = norm
            self.value_head_dim = 128

        def forward(self, x, gate):
            assert gdn_gated_norm.enabled(self, x, gate)
            return GatedDeltaNet._apply_gated_norm(self, x, gate)

    x = torch.randn((1, 1027, 16, 128), device="cuda", dtype=torch.bfloat16).requires_grad_()
    projection = torch.randn((1, 1027, 5152), device="cuda", dtype=x.dtype)
    gate = projection[..., 3072:5120].reshape_as(x)
    if not strided_gate:
        gate = gate.contiguous()
    gate = gate.detach().requires_grad_()

    outputs, grads = assert_module_replays_bit_exact(
        GatedNorm(), (x, gate), replays=3, contention=True, what="GDN output RMSNorm/SiLU"
    )
    assert len(outputs) == 1 and len(grads) == 3
