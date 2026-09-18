# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Output-gating replay coverage adapted from Layali Rashid's PR #7368."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions import fused_gated_norm
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet
from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("strided_gate", [False, True])
def test_output_norm_replays(zero_centered, strided_gate):
    te = pytest.importorskip("transformer_engine.pytorch")
    seeded()
    norm = te.RMSNorm(
        128, eps=1e-6, params_dtype=torch.bfloat16, zero_centered_gamma=zero_centered, device="cuda"
    )

    class GatedNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                deterministic_mode=False, gdn_gated_output_norm_fusion=True
            )
            self.cp_size = 1
            self.activation = "silu"
            self.out_norm = norm
            self.value_head_dim = 128

        def forward(self, x, gate):
            # The module entry checks the supported layout on every replay.
            result = GatedDeltaNet._apply_gated_norm(self, x, gate)
            assert result.shape == x.shape
            return result

    x = torch.randn((1, 1027, 16, 128), device="cuda", dtype=torch.bfloat16).requires_grad_()
    projection = torch.randn((1, 1027, 5152), device="cuda", dtype=x.dtype)
    gate = projection[..., 3072:5120].reshape_as(x)
    if not strided_gate:
        gate = gate.contiguous()
    gate = gate.detach().requires_grad_()

    with patch.object(
        fused_gated_norm, "fused_gated_norm", wraps=fused_gated_norm.fused_gated_norm
    ) as fused:
        outputs, grads = assert_module_replays_bit_exact(
            GatedNorm(), (x, gate), replays=3, contention=True, what="GDN output RMSNorm/SiLU"
        )
        assert fused.call_count >= 3
    assert len(outputs) == 1 and len(grads) == 3
