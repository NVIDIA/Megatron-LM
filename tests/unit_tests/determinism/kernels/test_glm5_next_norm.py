# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay coverage for GLM5NextRMSNorm, eager and compiled (rmsnorm_cast_before_weight_compiled)."""

import pytest
import torch

from megatron.core.transformer.glm5_next_norm import GLM5NextRMSNorm
from tests.unit_tests.determinism.kernels.harness import assert_module_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("width", [128, 4096])
def test_glm5_next_rms_norm_replays(compiled, width):
    """Forward, input gradient and gamma gradient replay bit-exactly on both paths."""
    seeded()
    module = GLM5NextRMSNorm(
        width, eps=1e-5, params_dtype=torch.bfloat16, device="cuda", compiled=compiled
    )
    with torch.no_grad():
        module.weight.copy_(1.0 + 0.2 * torch.randn(width, device="cuda"))
    hidden = (3 * torch.randn(257, 3, width, device="cuda")).to(torch.bfloat16)
    hidden.requires_grad_(True)
    grad_output = torch.randn(257, 3, width, device="cuda").to(torch.bfloat16)
    assert_module_replays_bit_exact(
        module, (hidden,), grad_output=grad_output, what=f"GLM5NextRMSNorm(compiled={compiled})"
    )
