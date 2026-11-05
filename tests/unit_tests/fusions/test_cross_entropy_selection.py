# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Version/API dispatch and numerical parity for the TE-preferring tiled loss."""

import pytest
import torch
from packaging.version import Version

from megatron.core.extensions import transformer_engine as te_extensions
from megatron.core.models import backends
from megatron.core.ops.cross_entropy import triton_cross_entropy
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "version,api,available,prefer_te",
    [
        ("2.18.0", True, True, False),
        ("2.19.0.dev0", False, True, False),
        ("2.19.0.dev0", True, True, True),
        ("2.19.0", True, True, True),
        ("2.20.0", True, False, False),
        ("2.20.0", False, True, False),
    ],
)
def test_auto_checks_version_and_api(monkeypatch, version, api, available, prefer_te):
    monkeypatch.setattr(
        backends, "is_te_min_version", lambda minimum: Version(version) >= Version(minimum)
    )
    monkeypatch.setattr(te_extensions, "_TE_FUSED_PARALLEL_CE_OVERWRITE_INPUT", api, raising=False)
    monkeypatch.setattr(
        te_extensions, "te_parallel_cross_entropy", (lambda: None) if available else None
    )
    target = backends.select_cross_entropy(True, "auto")
    if prefer_te:
        assert target.func is backends.te_cross_entropy
        assert target.keywords["overwrite_input"] is False
    else:
        assert target is triton_cross_entropy


@pytest.mark.parametrize("implementation", ["fallback", "te"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_auto_loss_and_ignored_rows(implementation, dtype, monkeypatch):
    if implementation == "te":
        if not backends._te_has_memory_efficient_cross_entropy():
            pytest.skip("TE >= 2.19 with memory-efficient cross entropy required")
    else:
        monkeypatch.setattr(backends, "_te_has_memory_efficient_cross_entropy", lambda: False)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        from megatron.core import parallel_state

        torch.manual_seed(1731)
        source = torch.randn(17, 3, 4097, device="cuda", dtype=dtype)
        logits = source.clone().requires_grad_()
        reference = source.float().clone().requires_grad_()
        labels = torch.randint(4097, (17, 3), device="cuda")
        labels[::3] = -100
        scale = torch.randn(17, 3, device="cuda")
        actual = backends.select_cross_entropy(True, "auto")(
            logits, labels, parallel_state.get_tensor_model_parallel_group()
        )
        expected = torch.nn.functional.cross_entropy(
            reference.reshape(-1, 4097), labels.reshape(-1), reduction="none"
        ).view_as(labels)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (actual * scale).sum().backward()
        (expected * scale).sum().backward()
        torch.testing.assert_close(
            logits.grad,
            reference.grad.to(dtype),
            rtol=8e-3 if dtype == torch.bfloat16 else 2e-5,
            atol=1e-7,
        )
        assert torch.equal(logits.detach(), source)
        assert torch.count_nonzero(logits.grad[::3]) == 0
    finally:
        Utils.destroy_model_parallel()
