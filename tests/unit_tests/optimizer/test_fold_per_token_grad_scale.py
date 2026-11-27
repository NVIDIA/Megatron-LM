# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, scale_grads_fp32
from megatron.core.optimizer.optimizer import ChainedOptimizer
from tests.unit_tests.optimizer.test_num_zeros_in_grads import _model_and_optimizer
from tests.unit_tests.test_utilities import Utils


def _params_with_grads():
    params = [torch.nn.Parameter(torch.randn(n, device="cuda")) for n in (7, 300, 70000)]
    for p in params:
        p.grad = torch.randn_like(p)
    return params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scale_grads_fp32():
    params = _params_with_grads()
    expected = [p.grad * 0.25 for p in params]
    scale_grads_fp32(params, torch.tensor(0.25, device="cuda"))
    for p, e in zip(params, expected):
        torch.testing.assert_close(p.grad, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("max_norm", [1e-3, 1e6])
def test_clip_with_deferred_scaling_equals_scale_then_clip(max_norm):
    params = _params_with_grads()
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    for r, p in zip(reference, params):
        r.grad = p.grad.clone()
    scaling = torch.tensor(1.0 / 37, device="cuda")
    norm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params)).item()
    scale_grads_fp32(reference, scaling)
    clip_grad_by_total_norm_fp32(reference, max_norm, norm / 37)
    clip_grad_by_total_norm_fp32(params, max_norm, norm / 37, grad_scaling=scaling)
    for p, r in zip(params, reference):
        torch.testing.assert_close(p.grad, r.grad, rtol=1e-6, atol=0)


class TestFoldPerTokenGradScale:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _step(self, fold, clip_grad):
        torch.manual_seed(0)
        model, optimizer = _model_and_optimizer(
            fp32_param=True, bucket_size=2000, clip_grad=clip_grad, fold_per_token_grad_scale=fold
        )
        assert getattr(model, "fold_grad_scaling", False) == fold
        torch.manual_seed(1)
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        model(x).float().sum().backward()
        finalize_model_grads([model], num_tokens=torch.tensor(37, device="cuda"))
        update_successful, grad_norm, _ = optimizer.step()
        assert update_successful
        assert getattr(model, "pending_grad_scaling", None) is None
        return grad_norm, [p.detach().float().clone() for p in model.parameters()]

    @pytest.mark.parametrize("clip_grad", [0.0, 1e-3, 1e6])
    def test_matches_scaling_in_finalize(self, clip_grad):
        norm_ref, params_ref = self._step(fold=False, clip_grad=clip_grad)
        norm, params = self._step(fold=True, clip_grad=clip_grad)
        if clip_grad > 0.0:
            # The reference rounds each scaled grad to the bf16 grad buffer; the fold scales the
            # fp32 norm, so they agree to bf16 rounding.
            torch.testing.assert_close(
                torch.tensor(float(norm)), torch.tensor(float(norm_ref)), rtol=2e-3, atol=0
            )
        for p, r in zip(params, params_ref):
            torch.testing.assert_close(p, r, rtol=1e-2, atol=1e-5)


def test_chained_step_with_found_inf_skips_and_drops_pending_scaling(monkeypatch):
    chunk = torch.nn.Module()
    chunk.pending_grad_scaling = torch.tensor(0.5)
    optimizer = ChainedOptimizer.__new__(ChainedOptimizer)
    optimizer.chained_optimizers = []
    optimizer.model_chunks = [chunk]
    monkeypatch.setattr(optimizer, "prepare_grads", lambda: True)

    assert optimizer.step() == (False, None, None)
    assert chunk.pending_grad_scaling is None
