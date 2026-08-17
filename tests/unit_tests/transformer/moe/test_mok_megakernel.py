# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe import mok_megakernel


def _parameter_with_main_grad(shape=(4, 8)):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(shape, dtype=torch.float32)
    param.grad_added_to_main_grad = False
    return param


def test_accumulate_weight_gradient_adds_to_main_grad_and_returns_dummy(monkeypatch):
    param = _parameter_with_main_grad()
    grad = torch.full_like(param, 0.5)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(mok_megakernel, "_dummy_weight_gradient", lambda _: dummy)

    actual = mok_megakernel._accumulate_weight_gradient(param, grad)
    mok_megakernel._accumulate_weight_gradient(param, grad)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.ones_like(param.main_grad))
    assert param.grad_added_to_main_grad


def test_accumulate_weight_gradient_requires_main_grad():
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="param.main_grad"):
        mok_megakernel._accumulate_weight_gradient(param, torch.zeros_like(param))


def test_accumulate_weight_gradient_rejects_shape_mismatch():
    param = _parameter_with_main_grad()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        mok_megakernel._accumulate_weight_gradient(
            param, torch.zeros((2, 16), dtype=torch.bfloat16)
        )


def test_finish_weight_gradient_marks_ready_without_accumulating(monkeypatch):
    param = _parameter_with_main_grad()
    param.main_grad.fill_(0.25)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(mok_megakernel, "_dummy_weight_gradient", lambda _: dummy)

    actual = mok_megakernel._finish_weight_gradient(param)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.full_like(param.main_grad, 0.25))
    assert param.grad_added_to_main_grad


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_main_grad_buffer_requires_contiguous_fp32(dtype):
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.zeros((4, 8), dtype=dtype)

    with pytest.raises(RuntimeError, match="contiguous FP32"):
        mok_megakernel._main_grad_buffer(param)
