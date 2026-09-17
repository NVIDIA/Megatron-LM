# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_head_publishes_weight_before_cast(dtype):
    weight = torch.nn.Parameter(torch.zeros(3, 4))
    calls = []

    def publish():
        calls.append(True)
        with torch.no_grad():
            weight.fill_(2)

    setattr(weight, PARAM_READY_CALLBACK_ATTR, publish)
    model = SimpleNamespace(
        head=SimpleNamespace(col=SimpleNamespace(linear=SimpleNamespace(weight=weight)))
    )
    hidden = torch.zeros(1, 4, dtype=dtype)
    actual = Qwen3MoEModel._head_weight_for_fused_ce(model, hidden)
    assert calls == [True]
    assert actual.dtype == dtype
    assert torch.equal(actual, torch.full_like(actual, 2))
    actual.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))
    delattr(weight, PARAM_READY_CALLBACK_ATTR)
    assert torch.equal(Qwen3MoEModel._head_weight_for_fused_ce(model, hidden), actual)
