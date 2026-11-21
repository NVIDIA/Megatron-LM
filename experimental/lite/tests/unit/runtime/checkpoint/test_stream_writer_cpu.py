# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import torch

from megatron.lite.runtime import megatron_utils


def test_gradient_release_restore_keeps_aliases_and_clears_stale_gradients(monkeypatch):
    monkeypatch.setattr(megatron_utils, '_is_megatron_ddp', lambda _: True)
    grad = torch.arange(8.0)
    alias = grad[2:6]
    params = torch.arange(4.0)
    buffer = SimpleNamespace(grad_data=grad, param_data=params)
    model = SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[])
    megatron_utils.release_grad_buffers([model])
    megatron_utils.release_grad_buffers([model])
    assert grad.untyped_storage().nbytes() == 0
    assert torch.equal(params, torch.arange(4.0))
    megatron_utils.restore_grad_buffers([model])
    assert torch.equal(alias, torch.zeros(4))
    alias.add_(3)
    megatron_utils.restore_grad_buffers([model])
    assert torch.equal(alias, torch.full((4,), 3.0))
