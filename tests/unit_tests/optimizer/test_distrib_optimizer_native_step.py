# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from types import SimpleNamespace

import torch

from megatron.core.optimizer import distrib_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


def test_native_adam_step_counters_remain_independent_after_restore(monkeypatch):
    """A native Adam resume must advance each parameter's counter exactly once."""
    # Exercise the native backend even in CI images containing TE/Apex. The
    # optimizer and updates below are real Torch AdamW operations on CPU.
    monkeypatch.setattr(distrib_optimizer, 'HAVE_APEX_OR_TE', False)

    def make_optimizer():
        params = [torch.nn.Parameter(torch.tensor([1.0])), torch.nn.Parameter(torch.tensor([2.0]))]
        native = torch.optim.AdamW(params, lr=0.1, weight_decay=0.0)
        wrapped = DistributedOptimizer.__new__(DistributedOptimizer)
        wrapped.optimizer = native
        wrapped.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        wrapped.config = SimpleNamespace(fp16=False)
        wrapped.grad_scaler = None
        return params, wrapped

    def step(params, optimizer):
        for i, param in enumerate(params):
            param.grad = torch.full_like(param, (i + 1) * 0.25)
        optimizer.optimizer.step()

    params, optimizer = make_optimizer()
    restored_params, restored = make_optimizer()
    step(params, optimizer)
    step(restored_params, restored)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(params, optimizer)
    step(restored_params, restored)
    for actual, expected in zip(restored_params, params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(state['step'].item() == 2 for state in restored.optimizer.state.values())
