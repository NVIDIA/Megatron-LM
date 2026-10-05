# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from types import SimpleNamespace

import pytest
import torch

import megatron.core.optimizer.distrib_optimizer as distrib_optimizer


def _checkpoint_wrapper(optimizer):
    """Use the real checkpoint methods with already allocated CPU Adam state."""
    wrapper = object.__new__(distrib_optimizer.DistributedOptimizer)
    wrapper.optimizer = optimizer
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    wrapper.config = SimpleNamespace(fp16=False)
    wrapper.grad_scaler = None
    return wrapper


def _set_gradients(params, iteration):
    for index, param in enumerate(params):
        param.grad = torch.tensor([0.1 * (index + 1) + 0.005 * iteration, -0.2 - 0.007 * iteration])


@pytest.mark.parametrize('param_count', [1, 2, 4])
@pytest.mark.parametrize('foreach', [False, True])
def test_torch_adam_checkpoint_next_update(monkeypatch, param_count, foreach):
    """Restored counters advance once per update and retain Adam bias correction."""
    # Exercise the native Adam fallback even when the CI image has TE/Apex installed.
    monkeypatch.setattr(distrib_optimizer, 'HAVE_APEX_OR_TE', False)
    params = [
        torch.nn.Parameter(torch.tensor([0.5 + index, -0.25 - index]))
        for index in range(param_count)
    ]
    optimizer = torch.optim.Adam(params, lr=0.03, foreach=foreach)
    for iteration in range(1, 13):
        _set_gradients(params, iteration)
        optimizer.step()
    full_state = copy.deepcopy(optimizer.state_dict())
    checkpoint = copy.deepcopy(_checkpoint_wrapper(optimizer).state_dict())

    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    resumed_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    reference = torch.optim.Adam(reference_params, lr=0.03, foreach=foreach)
    resumed = torch.optim.Adam(resumed_params, lr=0.03, foreach=foreach)
    reference.load_state_dict(copy.deepcopy(full_state))
    # Parameter moments are loaded separately from DistributedOptimizer's common metadata.
    # Preallocate them here so this regression does not require CUDA buffer allocation.
    resumed.load_state_dict(copy.deepcopy(full_state))
    _checkpoint_wrapper(resumed).load_state_dict(checkpoint)

    assert all(state['step'].item() == 12 for state in resumed.state.values())
    _set_gradients(reference_params, 13)
    _set_gradients(resumed_params, 13)
    reference.step()
    resumed.step()

    assert all(state['step'].item() == 13 for state in resumed.state.values())
    for actual, expected in zip(resumed_params, reference_params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            torch.testing.assert_close(
                resumed.state[actual][key], reference.state[expected][key], rtol=0, atol=0
            )
