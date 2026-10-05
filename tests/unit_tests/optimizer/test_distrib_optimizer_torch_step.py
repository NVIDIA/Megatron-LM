# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import megatron.core.optimizer.distrib_optimizer as distrib_optimizer
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
from megatron.core.optimizer.distrib_optimizer import Range


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


@pytest.fixture
def checkpoint_process_group():
    """CPU checkpoint I/O also runs outside the GPU test container."""
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(backend='gloo')
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


def _bucket_checkpoint_wrapper(optimizer, group):
    wrapper = _checkpoint_wrapper(optimizer)
    wrapper.config = SimpleNamespace(
        fp16=False, use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
    )
    params = optimizer.param_groups[0]['params']
    wrapper.model_param_group_index_map = {param: (0, i) for i, param in enumerate(params)}
    numel = sum(param.numel() for param in params) * group.size()
    wrapper.gbuf_ranges = [
        {
            torch.float32: [
                {
                    'param_map': {
                        param: {'gbuf_local': Range(i * 2, (i + 1) * 2)}
                        for i, param in enumerate(params)
                    }
                }
            ]
        }
    ]
    wrapper.buffers = [
        SimpleNamespace(
            buckets=[SimpleNamespace(numel_unpadded=numel, grad_data=torch.empty(numel))]
        )
    ]
    wrapper.per_bucket_numel = [numel]
    wrapper.per_bucket_numel_unpadded = [numel]
    wrapper.data_parallel_group = group
    wrapper.data_parallel_group_idx = 0
    wrapper.distributed_optimizer_instance_id = 0
    wrapper._checkpoint_version_for_load = None
    return wrapper


@pytest.mark.parametrize('param_count', [1, 2, 4])
@pytest.mark.parametrize('foreach', [False, True])
def test_torch_adam_dp_reshardable_round_trip(
    monkeypatch, checkpoint_process_group, param_count, foreach
):
    """Checkpoint I/O retains the saved step rather than the local load-template step."""
    monkeypatch.setattr(distrib_optimizer, 'HAVE_APEX_OR_TE', False)
    group = checkpoint_process_group
    params = [torch.nn.Parameter(torch.tensor([0.5 + i, -0.25 - i])) for i in range(param_count)]
    optimizer = torch.optim.Adam(params, lr=0.03, foreach=foreach)
    for iteration in range(1, 13):
        _set_gradients(params, iteration)
        optimizer.step()
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    reference = torch.optim.Adam(reference_params, lr=0.03, foreach=foreach)
    reference.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(directory, src=0)
    metadata = {'distrib_optim_sharding_type': 'dp_reshardable', 'checkpoint_version': 3.1}
    try:
        save(
            _bucket_checkpoint_wrapper(optimizer, group).sharded_state_dict({}, metadata=metadata),
            directory[0],
        )
        resumed_params = [torch.nn.Parameter(torch.zeros_like(param)) for param in params]
        resumed = torch.optim.Adam(resumed_params, lr=0.03, foreach=foreach)
        _set_gradients(resumed_params, 1)
        resumed.step()
        wrapper = _bucket_checkpoint_wrapper(resumed, group)
        template = wrapper.sharded_state_dict({}, is_loading=True, metadata=metadata)
        template_step = template['param_state'][0][torch.float32][0][0]['step']
        assert isinstance(template_step, LocalNonpersistentObject)
        assert template_step.unwrap().item() == 1
        loaded = load(template, directory[0])
        assert all(group_state['step'] == 12 for group_state in loaded['optimizer']['param_groups'])
        assert loaded['param_state'][0][torch.float32][0][0]['step'].item() == 1
        with torch.no_grad():
            wrapper.load_state_dict(loaded)
        assert all(state['step'].item() == 12 for state in resumed.state.values())
        for actual, expected in zip(resumed_params, reference_params):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in ['step', 'exp_avg', 'exp_avg_sq']:
                torch.testing.assert_close(
                    resumed.state[actual][key], reference.state[expected][key], rtol=0, atol=0
                )
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
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(directory[0])
