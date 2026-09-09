# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    FixedPoolAllocator,
    MaxPoolAllocator,
    ParamAndGradBuffer,
    _ManualRegistrationState,
)
from megatron.training import training

pytestmark = pytest.mark.internal


def _group(name):
    return SimpleNamespace(group_desc=name, size=lambda: 2)


def _manual_buffer(pool, groups):
    buffer = ParamAndGradBuffer.__new__(ParamAndGradBuffer)
    buffer.ddp_config = SimpleNamespace(
        nccl_ub=True,
        fsdp_double_buffer=True,
        fsdp_manual_registration=True,
        disable_symmetric_registration=False,
    )
    buffer.nccl_mem_pool = pool
    buffer.ubr_groups = groups
    buffer._manual_registration_state = _ManualRegistrationState.UNREGISTERED
    buffer._registered_ubr_groups = []
    return buffer


def test_manual_registration_uses_each_wrappers_pool(monkeypatch):
    pools = [object(), object()]
    groups = [_group("first"), _group("second")]
    buffers = [_manual_buffer(pool, [group]) for pool, group in zip(pools, groups)]
    calls = []

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        param_and_grad_buffer.nccl_allocator,
        "register_mem_pool",
        lambda pool, group, symmetric: calls.append((pool, group)),
    )

    for buffer in buffers:
        buffer.manual_buffer_registration()

    assert calls == list(zip(pools, groups))
    assert all(buffer.already_registered for buffer in buffers)
    with pytest.raises(AssertionError, match="Mem pool cannot be registered"):
        buffers[0].manual_buffer_registration()


def test_partial_registration_failure_is_not_reported_as_success(monkeypatch):
    groups = [_group("first"), _group("second")]
    buffer = _manual_buffer(object(), groups)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)

    def register_mem_pool(pool, group, symmetric):
        if group is groups[1]:
            raise RuntimeError("injected registration failure")

    monkeypatch.setattr(
        param_and_grad_buffer.nccl_allocator, "register_mem_pool", register_mem_pool
    )

    with pytest.raises(RuntimeError, match="injected registration failure"):
        buffer.manual_buffer_registration()

    assert buffer._manual_registration_state == _ManualRegistrationState.FAILED
    assert buffer._registered_ubr_groups == groups[:1]
    assert not buffer.already_registered


def test_deregistration_is_reversed_and_idempotent(monkeypatch):
    pool = object()
    groups = [_group("first"), _group("second")]
    buffer = _manual_buffer(pool, groups)
    buffer._manual_registration_state = _ManualRegistrationState.REGISTERED
    buffer._registered_ubr_groups = list(groups)
    calls = []

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(param_and_grad_buffer, "NCCL_ALLOCATOR", "MCORE")
    monkeypatch.setattr(
        param_and_grad_buffer.nccl_allocator,
        "deregister_mem_pool",
        lambda call_pool, group: calls.append((call_pool, group)),
    )

    buffer.manual_buffer_deregistration()
    buffer.manual_buffer_deregistration()

    assert calls == [(pool, groups[1]), (pool, groups[0])]
    assert buffer._manual_registration_state == _ManualRegistrationState.DEREGISTERED


def test_training_teardown_deregisters_every_owned_pool(monkeypatch):
    calls = []

    class _FSDP:
        def __init__(self, index, manual_registration=True):
            self.param_and_grad_buffer = SimpleNamespace(
                ddp_config=SimpleNamespace(fsdp_manual_registration=manual_registration),
                nccl_mem_pool=object(),
                manual_buffer_deregistration=lambda: calls.append(("fsdp", index)),
            )

    class _DDP:
        def __init__(self, buffers, expert_parallel_buffers):
            self.buffers = buffers
            self.expert_parallel_buffers = expert_parallel_buffers

    fsdp_modules = [_FSDP(0), _FSDP(1), _FSDP(2, manual_registration=False)]
    dense_pool, expert_pool = object(), object()
    dense_group, expert_group = object(), object()
    ddp_module = _DDP(
        buffers=[
            SimpleNamespace(nccl_mem_pool=dense_pool, data_parallel_group=dense_group),
            SimpleNamespace(nccl_mem_pool=None, data_parallel_group=object()),
        ],
        expert_parallel_buffers=[
            SimpleNamespace(nccl_mem_pool=expert_pool, data_parallel_group=expert_group)
        ],
    )

    monkeypatch.setattr(training, "DDP", _DDP)
    monkeypatch.setattr(training, "FullyShardedDataParallelV1", _FSDP)
    monkeypatch.setattr(training, "FullyShardedDataParallelV2", _FSDP)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: calls.append(("barrier", None)))
    monkeypatch.setattr(
        training.nccl_allocator,
        "deregister_mem_pool",
        lambda pool, group: calls.append(("ddp", pool, group)),
    )

    training._deregister_nccl_memory_pools(
        [*fsdp_modules, ddp_module],
        SimpleNamespace(gtp_remat_nccl_ub=False, gtp_expert_remat_nccl_ub=False),
    )

    assert calls == [
        ("barrier", None),
        ("fsdp", 0),
        ("fsdp", 1),
        ("ddp", dense_pool, dense_group),
        ("ddp", expert_pool, expert_group),
    ]


@pytest.mark.parametrize("allocator_type", [FixedPoolAllocator, MaxPoolAllocator])
def test_persistent_allocator_arenas_do_not_alias(allocator_type):
    parameter_group = SimpleNamespace(
        fsdp_unit_id=0, dtype=torch.float32, params=[torch.nn.Parameter(torch.empty(8))]
    )
    allocators = [
        allocator_type(
            name="wrapper-local-name",
            fsdp_param_groups=[parameter_group],
            size=1,
            memory_buffer=param_and_grad_buffer.GlobalMemoryBuffer(),
        )
        for _ in range(2)
    ]

    buckets = [
        allocator.allocate(bucket_id=0, size=8, dtype=torch.float32, device=torch.device("cuda"))
        for allocator in allocators
    ]

    assert buckets[0].data.data_ptr() != buckets[1].data.data_ptr()
