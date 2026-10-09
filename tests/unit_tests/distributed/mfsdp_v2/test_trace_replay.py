"""Regression coverage for opt-in ordinary MFSDP communication replay."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import (
    SchedulePolicy,
    TraceAndReplayScheduler,
)


def _fake_module():
    module = SimpleNamespace(_unshard_event=None, _schedule_policy=SchedulePolicy())

    def gather():
        module._unshard_event = object()

    def release():
        module._unshard_event = None

    module._unshard_parameter_groups = Mock(side_effect=gather)
    module._reshard_parameter_groups = Mock(side_effect=release)
    return module


def _consume(scheduler, module, phase="forward"):
    scheduler.record_unshard(module, phase)
    module._unshard_parameter_groups()
    scheduler.prefetch(module, phase)


def _release(scheduler, module):
    if not scheduler.record_reshard(module):
        module._reshard_parameter_groups()


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_repeated_module_retains_only_on_replay(phase):
    scheduler = TraceAndReplayScheduler()
    module = _fake_module()
    for iteration in range(2):
        scheduler.begin_iteration()
        _consume(scheduler, module, phase)
        retained = scheduler.record_reshard(module)
        assert retained == (iteration == 1)
        if not retained:
            module._reshard_parameter_groups()
        _consume(scheduler, module, phase)
        _release(scheduler, module)
        scheduler.end_iteration()
        assert module._unshard_event is None


def test_divergence_releases_prefetch_and_retraces_full_iteration():
    scheduler = TraceAndReplayScheduler()
    first, second, alternate = (_fake_module() for _ in range(3))
    scheduler.begin_iteration()
    for module in (first, second):
        _consume(scheduler, module)
        _release(scheduler, module)
    scheduler.end_iteration()
    scheduler.begin_iteration()
    _consume(scheduler, first)
    assert second._unshard_event is not None
    _release(scheduler, first)
    _consume(scheduler, alternate)
    assert second._unshard_event is None
    _release(scheduler, alternate)
    scheduler.end_iteration()
    assert scheduler._plan == []
    scheduler.begin_iteration()
    _consume(scheduler, alternate)
    _release(scheduler, alternate)
    scheduler.end_iteration()
    assert [event.module for event in scheduler._plan] == [alternate, alternate]


@pytest.mark.parametrize("abort", [False, True])
def test_incomplete_iteration_cleans_materializations(abort):
    scheduler = TraceAndReplayScheduler()
    first, second = _fake_module(), _fake_module()
    scheduler.begin_iteration()
    for module in (first, second):
        _consume(scheduler, module)
        _release(scheduler, module)
    scheduler.end_iteration()
    scheduler.begin_iteration()
    _consume(scheduler, first)
    assert second._unshard_event is not None
    if abort:
        scheduler.abort_iteration()
    else:
        scheduler.end_iteration()
    assert first._unshard_event is None
    assert second._unshard_event is None
    assert scheduler._plan == []


def test_prefetch_does_not_gather_before_target_release():
    scheduler = TraceAndReplayScheduler()
    first, second = _fake_module(), _fake_module()
    scheduler.begin_iteration()
    _consume(scheduler, first)
    _release(scheduler, second)
    _release(scheduler, first)
    _consume(scheduler, second)
    _release(scheduler, second)
    scheduler.end_iteration()
    scheduler.begin_iteration()
    _consume(scheduler, first)
    assert second._unshard_event is None
    scheduler.abort_iteration()


def test_divergence_releases_retained_module():
    scheduler = TraceAndReplayScheduler()
    module, alternate = _fake_module(), _fake_module()
    scheduler.begin_iteration()
    for _ in range(2):
        _consume(scheduler, module)
        _release(scheduler, module)
    scheduler.end_iteration()
    scheduler.begin_iteration()
    _consume(scheduler, module)
    assert scheduler.record_reshard(module)
    _consume(scheduler, alternate)
    assert module._unshard_event is None
    _release(scheduler, alternate)
    scheduler.end_iteration()
    assert scheduler._plan == []


def test_zero_budget_disables_replay_prefetch():
    scheduler = TraceAndReplayScheduler()
    first, second = _fake_module(), _fake_module()
    first._schedule_policy = SchedulePolicy(forward_prefetch_size=0)
    for _ in range(2):
        scheduler.begin_iteration()
        _consume(scheduler, first)
        assert second._unshard_event is None
        _release(scheduler, first)
        _consume(scheduler, second)
        _release(scheduler, second)
        scheduler.end_iteration()


@pytest.mark.parametrize("forward_budget", [None, 0])
@pytest.mark.parametrize("backward_budget", [None, 0])
def test_none_phase_suppresses_prefetch(forward_budget, backward_budget):
    scheduler = TraceAndReplayScheduler()
    first, second = _fake_module(), _fake_module()
    first._schedule_policy = SchedulePolicy(forward_budget, backward_budget)
    for _ in range(2):
        scheduler.begin_iteration()
        _consume(scheduler, first, "none")
        assert second._unshard_event is None
        _release(scheduler, first)
        _consume(scheduler, second)
        _release(scheduler, second)
        scheduler.end_iteration()


@pytest.mark.parametrize("backward_budget", [None, 0])
def test_backward_prefetch_uses_backward_budget(backward_budget):
    scheduler = TraceAndReplayScheduler()
    first, second = _fake_module(), _fake_module()
    first._schedule_policy = SchedulePolicy(0, backward_budget)
    for iteration in range(2):
        scheduler.begin_iteration()
        _consume(scheduler, first, "backward")
        assert (second._unshard_event is not None) == (iteration == 1 and backward_budget is None)
        _release(scheduler, first)
        _consume(scheduler, second, "backward")
        _release(scheduler, second)
        scheduler.end_iteration()


def test_failed_speculative_gather_is_released_on_abort():
    scheduler = TraceAndReplayScheduler()
    first, second = _fake_module(), _fake_module()
    scheduler.begin_iteration()
    for module in (first, second):
        _consume(scheduler, module)
        _release(scheduler, module)
    scheduler.end_iteration()
    second._reshard_parameter_groups.reset_mock()

    def partial_gather():
        raise RuntimeError("partial allocation")

    second._unshard_parameter_groups.side_effect = partial_gather
    scheduler.begin_iteration()
    with pytest.raises(RuntimeError, match="partial allocation"):
        _consume(scheduler, first)
    scheduler.abort_iteration()
    second._reshard_parameter_groups.assert_called_once()
    assert first._unshard_event is None
    assert second._unshard_event is None
    assert not scheduler._held
    assert not scheduler._plan


def test_iteration_boundaries_are_explicit():
    scheduler = TraceAndReplayScheduler()
    with pytest.raises(RuntimeError, match="begin_iteration"):
        _consume(scheduler, _fake_module())
    with pytest.raises(RuntimeError, match="No FSDP"):
        scheduler.end_iteration()
    scheduler.begin_iteration()
    with pytest.raises(RuntimeError, match="already active"):
        scheduler.begin_iteration()
    scheduler.abort_iteration()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("single_module", [False, True])
def test_ordinary_hooks_match_dense_gradients(
    distributed_setup, enabled, single_module, checkpoint_mode=None
):
    """Average rank-distinct microbatch gradients and match repeated SGD updates."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(42)

    def build_model():
        if single_module:
            return nn.Linear(8, 8, bias=False)
        return nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 8, bias=False))

    dense = build_model()
    model = build_model()
    model.load_state_dict(dense.state_dict())
    dense = dense.to(device)
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(device=device, use_trace_replay=enabled) as context:
        if not single_module:
            fully_shard(model[0], mesh=mesh, placements=placements)
            fully_shard(model[2], mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    assert (context.scheduler is not None) == enabled
    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    num_microbatches = 3
    base_inputs = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / 16
    for iteration in range(3):
        dense_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        with context.iteration():
            for index in range(num_microbatches):
                inputs = base_inputs + 0.1 * distributed_setup.rank + 0.02 * (iteration + index)
                inputs.requires_grad_(checkpoint_mode is not None)
                expected = dense(inputs)
                (expected.square().mean() / num_microbatches).backward()
                with microbatch(context, is_last=index == num_microbatches - 1):
                    actual = (
                        model(inputs)
                        if checkpoint_mode is None
                        else checkpoint(model, inputs, use_reentrant=checkpoint_mode)
                    )
                    (actual.square().mean() / num_microbatches).backward()
                torch.testing.assert_close(actual, expected)
            context.finish_grad_sync()
        for reference in dense.parameters():
            dist.all_reduce(reference.grad, group=mesh.get_group())
            reference.grad.div_(distributed_setup.world_size)
        for reference, sharded in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(sharded.grad.full_tensor(), reference.grad)
        dense_optimizer.step()
        optimizer.step()
        for reference, sharded in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(sharded.full_tensor(), reference)
        modules = (model,) if single_module else (model, model[0], model[2])
        assert all(module._unshard_event is None for module in modules)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpointed_ordinary_hooks_match_dense_gradients(distributed_setup, use_reentrant):
    test_ordinary_hooks_match_dense_gradients(
        distributed_setup, enabled=True, single_module=False, checkpoint_mode=use_reentrant
    )


def test_repeated_ordinary_forward_and_exception_cleanup(distributed_setup):
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, bias=False)
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(device=device, use_trace_replay=True) as context:
        fully_shard(model, mesh=mesh, placements=placements)
    inputs = torch.ones(2, 8, device=device)
    for _ in range(3):
        with context.iteration(), torch.no_grad():
            first = model(inputs)
            second = model(inputs)
        torch.testing.assert_close(first, second)
        assert model._unshard_event is None
    with pytest.raises(ValueError, match="abort"):
        with context.iteration(), torch.no_grad():
            model(inputs)
            raise ValueError("abort")
    assert model._unshard_event is None
    assert context.scheduler._plan == []
