"""Core regression coverage for ordinary MFSDP communication replay."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

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
    module = SimpleNamespace(
        _unshard_event=None,
        _schedule_policy=SchedulePolicy(),
        context=SimpleNamespace(current_stream=Mock(return_value=Mock())),
    )

    def gather():
        if module._unshard_event is None:
            module._unshard_event = object()

    def release():
        module._unshard_event = None

    module._unshard_parameter_groups = Mock(side_effect=gather)
    module._reshard_parameter_groups = Mock(side_effect=release)
    return module


@pytest.mark.parametrize("max_reuse_distance", [None, 0, 2])
def test_replay_prefetch_and_retention(max_reuse_distance):
    """Compile occurrence-specific prefetch and retain only bounded same-module reuse."""
    scheduler = TraceAndReplayScheduler(max_reuse_distance=max_reuse_distance)
    repeated, second, third = (_fake_module() for _ in range(3))
    sequence = (repeated, repeated, second, repeated, third)
    for iteration in range(3):
        scheduler.begin_iteration()
        for index, module in enumerate(sequence):
            previous_event = module._unshard_event
            scheduler.unshard(module, prefetch="forward")
            module.context.current_stream().wait_event.assert_called_with(module._unshard_event)
            if iteration > 0 and index == 0:
                assert second._unshard_event is not None
            if iteration > 0 and index == 3:
                assert third._unshard_event is not None
            if previous_event is not None:
                assert module._unshard_event is previous_event
            scheduler.reshard(module)
            if module is repeated:
                retained = (
                    iteration > 0
                    and max_reuse_distance is not None
                    and (index == 0 or (index == 1 and max_reuse_distance >= 2))
                )
                assert (module._unshard_event is not None) == retained
        scheduler.end_iteration()
        assert all(module._unshard_event is None for module in (repeated, second, third))
        assert len(scheduler._plan) == 2 * len(sequence)
        assert scheduler._actions[0].prefetch_target is second
        assert scheduler._actions[6].prefetch_target is third
        if iteration == 0:
            scheduler._compile_actions = Mock(side_effect=AssertionError("recompiled replay"))
    scheduler._compile_actions.assert_not_called()


def test_replay_matches_dense_training(distributed_setup):
    """Match rank-averaged gradients and SGD updates across microbatches and replay."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(42)
    dense = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 8, bias=False))
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 8, bias=False))
    model.load_state_dict(dense.state_dict())
    dense = dense.to(device)
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(device=device, use_trace_replay=True, max_reuse_distance=2) as context:
        fully_shard(model[0], mesh=mesh, placements=placements)
        fully_shard(model[2], mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
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
                expected = dense(inputs)
                (expected.square().mean() / num_microbatches).backward()
                with microbatch(context, is_last=index == num_microbatches - 1):
                    actual = model(inputs)
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
        assert all(module._unshard_event is None for module in (model, model[0], model[2]))


@pytest.mark.parametrize("failure", ["mismatch", "unclosed_trace"])
def test_replay_fails_fast(failure):
    """Reject an unclosed trace or unexpected replay before executing new work."""
    scheduler = TraceAndReplayScheduler()
    repeated, prefetched, unexpected = (_fake_module() for _ in range(3))
    scheduler.begin_iteration()
    if failure == "unclosed_trace":
        scheduler.unshard(repeated, prefetch="forward")
        with pytest.raises(RuntimeError, match="end with reshard"):
            scheduler.end_iteration()
        return
    for module in (repeated, repeated, prefetched):
        scheduler.unshard(module, prefetch="forward")
        scheduler.reshard(module)
    scheduler.end_iteration()
    scheduler.begin_iteration()
    scheduler.unshard(repeated, prefetch="forward")
    scheduler.reshard(repeated)
    assert repeated._unshard_event is not None
    assert prefetched._unshard_event is not None
    with pytest.raises(RuntimeError, match="diverged"):
        scheduler.unshard(unexpected, prefetch="forward")
    unexpected._unshard_parameter_groups.assert_not_called()
    with pytest.raises(RuntimeError, match="already active"):
        scheduler.begin_iteration()
