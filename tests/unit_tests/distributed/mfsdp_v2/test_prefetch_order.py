# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for recording MFSDP demand-unshard order."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.indexed_order import IndexedOrder
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import (
    FsdpContext,
    FsdpModule,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import SchedulePolicy


def test_recording_scope():
    """Keep repeated demand calls, exclude recompute, and clean up failed scopes."""
    with patch('torch.cuda.Stream'):
        context = FsdpContext(torch.device('cuda'))
    with pytest.raises(RuntimeError, match='not finalized'):
        with context.record_prefetch_order():
            pass
    context.finalize()
    module = object.__new__(FsdpModule)
    module._context = context
    module._schedule_policy = SchedulePolicy()
    module._unshard_event = object()
    module._nvtx_range = Mock(return_value=nullcontext())
    module._unshard_parameter_groups = Mock()
    module._prefetch_parameter_groups = Mock()
    context.current_stream = Mock(return_value=Mock())

    original_forward, original_backward = context.forward_order, context.backward_order
    with context.record_prefetch_order():
        with pytest.raises(RuntimeError, match='already active'):
            with context.record_prefetch_order():
                pass
        for phase in ('forward', 'forward', 'none', 'backward'):
            module.unshard(prefetch=phase)
    assert list(context.forward_order) == [module, module]
    assert list(context.backward_order) == [module]
    assert context.forward_order is not original_forward
    assert context.backward_order is not original_backward
    assert module._unshard_parameter_groups.call_count == 4
    assert context.current_stream().wait_event.call_count == 4
    module._prefetch_parameter_groups.assert_not_called()

    forward, backward = context.forward_order, context.backward_order
    with pytest.raises(ValueError, match='interrupted'):
        with context.record_prefetch_order():
            module.unshard(prefetch='forward')
            raise ValueError('interrupted')
    with context.record_prefetch_order():
        pass
    assert context.forward_order is forward
    assert context.backward_order is backward
    assert context._recorded_orders is None


@pytest.mark.parametrize('budget', [None, 0, 2])
def test_prefetch_replays_occurrences(budget):
    """A reused module prefetches different successors at each recorded position."""
    modules = []
    for _ in range(3):
        module = object.__new__(FsdpModule)
        parameter = SimpleNamespace(unsharded=torch.empty(1))
        module._parameter_groups = (SimpleNamespace(fsdp_parameters=[parameter]),)
        module._unshard_parameter_groups = Mock()
        modules.append(module)
    first, second, third = modules
    sequence = [first, second, first, third]
    order = IndexedOrder(sequence)
    expected = [[second], [first], [third], []]
    if budget == 0:
        expected = [[], [], [], []]
    elif budget == 2:
        expected = [[second, first], [first, third], [third], []]
    for _ in range(2):
        for module, targets in zip(sequence, expected):
            for target in modules:
                target._unshard_parameter_groups.reset_mock()
            module._prefetch_parameter_groups(order, budget)
            for target in modules:
                assert target._unshard_parameter_groups.call_count == targets.count(target)
    assert list(order) == sequence
    with pytest.raises(RuntimeError, match='diverged'):
        order.next_items(third)
    assert list(order.next_items(first)) == [second, first, third]


def test_static_order_lookup():
    """Construction order still permits demand calls outside the static order."""
    first, second, third = Mock(), Mock(), Mock()
    order = IndexedOrder()
    for module in (first, second, third):
        order.append(module)
    assert list(order.next_items(second)) == [third]
    assert list(order.next_items(first)) == [second, third]
    with pytest.raises(ValueError, match='duplicate'):
        order.append(first)


def test_recording_matches_dense_training(distributed_setup):
    """Observe runtime order across microbatches without changing gradients or updates."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(8, 8, bias=False)
            self.second = nn.Linear(8, 8, bias=False)

        def forward(self, inputs):
            return self.first(self.second(inputs)).relu()

    torch.manual_seed(42)
    dense, model = Model().to(device), Model()
    model.load_state_dict(dense.state_dict())
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(device=device) as context:
        fully_shard(model.first, mesh=mesh, placements=placements)
        fully_shard(model.second, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    num_microbatches = 3
    base_inputs = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / 16
    for iteration in range(3):
        dense_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        recording = context.record_prefetch_order() if iteration == 0 else nullcontext()
        with recording:
            for index in range(num_microbatches):
                inputs = base_inputs + 0.1 * distributed_setup.rank + 0.02 * (iteration + index)
                expected = dense(inputs)
                (expected.square().mean() / num_microbatches).backward()
                with microbatch(context, is_last=index == num_microbatches - 1):
                    actual = model(inputs)
                    (actual.square().mean() / num_microbatches).backward()
                torch.testing.assert_close(actual, expected)
            context.finish_grad_sync()
        assert list(context.forward_order) == [model, model.second, model.first] * num_microbatches
        assert list(context.backward_order) == [model, model.first, model.second] * num_microbatches
        for reference in dense.parameters():
            dist.all_reduce(reference.grad, group=mesh.get_group())
            reference.grad.div_(distributed_setup.world_size)
        for reference, sharded in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(sharded.grad.full_tensor(), reference.grad)
        dense_optimizer.step()
        optimizer.step()
        for reference, sharded in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(sharded.full_tensor(), reference)
        assert all(module._unshard_event is None for module in (model, model.first, model.second))
