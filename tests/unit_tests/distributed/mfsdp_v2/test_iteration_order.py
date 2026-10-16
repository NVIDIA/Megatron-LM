# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration demand recording, strict replay, and weak module ownership."""

import gc
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.iteration import (
    IterationOrder,
    iteration,
)


class _Unit(nn.Module):
    def __init__(self):
        super().__init__()
        self._unshard_event = None
        self.num_parameter_elements = 1
        self._unshard_parameter_groups = Mock()


def test_record_and_replay_mixed_occurrences():
    """Record every demand, including a repeated materialized owner, across phases."""
    a, b = _Unit(), _Unit()
    order = IterationOrder()
    events = [
        (a, "forward"),
        (a, "forward"),
        (b, "forward"),
        (b, "backward"),
        (a, "forward"),
        (a, "backward"),
    ]
    for replay in (False, True, True):
        with order.iteration("train"):
            for module, phase in events:
                order.advance(module, phase)
                # Demand always materializes the current owner before prefetch.
                module._unshard_event = object()
                before = (
                    a._unshard_parameter_groups.call_count + b._unshard_parameter_groups.call_count
                )
                order.prefetch(None)
                after = (
                    a._unshard_parameter_groups.call_count + b._unshard_parameter_groups.call_count
                )
                assert after - before <= int(replay)
                module._unshard_event = None
        assert not order.active
    assert a._unshard_parameter_groups.called and b._unshard_parameter_groups.called
    with order.iteration("validation"):
        order.advance(a, "forward")
    assert len(order._orders) == 2
    # Zero disables prefetch; a positive budget allows more than one owner.
    with order.iteration("train"):
        order.advance(a, "forward")
        a._unshard_event = object()
        before = b._unshard_parameter_groups.call_count
        order.prefetch(0)
        assert b._unshard_parameter_groups.call_count == before
        order.prefetch(2)
        assert b._unshard_parameter_groups.call_count == before + 1
        for module, phase in events[1:]:
            order.advance(module, phase)


def test_replay_rejects_changed_or_incomplete_iterations():
    """Divergence, missing/extra demand, and failed recordings invalidate the cache."""
    a, b = _Unit(), _Unit()
    order = IterationOrder()
    for changed in ([(b, "forward")], [(a, "backward")], [], [(a, "forward"), (a, "forward")]):
        with order.iteration("train"):
            order.advance(a, "forward")
        with pytest.raises(RuntimeError):
            with order.iteration("train"):
                for module, phase in changed:
                    order.advance(module, phase)
        assert not order.active and "train" not in order._orders
    with pytest.raises(ValueError):
        with order.iteration("train"):
            order.advance(a, "forward")
            raise ValueError("failed compute")
    assert "train" not in order._orders
    with order.iteration("train"):
        with pytest.raises(RuntimeError, match="overlap"):
            with order.iteration("validation"):
                pass
        order.advance(a, "forward")
    with order.iteration("train", rerecord=True):
        order.advance(b, "forward")
    with order.iteration("train"):
        order.advance(b, "forward")


def test_iteration_does_not_retain_modules():
    """Completed sequences retain only weak references and detect a missing owner."""
    context = SimpleNamespace(ensure_finalized=Mock(), iteration_order=IterationOrder())
    a, b = _Unit(), _Unit()
    with iteration(context):
        context.iteration_order.advance(a, "forward")
        context.iteration_order.advance(b, "backward")
    b_ref = ref(b)
    del b
    gc.collect()
    assert b_ref() is None
    with pytest.raises(RuntimeError, match="destroyed"):
        with iteration(context):
            context.iteration_order.advance(a, "forward")
            context.iteration_order.prefetch(None)
    assert not context.iteration_order.active


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_demand_recording(distributed_setup, reentrant):
    """Both checkpoint variants replay recomputation demands and preserve SGD."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = checkpoint(self.fc1, x, use_reentrant=reentrant)
            return checkpoint(self.fc2, x.relu(), use_reentrant=reentrant)

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(1234)
    baseline, model = Model().to(device), Model().to(device)
    model.load_state_dict(baseline.state_dict())
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device) as context:
        fully_shard(model.fc1, mesh=mesh, placements=placements)
        fully_shard(model.fc2, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    ref_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    torch.manual_seed(4321 + distributed_setup.rank)
    x = torch.randn(2, 8, device=device)
    for _ in range(3):
        ref_optimizer.zero_grad()
        expected = baseline(x.detach().requires_grad_()).square().mean()
        expected.backward()
        for parameter in baseline.parameters():
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(distributed_setup.world_size)
        ref_optimizer.step()
        optimizer.zero_grad()
        with iteration(context):
            actual = model(x.detach().requires_grad_()).square().mean()
            actual.backward()
        optimizer.step()
        torch.testing.assert_close(actual, expected)
    events = context.iteration_order._orders["default"]
    assert any(phase == "recompute" for _, phase in events)
    for expected, actual in zip(baseline.parameters(), model.parameters()):
        torch.testing.assert_close(actual.full_tensor(), expected)
