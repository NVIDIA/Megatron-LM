# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration capture/replay and checkpoint training correctness."""

from contextlib import contextmanager, nullcontext
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
    iteration,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
    PrefetchMode,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


@contextmanager
def _validation(context):
    previous_mode = context.prefetch_mode
    context.prefetch_mode = PrefetchMode.DEMAND
    try:
        yield
    finally:
        context.prefetch_mode = previous_mode


class _Unit(nn.Module):

    def __init__(self, context):
        super().__init__()
        self.context = context
        self.num_parameter_elements = 1
        self._unshard_parameter_groups = Mock()
        event = object()
        self._unshard_events = {Phase.FORWARD: event, Phase.BACKWARD: event}
        self.phase = Phase.RESTING
        self._schedule_policy = SimpleNamespace(
            forward_prefetch_size=None, backward_prefetch_size=None
        )

    def _nvtx_range(self, name):
        return nullcontext()

    def unshard(self, phase=None):
        if phase is not None:
            self.phase = Phase.BACKWARD if phase is Phase.RECOMPUTE else phase
            if phase is Phase.RECOMPUTE:
                phase = Phase.FORWARD
        return FsdpModule.unshard(self, phase)

    _prefetch_parameter_groups = FsdpModule._prefetch_parameter_groups


@pytest.fixture
def context():
    class Context(SimpleNamespace):
        iteration_recording = FsdpContext.iteration_recording

    context = Context(
        ensure_finalized=Mock(),
        execution_order=ExecutionOrder(),
        prefetch_mode=PrefetchMode.PRESET,
        iteration_captured=False,
        allgather_stream=Mock(),
        current_stream=Mock(),
    )
    context._advance_execution_order = lambda module, phase: FsdpContext._advance_execution_order(
        context, module, phase
    )
    return context


_MIXED_PHASE_EVENTS = (
    ('a', Phase.FORWARD),
    ('a', Phase.FORWARD),
    ('b', Phase.FORWARD),
    ('b', Phase.BACKWARD),
    ('a', Phase.RECOMPUTE),
    ('a', Phase.BACKWARD),
)


@pytest.mark.parametrize('events', [_MIXED_PHASE_EVENTS], ids=['mixed-phases'])
@pytest.mark.parametrize(
    'budget, expected_prefetch, replay_events, expected_error',
    [
        pytest.param(
            None,
            [
                ('a', Phase.FORWARD),
                ('b', Phase.FORWARD),
                ('b', Phase.BACKWARD),
                ('a', Phase.FORWARD),
                ('a', Phase.BACKWARD),
            ],
            None,
            None,
            id='one-successor',
        ),
        pytest.param(0, [], None, None, id='no-prefetch'),
        pytest.param(
            2,
            [
                ('a', Phase.FORWARD),
                ('b', Phase.FORWARD),
                ('b', Phase.FORWARD),
                ('b', Phase.BACKWARD),
                ('b', Phase.BACKWARD),
                ('a', Phase.FORWARD),
                ('a', Phase.FORWARD),
                ('a', Phase.BACKWARD),
                ('a', Phase.BACKWARD),
            ],
            None,
            None,
            id='two-successors',
        ),
        pytest.param(0, [], [('b', Phase.FORWARD)], 'module calls diverged', id='wrong-module'),
        pytest.param(0, [], [('a', Phase.BACKWARD)], 'module phase diverged', id='wrong-phase'),
        pytest.param(0, [], _MIXED_PHASE_EVENTS[:-1], 'ended before consuming', id='missing-call'),
        pytest.param(
            0, [], _MIXED_PHASE_EVENTS + (('a', Phase.FORWARD),), 'exceeded', id='extra-call'
        ),
    ],
)
def test_capture_replay_and_validation(
    context, events, budget, expected_prefetch, replay_events, expected_error
):
    """Repeated mixed-phase calls replay with prefetch; validation preserves the trace."""
    modules = {name: _Unit(context) for name in ('a', 'b')}
    gathers = Mock()
    gathers.attach_mock(modules['a']._unshard_parameter_groups, 'a')
    gathers.attach_mock(modules['b']._unshard_parameter_groups, 'b')
    gathers.attach_mock(context.current_stream.return_value.wait_event, 'wait')
    for module in modules.values():
        module._schedule_policy.forward_prefetch_size = budget
        module._schedule_policy.backward_prefetch_size = budget

    def run(sequence):
        prefetched = []
        with iteration(context):
            assert context.prefetch_mode is PrefetchMode.RECORD_REPLAY
            order = context.execution_order
            # Manual inspection gathers without recording or consuming an occurrence.
            before = list(order), order.position
            modules['a'].unshard()
            assert (list(order), order.position) == before
            for name, phase in sequence:
                before = len(gathers.mock_calls)
                modules[name].unshard(phase)
                calls = gathers.mock_calls[before:]
                physical_phase = Phase.BACKWARD if phase is Phase.BACKWARD else Phase.FORWARD
                assert calls[0][0] == name
                assert calls[0].args == (physical_phase,)
                assert calls[1][0] == 'wait'
                prefetched.extend((call[0], call.args[0]) for call in calls[2:])
        return prefetched

    def validate():
        order = context.execution_order
        before = list(order), order.position
        with _validation(context):
            start = len(gathers.mock_calls)
            modules['b'].unshard(Phase.FORWARD)
            assert [call[0] for call in gathers.mock_calls[start:]] == ['b', 'wait']
            assert context.execution_order is order
            assert (list(order), order.position) == before

    validate()
    assert not context.iteration_captured
    for replay in (False, True, True):
        assert run(events) == (expected_prefetch if replay else [])
        assert context.iteration_captured
        assert context.prefetch_mode is PrefetchMode.DEMAND
        assert list(context.execution_order) == [
            (ref(modules[name]), phase) for name, phase in events
        ]
        assert context.execution_order.position == (len(events) if replay else 0)
        validate()
        assert context.prefetch_mode is PrefetchMode.DEMAND

    if expected_error is not None:
        with pytest.raises(RuntimeError, match=expected_error):
            run(replay_events)
        assert not context.iteration_captured
        assert context.prefetch_mode is PrefetchMode.DEMAND


@pytest.mark.parametrize('reentrant', [False, True])
@pytest.mark.parametrize('record_iteration', [False, True], ids=['preset', 'recorded'])
def test_checkpoint_demand_recording(distributed_setup, reentrant, record_iteration):
    """Preset and recorded prefetch preserve SGD with both checkpoint variants."""

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = checkpoint(self.fc1, x, use_reentrant=reentrant)
            if not record_iteration and isinstance(self.fc2, FsdpModule):
                assert Phase.FORWARD in self.fc2._unshard_events
            return checkpoint(self.fc2, x.relu(), use_reentrant=reentrant)

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(1234)
    baseline, model = (Model().to(device), Model().to(device))
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

    def step(model, forward_only=False):
        if forward_only:
            return model(x)
        loss = model(x.detach().requires_grad_()).square().mean()
        loss.backward()
        return loss

    def run(model, forward_only=False):
        scope = (
            (_validation(context) if forward_only else iteration(context))
            if record_iteration
            else nullcontext()
        )
        with scope:
            return step(model, forward_only=forward_only)

    for _ in range(3):
        ref_optimizer.zero_grad()
        expected = baseline(x.detach().requires_grad_()).square().mean()
        expected.backward()
        for parameter in baseline.parameters():
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(distributed_setup.world_size)
        ref_optimizer.step()
        optimizer.zero_grad()
        actual = run(model)
        optimizer.step()
        torch.testing.assert_close(actual, expected)
    for _ in range(2):
        with torch.no_grad():
            torch.testing.assert_close(run(model, forward_only=True), baseline(x))
    if record_iteration:
        demands = context.execution_order
        assert Phase.RECOMPUTE in [phase for _, phase in demands]
    else:
        assert not context.iteration_captured
    with _validation(context):
        for unit in (model, model.fc1, model.fc2):
            unit.unshard()
        try:
            for expected, actual in zip(baseline.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected)
        finally:
            for unit in (model.fc2, model.fc1, model):
                unit.reshard()
