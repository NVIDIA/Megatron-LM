# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Single-iteration capture, scheduler replay, and weak module ownership."""

import gc
from contextlib import nullcontext
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
    IterationOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.common import mfsdp_iteration
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    _make_unshard_backward_hook,
    _make_unshard_forward_hook,
)
from megatron.core.models.common.mfsdp_iteration import with_mfsdp_iteration


def _demand(context, module, phase, budget=0):
    order = FsdpContext._advance_order(context, module, phase)
    if order is not None:
        FsdpModule._prefetch_parameter_groups(module, order, budget)


class _Unit(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.config = SimpleNamespace(
            cuda_graph_impl="none", virtual_pipeline_model_parallel_size=2
        )
        self.num_parameter_elements = 1
        self._unshard_parameter_groups = Mock()
        self._unshard_event = object()
        self._schedule_policy = SimpleNamespace(
            forward_prefetch_size=None, backward_prefetch_size=None
        )

    def _nvtx_range(self, name):
        return nullcontext()

    unshard = FsdpModule.unshard
    _prefetch_parameter_groups = FsdpModule._prefetch_parameter_groups


@pytest.fixture
def context(monkeypatch):
    # Exercise the production schedule wrapper without allocating CUDA resources.
    monkeypatch.setattr(mfsdp_iteration, "FsdpModule", _Unit)
    context = Mock(
        ensure_finalized=Mock(),
        iteration_order=IterationOrder(),
        allgather_stream=Mock(),
        current_stream=Mock(),
    )
    context._advance_order = lambda module, phase: FsdpContext._advance_order(
        context, module, phase
    )
    return context


def test_capture_replay_and_validation(context):
    """One training order survives validation; replay resets its execution cursor."""
    a, b = _Unit(context), _Unit(context)
    events = [
        (a, "forward"),
        (a, "forward"),
        (b, "forward"),
        (b, "backward"),
        (a, "recompute"),
        (a, "backward"),
    ]
    gathers = Mock()
    gathers.attach_mock(a._unshard_parameter_groups, "a")
    gathers.attach_mock(b._unshard_parameter_groups, "b")
    gathers.attach_mock(context.current_stream.return_value.wait_event, "wait")
    prefetched = []

    @with_mfsdp_iteration
    def run(model, forward_only=False, budget=None):
        order = context.iteration_order
        if forward_only:
            assert order.mode == "demand"
            before = b._unshard_parameter_groups.call_count
            FsdpModule.unshard(b, "forward")
            assert b._unshard_parameter_groups.call_count == before + 1
            return
        assert order.mode == "record_replay"
        for module, phase in events:
            module._schedule_policy.forward_prefetch_size = budget
            module._schedule_policy.backward_prefetch_size = budget
            before = len(gathers.mock_calls)
            FsdpModule.unshard(module, phase)
            calls = gathers.mock_calls[before:]
            assert calls[0][0] == ("a" if module is a else "b")
            assert calls[1][0] == "wait"
            prefetched.extend(call[0] for call in calls[2:])

    run(a, forward_only=True)
    assert not context.iteration_order.captured
    nested = with_mfsdp_iteration(run)
    for schedule, replay in ((run, False), (nested, True), (run, True)):
        gathers.reset_mock()
        prefetched.clear()
        schedule([a, b])
        assert prefetched == (["a", "b", "b", "a", "a"] if replay else [])
        order = context.iteration_order
        assert order.captured and not order.active
        assert order.position == len(events)
        _, phase = order[4]
        assert phase == "recompute"
        run([a, b], forward_only=True)
        assert context.iteration_order is order and order.position == len(events)
        assert order.mode == "record_replay"
        # The construction order was replaced; out-of-scope calls must not
        # consume the saved training trace (e.g. ordinary validation).
        before = len(gathers.mock_calls)
        FsdpModule.unshard(b, "forward")
        assert [call[0] for call in gathers.mock_calls[before:]] == ["b", "wait"]
        assert order.position == len(events)
    prefetched.clear()
    run(a, budget=0)
    assert not prefetched
    run(a, budget=2)
    # Each demand scans its own window, including overlapping occurrences.
    assert prefetched == ["a", "b", "b", "b", "b", "a", "a", "a", "a"]


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine", "full_iteration"])
def test_cuda_graph_demands_preserve_order_outside_scheduler(context, cuda_graph_impl):
    """Graph warmups may revisit units outside the scheduler's iteration boundary."""
    a, b = _Unit(context), _Unit(context)
    for module in (a, b):
        module.config.cuda_graph_impl = cuda_graph_impl
    order = context.iteration_order
    for module, phase in ((a, "forward"), (b, "forward"), (b, "backward"), (a, "backward")):
        order.append((ref(module), phase))
    order.backward_start = 2
    original = list(order), order.position
    events = [(a, "forward")] * 2 + [(b, "forward")] * 2
    events += [(b, "backward")] * 2 + [(a, "backward")] * 2
    gathers = Mock()
    gathers.attach_mock(a._unshard_parameter_groups, "a")
    gathers.attach_mock(b._unshard_parameter_groups, "b")

    @with_mfsdp_iteration
    def run(model):
        for module, phase in events:
            module.unshard(phase)
        assert order.mode == "demand" and not order.active

    for _ in range(2):
        gathers.reset_mock()
        run([a, b])
        assert [call[0] for call in gathers.mock_calls] == [
            "a" if module is a else "b" for module, _ in events
        ]
        # Local/TE graph creation also calls units after the scheduler returns.
        b.unshard("forward")
        b.unshard("backward")
        assert (list(order), order.position) == original
        assert not order.captured and not order.active

    # Combined hooks retain their root stream dependency without an active scope.
    a.is_root = Mock(return_value=True)
    context.allgather_stream.wait_stream.reset_mock()
    _make_unshard_forward_hook(a)(None, (), {})
    context.allgather_stream.wait_stream.assert_called_once_with(context.current_stream())


def test_nested_graph_scheduler_preserves_explicit_iteration(context):
    """A nested wrapper must not change the strategy of an already active scope."""
    module = _Unit(context)
    module.config.cuda_graph_impl = "full_iteration"

    @with_mfsdp_iteration
    def run(model):
        assert context.iteration_order.mode == "record_replay"
        assert context.iteration_order.active
        model.unshard("forward")

    with iteration(context):
        run(module)
    assert context.iteration_order.captured


def test_static_recompute_and_manual_unshard_preserve_cursor(context):
    """Only execution demands consume static order; its phase selects the budget."""
    a, b = _Unit(context), _Unit(context)
    order = context.iteration_order
    for module, phase in ((a, "forward"), (b, "forward"), (b, "backward"), (a, "backward")):
        order.append((ref(module), phase))
    order.backward_start = 2

    a.unshard()
    FsdpModule.unshard(a, "recompute")
    assert order.position == 0
    assert a._unshard_parameter_groups.call_count == 2
    b._unshard_parameter_groups.assert_not_called()

    FsdpModule.unshard(a, "forward")
    assert order.position == 1
    b._unshard_parameter_groups.assert_called_once_with()

    # Backward uses its own budget, independently of the forward setting.
    order.reset_cursor(order.backward_start)
    b._schedule_policy.backward_prefetch_size = 0
    a._unshard_parameter_groups.reset_mock()
    FsdpModule.unshard(b, "backward")
    assert order.position == 3
    a._unshard_parameter_groups.assert_not_called()


@pytest.mark.parametrize("captured", [False, True], ids=["capturing", "replaying"])
def test_manual_unshard_preserves_iteration_order(context, captured):
    """Parameter inspection must neither record demands nor consume the saved trace."""
    a, b = _Unit(context), _Unit(context)
    if captured:
        with iteration(context):
            a.unshard("forward")
            b.unshard("backward")
    with iteration(context):
        order = context.iteration_order
        before = list(order), order.position
        a.unshard()
        assert (list(order), order.position) == before
        a.unshard("forward")
        b.unshard("backward")


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("active", [False, True], ids=["static", "capturing"])
def test_combined_hooks_only_consume_active_order(context, phase, active):
    """Fine-grained repeated owner demands require an iteration scope."""
    owner, other = _Unit(context), _Unit(context)
    owner.is_root = Mock(return_value=True)
    order = context.iteration_order
    if active:
        order.mode = "record_replay"
        order.active = True
    else:
        order.append((ref(owner), phase))
        order.append((ref(other), phase))
    before = list(order), order.position
    if phase == "forward":
        hook = _make_unshard_forward_hook(owner)
        for _ in range(2):
            hook(None, (), {})
    else:
        hook = _make_unshard_backward_hook(owner)
        for _ in range(2):
            hook(None, ())
    assert owner._unshard_parameter_groups.call_count == 2
    if active:
        assert [(module_ref(), event_phase) for module_ref, event_phase in order] == [
            (owner, phase),
            (owner, phase),
        ]
        assert order.position == 2
    else:
        assert (list(order), order.position) == before
        other._unshard_parameter_groups.assert_not_called()


def test_recorded_root_backward_does_not_read_static_order(context, monkeypatch):
    """The first captured backward demand need not have a construction-order entry."""
    order = context.iteration_order
    order.mode = "record_replay"
    order.active = True
    module = Mock(spec=FsdpModule, context=context, phase=FsdpModule.Phase.RESTING)
    module.is_root.return_value = True
    module._nvtx_label = Mock(return_value="backward")
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", Mock())
    FsdpModule.pre_backward(module)
    module.unshard.assert_called_once_with("backward")
    context.register_post_backward_hook.assert_called_once_with()


def test_failed_or_changed_iterations_require_capture(context):
    """Mismatched, incomplete, extra, and failed calls invalidate the single order."""
    a, b = _Unit(context), _Unit(context)
    events = [(a, "forward")]

    @with_mfsdp_iteration
    def run(model, fail=False):
        for module, phase in events:
            _demand(context, module, phase)
        if fail:
            raise ValueError("failed compute")

    for changed in ([(b, "forward")], [(a, "backward")], [], [(a, "forward"), (a, "forward")]):
        events = [(a, "forward")]
        with iteration(context):
            _demand(context, a, "forward")
        events = changed
        with pytest.raises(RuntimeError):
            run(a)
        assert not context.iteration_order.active and not context.iteration_order.captured
    with pytest.raises(ValueError):
        run(a, fail=True)
    assert not context.iteration_order.captured
    with iteration(context):
        with pytest.raises(RuntimeError, match="active"):
            with iteration(context):
                pass
        _demand(context, b, "forward")
    events = [(b, "forward")]
    run(a)
    assert context.iteration_order.captured
    context.allgather_stream.wait_stream.side_effect = ValueError("failed stream sync")
    with pytest.raises(ValueError):
        with iteration(context):
            pass
    assert not context.iteration_order.active and not context.iteration_order.captured


def test_capture_does_not_retain_modules(context):
    """Captured occurrences keep weak references and reject a destroyed owner."""
    a, b = _Unit(context), _Unit(context)
    with iteration(context):
        _demand(context, a, "forward")
        _demand(context, b, "backward")
    b_ref = ref(b)
    del b
    gc.collect()
    assert b_ref() is None

    @with_mfsdp_iteration
    def run(model):
        _demand(context, model, "forward", budget=None)

    with pytest.raises(RuntimeError, match="destroyed"):
        run(a)
    assert not context.iteration_order.active and not context.iteration_order.captured


@pytest.mark.parametrize(
    "mode, captured",
    [("static", False), ("record_replay", False), ("record_replay", True), ("demand", True)],
)
def test_repeated_backward_keeps_static_guard(mode, captured, monkeypatch):
    """Active scopes observe repeated backward calls; static mode keeps its guard."""
    context = Mock(iteration_order=IterationOrder())
    context.iteration_order.mode = mode
    context.iteration_order.captured = captured
    context.iteration_order.active = mode != "static"
    module = Mock(spec=FsdpModule, context=context, phase=FsdpModule.Phase.BACKWARD)
    module.is_root.return_value = False
    push = Mock()
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", push)
    FsdpModule.pre_backward(module)
    if mode == "static":
        module.unshard.assert_not_called()
        context.current_stream.assert_not_called()
    else:
        module.unshard.assert_called_once_with("backward")
    push.assert_not_called()


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("record_iteration", [False, True], ids=["static", "recorded"])
def test_checkpoint_demand_recording(distributed_setup, reentrant, record_iteration, monkeypatch):
    """Static and recorded prefetch preserve SGD with both checkpoint variants."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = checkpoint(self.fc1, x, use_reentrant=reentrant)
            if not record_iteration and isinstance(self.fc2, FsdpModule):
                # The next owner is prefetched before its own demand hook runs,
                # including on the first step, without recording an iteration.
                assert self.fc2._unshard_event is not None
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
    reset = Mock(wraps=context.iteration_order.reset_cursor)
    monkeypatch.setattr(context.iteration_order, "reset_cursor", reset)
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

    run = with_mfsdp_iteration(step) if record_iteration else step
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
    assert reset.call_count == (0 if record_iteration else 8)
    if not record_iteration:
        assert [call.args for call in reset.call_args_list] == [
            (),
            (context.iteration_order.backward_start,),
        ] * 3 + [(), ()]
    if record_iteration:
        demands = context.iteration_order
        assert "recompute" in [phase for _, phase in demands]
    else:
        assert not context.iteration_order.captured
    # Group-packed shards need the FSDP layout to reconstruct parameters;
    # DTensor.full_tensor() assumes ordinary per-tensor chunking instead.
    with iteration(context, forward_only=True):
        for unit in (model, model.fc1, model.fc2):
            unit.unshard()
        try:
            for expected, actual in zip(baseline.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected)
        finally:
            for unit in (model.fc2, model.fc1, model):
                unit.reshard()
