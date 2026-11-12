# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP runtime contexts.

Test prefetch-order construction here; test communication/computation overlap
in ``test_overlap.py``.
"""

from unittest.mock import Mock
from weakref import ref

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    SchedulePolicy,
    finalize_gradients,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


class NestedModel(nn.Module):
    """Model with direct and child-owned parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(4))
        self.inner = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested model."""
        return self.inner(x) + self.bias


class MultiChildModel(nn.Module):
    """Model with direct parameters and multiple child FsdpModules."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class BranchModel(nn.Module):
    """Nested branch with its own child FsdpModule."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.inner = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested branch."""
        return torch.relu(self.inner(x) + self.bias)


class NestedSiblingModel(nn.Module):
    """Model with a nested left subtree and a right sibling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.left = BranchModel(dim)
        self.right = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested subtree before the right sibling."""
        return self.right(self.left(x) + self.bias)


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def test_recorded_execution_order(distributed_setup):
    """Recorded orders follow execution when submodules are registered in reverse."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.second = nn.Linear(4, 4, bias=False)
            self.first = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.second(self.first(x))

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = Model().to(device)
    unsharded_weights = {layer: layer.weight for layer in model.children()}
    with fully_shard_context(
        FsdpContext(device=device, prefetch_order=PrefetchOrder.RECORDED)
    ) as context:
        for layer in model.children():
            fully_shard(layer, mesh, _default_placements())
        fully_shard(model, mesh, _default_placements())

    inputs = torch.ones(2, 4, device=device)
    assert context.is_recording_execution_order
    model(inputs).sum().backward()
    assert not context.is_recording_execution_order
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD) for module in [model, model.first, model.second]
    ] + [ExecutionEvent(module, Phase.BACKWARD) for module in [model, model.second, model.first]]

    def check_prefetched(layer, _args):
        storage_bytes = unsharded_weights[layer].untyped_storage().nbytes()
        assert storage_bytes > 0

    for layer in model.children():
        # Check before FSDP's own hooks, so on-demand unsharding cannot satisfy the assertion.
        layer.register_forward_pre_hook(check_prefetched, prepend=True)
        layer.register_full_backward_pre_hook(check_prefetched, prepend=True)
    model.zero_grad(set_to_none=True)
    model(inputs).sum().backward()


def test_registration_order_rejects_reordered_execution(distributed_setup):
    """Execution must follow registration order when recording is disabled."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.first(self.second(x))

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = Model().to(device)
    with fully_shard_context(FsdpContext(device=device)):
        for layer in model.children():
            fully_shard(layer, mesh, _default_placements())
        fully_shard(model, mesh, _default_placements())

    with pytest.raises(RuntimeError, match="diverged from the expected execution order"):
        model(torch.ones(2, 4, device=device))


@pytest.mark.parametrize("overlap_allgather_reduction", [False, True])
def test_context_defaults_to_current_cuda_device(distributed_setup, overlap_allgather_reduction):
    """The policy controls stream sharing without changing the default device."""
    context = FsdpContext(
        runtime_policy=RuntimePolicy(overlap_allgather_reduction=overlap_allgather_reduction)
    )
    assert context.allgather_stream.device == torch.device("cuda", torch.cuda.current_device())
    assert (
        context.allgather_stream is not context.inner_reduction_stream
    ) == overlap_allgather_reduction


def test_context_rejects_non_cuda_device(distributed_setup):
    """The context constructor should reject non-CUDA devices."""
    with pytest.raises(ValueError, match="CUDA device"):
        FsdpContext(device=torch.device("cpu"))


def test_child_then_parent_share_one_context(distributed_setup):
    """Modules constructed together should eagerly share one context."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    context = FsdpContext(device=device)
    with fully_shard_context(context):
        fully_shard(model.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())
        assert model.context is context
        assert model.inner.context is context

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.inner.context is model.context
    assert model.is_root()
    assert not model.inner.is_root()


def test_context_does_not_keep_modules_alive(distributed_setup):
    """Keeping a finalized context alive must not retain its roots or descendants."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel().to(device)

    with fully_shard_context(FsdpContext(device=device)) as context:
        fully_shard(model.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    model_ref = ref(model)
    child_ref = ref(model.inner)
    del model

    assert model_ref() is None
    assert child_ref() is None
    assert list(context.execution_order) == []


def test_two_child_subtrees_then_parent_share_one_context(distributed_setup):
    """One construction scope should assign one context across child subtrees."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.layers[0], mesh=mesh, placements=_default_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.layers[0].context is model.context
    assert model.layers[1].context is model.context


def test_sibling_roots_share_context_and_cross_root_orders(distributed_setup):
    """Independent roots should share streams and follow construction order."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.layers[0], mesh=mesh, placements=_default_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_default_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.layers[0].context
    assert model.layers[1].context is context
    assert model.layers[0].is_root()
    assert model.layers[1].is_root()
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD) for module in [model.layers[0], model.layers[1]]
    ] + [ExecutionEvent(module, Phase.BACKWARD) for module in [model.layers[1], model.layers[0]]]


@pytest.mark.parametrize("caller_managed_grad_sync", [False, True])
def test_sibling_roots_finish_grad_sync_once_after_last_microbatch(
    distributed_setup, monkeypatch, caller_managed_grad_sync
):
    """Sibling roots should share one final wait across accumulated microbatches."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(
        FsdpContext(
            device=device,
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=caller_managed_grad_sync),
        )
    ) as context:
        fully_shard(model.layers[0], mesh=mesh, placements=_default_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_default_placements())

    finish_grad_sync = Mock(wraps=context.finish_grad_sync)
    monkeypatch.setattr(context, "finish_grad_sync", finish_grad_sync)

    with finalize_gradients(context, enabled=False):
        model(torch.ones(2, 4, device=device)).sum().backward()
    finish_grad_sync.assert_not_called()

    model(torch.ones(2, 4, device=device)).sum().backward()

    if caller_managed_grad_sync:
        finish_grad_sync.assert_not_called()
        context.finish_grad_sync()
    finish_grad_sync.assert_called_once_with()


def test_nested_prefetch_orders_use_dfs(distributed_setup):
    """Nested FsdpModules should use DFS orders for one-step prefetch."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.left.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model.left, mesh=mesh, placements=_default_placements())
        fully_shard(model.right, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.context
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD)
        for module in [model, model.left, model.left.inner, model.right]
    ] + [
        ExecutionEvent(module, Phase.BACKWARD)
        for module in [model, model.right, model.left, model.left.inner]
    ]


def test_register_post_backward_hook_handles_parameterless_module(distributed_setup):
    """A parameterless FSDP unit should invoke the external scheduler callback."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Identity().to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(
            model,
            mesh=mesh,
            placements=_default_placements(),
            schedule_policy=SchedulePolicy(register_hooks=False),
        )

    callback_modules = []
    model.register_post_backward_hook(callback_modules.append)
    model(torch.ones(2, 4, device=device, requires_grad=True)).sum().backward()

    assert callback_modules == [model]


def test_nested_and_sibling_roots_use_cross_root_orders(distributed_setup):
    """Context orders should concatenate nested roots at construction boundaries."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.left.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model.left, mesh=mesh, placements=_default_placements())
        fully_shard(model.right, mesh=mesh, placements=_default_placements())

    context = model.left.context
    assert model.left.is_root()
    assert model.right.is_root()
    assert not model.left.inner.is_root()
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD)
        for module in [model.left, model.left.inner, model.right]
    ] + [
        ExecutionEvent(module, Phase.BACKWARD)
        for module in [model.right, model.left, model.left.inner]
    ]


def test_fully_shard_requires_context(distributed_setup):
    """fully_shard should reject construction without an active context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)

    with pytest.raises(RuntimeError, match="inside fully_shard_context"):
        fully_shard(model, mesh=mesh, placements=_default_placements())


def test_forward_requires_finalized_context(distributed_setup):
    """Forward should be unavailable until construction scope exit."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    x = torch.ones(2, 4, device=device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model, mesh=mesh, placements=_default_placements())
        with pytest.raises(RuntimeError, match="Exit fully_shard_context"):
            model(x)

    model(x)


def test_fully_shard_context_rejects_nesting(distributed_setup):
    """A construction scope should reject an ambiguous nested context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model[0], mesh=mesh, placements=_default_placements())
        outer_context = model[0].context
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(FsdpContext(device=device)):
                pass
        fully_shard(model[1], mesh=mesh, placements=_default_placements())

    assert model[0].context is outer_context
    assert model[1].context is outer_context


def test_fully_shard_rejects_child_from_another_context(distributed_setup):
    """A parent cannot join a context different from an FSDP child context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    with fully_shard_context(FsdpContext(device=device)) as first_context:
        fully_shard(model.inner, mesh=mesh, placements=_default_placements())

    with fully_shard_context(FsdpContext(device=device)):
        with pytest.raises(ValueError, match="another fully_shard_context"):
            fully_shard(model, mesh=mesh, placements=_default_placements())

    assert model.inner.context is first_context


def test_finalize_gradients_scopes_context(distributed_setup):
    """Finalization is enabled by default and scoped to the supplied context."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)).to(device)
    with fully_shard_context(FsdpContext(device=device)) as context:
        for layer in model:
            fully_shard(layer, mesh=mesh, placements=_default_placements())

    with finalize_gradients(context, enabled=False):
        assert not context.finalize_gradients
        with finalize_gradients(context):
            assert context.finalize_gradients
        assert not context.finalize_gradients

    assert context.finalize_gradients


@pytest.mark.parametrize("overlap_allgather_reduction", [False, True])
@pytest.mark.parametrize("overlap_outer_communication", [False, True])
def test_communication_stream_assignment(
    distributed_setup, overlap_allgather_reduction, overlap_outer_communication
):
    """All-gather/reduction and inner/outer stream separation are independent."""
    context = FsdpContext(
        runtime_policy=RuntimePolicy(
            overlap_allgather_reduction=overlap_allgather_reduction,
            overlap_outer_communication=overlap_outer_communication,
        )
    )
    assert (
        context.allgather_stream != context.inner_reduction_stream
    ) == overlap_allgather_reduction
    assert (
        context.outer_reduction_stream != context.inner_reduction_stream
    ) == overlap_outer_communication
    if overlap_outer_communication:
        assert context.outer_reduction_stream != context.allgather_stream


def test_delayed_wgrad_requires_caller_managed_sync(distributed_setup):
    """Delayed weight gradients require explicit gradient synchronization."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = te.Linear(16, 16, device=device, delay_wgrad_compute=True)
    with (
        fully_shard_context(FsdpContext(device=device)),
        pytest.raises(ValueError, match="caller_managed_grad_sync=True"),
    ):
        fully_shard(model, mesh=mesh, placements=_default_placements())


@pytest.mark.parametrize("difference", ["weight-trace", "retain-distance"])
def test_recorded_weight_retention_rejects_rank_divergence(distributed_setup, difference):
    """Reject different weight tapes or retention thresholds without stranding peers."""
    setup = distributed_setup
    if setup.world_size < 2:
        pytest.skip("Weight-retention consensus requires multiple ranks.")
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    late = setup.rank == setup.world_size - 1
    distance = 4 if difference == "retain-distance" and late else 3
    torch.manual_seed(73)
    model = MultiChildModel(dim=32, num_children=2).to(setup.device)
    with fully_shard_context(
        FsdpContext(
            device=setup.device,
            prefetch_order=PrefetchOrder.RECORDED,
            prefetch_process_group=mesh.get_group(0),
            weight_retain_distance=distance,
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
        )
    ) as context:
        for module in (*model.layers, model):
            fully_shard(module, mesh=mesh, placements=_default_placements())

    inputs = torch.ones(2, 32, device=setup.device, requires_grad=True)
    for microbatch in range(3):
        with finalize_gradients(context, enabled=microbatch == 2):
            model(inputs).square().mean().backward()
    assert context.is_recording_execution_order
    assert context._weight_events
    if difference == "weight-trace" and late:
        event = context._weight_events[-1]
        assert event.operation == "reshard"
        context._weight_events[-1] = event._replace(operation="unshard")
    with pytest.raises(
        RuntimeError, match=r"Recorded FSDP execution or weight retention differs across ranks\."
    ):
        context.finish_grad_sync()
    # All ranks must reject the signature before any retention plan is replayed.
    dist.barrier(group=mesh.get_group(0), device_ids=[setup.device.index])


@pytest.mark.parametrize("prefetch", [False, True], ids=["on_demand", "prefetch"])
def test_cross_microbatch_weight_retention_matches_unretained_training(
    distributed_setup, monkeypatch, prefetch
):
    """Recorded retention saves real gathers without changing complete optimizer steps."""
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Gather reuse requires multiple ranks.")
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    prefetch_size = None if prefetch else 0
    models = []
    contexts = []
    optimizers = []
    gathers = []
    groups_by_model = []
    for distance in (0, 3):
        torch.manual_seed(73)
        model = MultiChildModel(dim=32, num_children=2).to(device)
        with fully_shard_context(
            FsdpContext(
                device=device,
                prefetch_order=PrefetchOrder.RECORDED,
                prefetch_process_group=mesh.get_group(0),
                weight_retain_distance=distance,
                runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
            )
        ) as context:
            # Speculation must not consume weight events or defeat gather reuse.
            for module in (*model.layers, model):
                fully_shard(
                    module,
                    mesh=mesh,
                    placements=_default_placements(),
                    schedule_policy=SchedulePolicy(
                        forward_prefetch_size=prefetch_size, backward_prefetch_size=prefetch_size
                    ),
                )
        groups = [group for module in (model, *model.layers) for group in module.parameter_groups]
        assert len(groups) == 3
        counters = []
        for group in groups:
            gather = Mock(wraps=group._allgather_model_weight)
            monkeypatch.setattr(group, "_allgather_model_weight", gather)
            counters.append(gather)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        fully_shard_optimizer(optimizer)
        models.append(model)
        contexts.append(context)
        optimizers.append(optimizer)
        gathers.append(counters)
        groups_by_model.append(groups)

    torch.manual_seed(101 + distributed_setup.rank)
    inputs = torch.randn(3, 4, 32, device=device, requires_grad=True)
    previous_weights = torch.cat(
        [parameter.detach().to_local().flatten() for parameter in models[0].parameters()]
    )

    def train_step(index):
        model, context, optimizer = models[index], contexts[index], optimizers[index]
        optimizer.zero_grad(set_to_none=False)
        if inputs.grad is not None:
            inputs.grad.zero_()
        losses = []
        for microbatch, x in enumerate(inputs):
            with finalize_gradients(context, enabled=microbatch == len(inputs) - 1):
                loss = model(x).square().mean() / len(inputs)
                loss.backward()
            losses.append(loss.detach())
        context.finish_grad_sync()
        # This boundary must invalidate every retained gather before updating masters.
        for group in groups_by_model[index]:
            assert not group._unshard_events
            assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0
        gradients = torch.cat(
            [parameter.grad.to_local().flatten() for parameter in model.parameters()]
        )
        optimizer.step()
        weights = torch.cat(
            [parameter.detach().to_local().flatten() for parameter in model.parameters()]
        )
        return torch.stack(losses), gradients, weights

    def check_parity(outputs):
        nonlocal previous_weights
        for actual, expected in zip(outputs[1], outputs[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not torch.equal(outputs[0][2], previous_weights)
        previous_weights = outputs[0][2].clone()

    def reset_counts():
        for counters in gathers:
            for counter in counters:
                counter.reset_mock()

    def check_counts(expected):
        assert [
            tuple(counter.call_count for counter in counters) for counters in gathers
        ] == expected

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    # Record a complete three-microbatch step, then replay it eagerly with fresh masters.
    for step, expected in enumerate(([(6, 6, 6), (6, 6, 6)], [(6, 6, 6), (1, 4, 3)])):
        reset_counts()
        with torch.cuda.stream(stream):
            outputs = [train_step(index) for index in range(2)]
        torch.cuda.current_stream(device).wait_stream(stream)
        check_counts(expected)
        check_parity(outputs)
        stream.wait_stream(torch.cuda.current_stream(device))
        if step == 0:
            evaluation_outputs = []
            with torch.cuda.stream(stream), torch.no_grad():
                for model, context in zip(models, contexts, strict=True):
                    execution = tuple(context.execution_order)
                    weight_events = tuple(context._weight_events)
                    positions = (context.execution_order.position, context._weight_position)
                    model.eval()
                    evaluation_outputs.append(model(inputs[0]))
                    model.train()
                    assert tuple(context.execution_order) == execution
                    assert tuple(context._weight_events) == weight_events
                    assert (context.execution_order.position, context._weight_position) == positions
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.testing.assert_close(evaluation_outputs[1], evaluation_outputs[0], rtol=0, atol=0)
            stream.wait_stream(torch.cuda.current_stream(device))

    # Each MB uses root/child0/child1 forward and root/child1/child0 backward.
    # Distance three retains root across all uses, child1 F->B, and child0 B->next F.
    # Thus the retained step gathers (root=1, child0=4, child1=3), versus (6, 6, 6).
    reset_counts()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [train_step(index) for index in range(2)]
    check_counts([(6, 6, 6), (1, 4, 3)])
    torch.cuda.current_stream(device).wait_stream(stream)
    for _ in range(3):
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream(device).wait_stream(stream)
        check_parity(outputs)
        # Replay executes the captured collectives without invoking Python wrappers.
        check_counts([(6, 6, 6), (1, 4, 3)])
        stream.wait_stream(torch.cuda.current_stream(device))
