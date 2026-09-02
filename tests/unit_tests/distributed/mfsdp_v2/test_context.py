# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP runtime contexts."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_runner import EventKind


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


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _record_unshard_and_prefetch(runner, module, orientation):
    """Record an unshard on the runner and return the suggested prefetch."""
    runner.record_unshard(module, orientation)
    return runner.suggest_prefetch(module, orientation)


def test_child_then_parent_share_one_context(distributed_setup):
    """Modules constructed together should eagerly share one context."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    with fully_shard_context(device=device) as context:
        fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())
        assert model.context is context
        assert model.inner.context is context

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.inner.context is model.context
    assert model.is_root()
    assert not model.inner.is_root()


def test_two_child_subtrees_then_parent_share_one_context(distributed_setup):
    """One construction scope should assign one context across child subtrees."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.layers[0].context is model.context
    assert model.layers[1].context is model.context


def test_sibling_roots_share_context_and_cross_root_orders(distributed_setup):
    """Independent roots should share streams and follow construction order."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.layers[0].context
    assert model.layers[1].context is context
    assert model.layers[0].is_root()
    assert model.layers[1].is_root()
    assert list(context.forward_order) == [model.layers[0], model.layers[1]]
    assert list(context.backward_order) == [model.layers[1], model.layers[0]]


def test_skip_forward_backward_hooks(distributed_setup):
    """Integrations may replace the standard FSDP module lifecycle hooks."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Sequential(nn.Linear(4, 4, bias=False)).to(device)

    with fully_shard_context(device=device):
        fully_shard(
            model, mesh=mesh, placements=_flat_placements(), skip_forward_backward_hooks=True
        )

    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    assert not model._backward_pre_hooks
    assert not model._backward_hooks


def test_nested_prefetch_orders_use_dfs(distributed_setup):
    """Nested FsdpModules should use DFS orders for one-step prefetch."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.left.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model.left, mesh=mesh, placements=_flat_placements())
        fully_shard(model.right, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.context
    assert list(context.forward_order) == [model, model.left, model.left.inner, model.right]
    assert list(context.backward_order) == [model, model.right, model.left, model.left.inner]


def test_nested_and_sibling_roots_use_cross_root_orders(distributed_setup):
    """Context orders should concatenate nested roots at construction boundaries."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.left.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model.left, mesh=mesh, placements=_flat_placements())
        fully_shard(model.right, mesh=mesh, placements=_flat_placements())

    context = model.left.context
    assert model.left.is_root()
    assert model.right.is_root()
    assert not model.left.inner.is_root()
    assert list(context.forward_order) == [model.left, model.left.inner, model.right]
    assert list(context.backward_order) == [model.right, model.left, model.left.inner]


def test_fully_shard_requires_context(distributed_setup):
    """fully_shard should reject construction without an active context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)

    with pytest.raises(RuntimeError, match="inside fully_shard_context"):
        fully_shard(model, mesh=mesh, placements=_flat_placements())


def test_forward_requires_finalized_context(distributed_setup):
    """Forward should be unavailable until construction scope exit."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    x = torch.ones(2, 4, device=device)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())
        with pytest.raises(RuntimeError, match="Exit fully_shard_context"):
            model(x)

    model(x)


def test_fully_shard_context_rejects_nesting(distributed_setup):
    """A construction scope should reject an ambiguous nested context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)

    with fully_shard_context(device=device):
        fully_shard(model[0], mesh=mesh, placements=_flat_placements())
        outer_context = model[0].context
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=device):
                pass
        fully_shard(model[1], mesh=mesh, placements=_flat_placements())

    assert model[0].context is outer_context
    assert model[1].context is outer_context


def test_fully_shard_rejects_child_from_another_context(distributed_setup):
    """A parent cannot join a context different from an FSDP child context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    with fully_shard_context(device=device) as first_context:
        fully_shard(model.inner, mesh=mesh, placements=_flat_placements())

    with fully_shard_context(device=device):
        with pytest.raises(ValueError, match="another fully_shard_context"):
            fully_shard(model, mesh=mesh, placements=_flat_placements())

    assert model.inner.context is first_context


def test_multiple_forwards_before_backwards_reset_gradient_readiness(
    distributed_setup, monkeypatch
):
    """Consecutive pipeline backwards should each finalize gradient reduction."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    reduce_calls = []
    original_reduce_gradient_groups = model._reduce_gradient_groups

    def record_reduce_gradient_groups() -> None:
        reduce_calls.append(None)
        original_reduce_gradient_groups()

    monkeypatch.setattr(model, "_reduce_gradient_groups", record_reduce_gradient_groups)

    # Pipeline warmup may run multiple forwards before cooldown runs consecutive
    # backwards. Each backward must complete gradient reduction independently.
    losses = [model(torch.ones(2, 4, device=device)).sum() for _ in range(2)]
    for loss in reversed(losses):
        loss.backward()

    assert len(reduce_calls) == 2
    assert model.phase is model.Phase.RESTING


def test_vpp_chunks_share_one_context_via_reuse(distributed_setup):
    """VPP chunks wrapped inside one scope should share a single FsdpContext.

    Simulates the training-loop wrapping of multiple virtual-pipeline chunks:
    the outer fully_shard_context() is opened once, and each chunk's adapter
    (modeled here by nested reuse_existing scopes) joins it instead of
    creating a new context. All chunks must share streams and cross-root
    prefetch orders.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    chunks = [MultiChildModel(dim=4, num_children=2).to(device) for _ in range(2)]

    with fully_shard_context(device=device) as outer:
        for chunk in chunks:
            # Mirrors FullyShardedDataParallelV2.__init__ wrapping a chunk:
            # reuse_existing joins the training-loop context.
            with fully_shard_context(device=device, reuse_existing=True):
                fully_shard(chunk, mesh=mesh, placements=_flat_placements())

        assert chunk.context is outer

    # After finalize, every chunk root is registered in the shared context's
    # cross-root orders.
    for chunk in chunks:
        assert chunk.context is outer
        assert chunk.is_root()

    assert len(list(outer.forward_order)) == 2
    assert len(list(outer.backward_order)) == 2
    assert outer.allgather_stream is chunks[0].context.allgather_stream
    assert outer.reduce_scatter_stream is chunks[0].context.reduce_scatter_stream


def test_vpp_chunks_reuse_context_on_same_device_only(distributed_setup):
    """reuse_existing must join only a context on the same device."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        with pytest.raises(ValueError, match="different device"):
            with fully_shard_context(device=torch.device("cpu"), reuse_existing=True):
                pass
        fully_shard(model, mesh=mesh, placements=_flat_placements())


@pytest.mark.parametrize(
    "option", ["use_trace_replay", "use_symmetric_memory", "unify_communication_stream"]
)
def test_vpp_chunks_reuse_context_requires_matching_options(distributed_setup, option):
    """A reused context must preserve all runtime-affecting options."""
    device = distributed_setup.device

    with fully_shard_context(device=device):
        with pytest.raises(ValueError, match=rf"different options: {option}"):
            with fully_shard_context(device=device, reuse_existing=True, **{option: True}):
                pass


def test_prefetch_traces_and_replays_actual_consume_order(distributed_setup):
    """The runner should trace batch 1 and replay the actual consume order.

    The fine-grained schedule can consume modules in an order that differs
    from forward_order/backward_order (e.g. F L0 -> B L2 -> F L1). The first
    batch traces that order and returns no prefetch; later batches replay it
    and prefetch the actual next consumer.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner
    assert runner.is_tracing

    # Batch 1 (trace): consume in schedule order F L0, B L2, F L1. No
    # prefetch during tracing.
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") is None
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") is None
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None
    assert runner.is_tracing

    # Batch boundary compiles the trace for replay.
    runner.complete_trace()
    assert not runner.is_tracing

    # Batch 2 (replay): consume in the same order; each call returns the
    # traced next consumer, except the final occurrence. Prefetch must not
    # cross the optimizer boundary because those full weights would become
    # stale as soon as the optimizer updates the sharded weights.
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[2], "colwise")
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") == (layers[1], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None

    # The explicit next boundary resets the non-cyclic cursor.
    runner.complete_trace()
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[2], "colwise")


def test_eager_pre_forward_feeds_context_runner(distributed_setup):
    """In trace-replay mode, the eager pre_forward feeds the context runner.

    The runner is shared across the full FsdpContext, so a consume driven by
    the eager forward hooks is traced identically to a fine-grained consume:
    batch 1 records (demand-only), and after the batch boundary batch 2
    replays the traced order.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    ctx = model.context
    assert ctx.runner.use_trace_replay
    assert ctx.runner.is_tracing

    # Batch 1: eager forward consumes are recorded; without an optimizer
    # boundary the runner stays tracing.
    with torch.no_grad():
        model(torch.ones(2, 4, device=device))
    assert ctx.runner.is_tracing
    assert len(ctx.runner._trace) >= 2  # root + child layers

    # The batch boundary compiles the trace; subsequent forwards replay it.
    ctx.runner.complete_trace()
    assert not ctx.runner.is_tracing
    with torch.no_grad():
        model(torch.ones(2, 4, device=device))
    assert not ctx.runner.is_tracing


def test_default_mode_uses_static_order_prefetch(distributed_setup):
    """Default mode keeps static-order prefetch and skips the runner.

    With use_trace_replay=False (the default), pre_forward/pre_backward
    prefetch via forward_order/backward_order exactly as before the runner,
    and the runner's trace stays untouched.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    ctx = model.context
    assert not ctx.runner.use_trace_replay
    assert ctx.runner.is_tracing

    # A full forward does not feed the runner in default mode.
    with torch.no_grad():
        model(torch.ones(2, 4, device=device))
    assert ctx.runner.is_tracing
    assert not ctx.runner._trace


def test_default_mode_prefetches_static_backward_successor(distributed_setup):
    """Backward compute should prefetch the next static backward unit.

    A module's BACKWARD phase suppresses rowwise activation-recompute
    prefetch, but it must not suppress a genuine colwise backward successor.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    model.phase = model.Phase.BACKWARD
    assert model.context.runner.suggest_prefetch(model, "rowwise") is None
    assert model.context.runner.suggest_prefetch(model, "colwise") == (model.layers[1], "colwise")


def test_runner_does_not_prefetch_across_global_batch_boundary(distributed_setup):
    """The final replay occurrence must not prefetch next-batch weights."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Batch 1 traces 0 -> 1 -> 2, including each release.
    for layer in layers:
        assert _record_unshard_and_prefetch(runner, layer, "rowwise") is None
        runner.record_reshard(layer)
    runner.complete_trace()

    # Batch 2 may prefetch only successors inside this batch. At the final
    # consume there is no successor; wrapping to L0 would retain stale full
    # weights across the optimizer update.
    for layer, next_layer in zip(layers, [layers[1], layers[2], None]):
        prefetch = _record_unshard_and_prefetch(runner, layer, "rowwise")
        expected = None if next_layer is None else (next_layer, "rowwise")
        assert prefetch == expected
        runner.record_reshard(layer)
        if next_layer is None:
            assert not runner.suggest_skip_reshard(layer)

    # Only the explicit boundary permits the next batch to start at L0.
    runner.complete_trace()
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")


def test_shorter_replay_invalidates_trace_at_global_batch_boundary(distributed_setup):
    """A short replay batch must not reuse a stale suffix in the next batch."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner
    for layer in layers:
        _record_unshard_and_prefetch(runner, layer, "rowwise")
    runner.complete_trace()

    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    runner.complete_trace()

    assert runner.is_tracing
    assert runner._trace == []
    assert runner._replay_index == 0
    assert runner._divergences == 1


def test_runner_wrap_within_batch_multiple_cycles(distributed_setup):
    """A single global batch can contain several full chunk cycles.

    The trace records every occurrence of the batch (e.g. two passes over
    the chunk cycle), and replay returns the exact next occurrence,
    including the wrap between occurrences inside the batch, but never across
    the optimizer boundary.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Batch 1: two full cycles (0,1,2,0,1,2), each unshard followed by its
    # reshard round.
    for _ in range(2):
        for layer in layers:
            assert _record_unshard_and_prefetch(runner, layer, "rowwise") is None
            runner.record_reshard(layer)
    runner.complete_trace()
    assert len(runner._trace) == 6 * 2

    # Batch 2: each occurrence returns the exact next occurrence. L2 in the
    # first cycle may prefetch L0 in the second cycle, while the final L2 has
    # no successor and must not wrap to the next batch.
    expected_cycle = [layers[0], layers[1], layers[2], layers[0], layers[1], layers[2]]
    for i, layer in enumerate(expected_cycle):
        expected_next = expected_cycle[i + 1] if i + 1 < len(expected_cycle) else None
        prefetch = _record_unshard_and_prefetch(runner, layer, "rowwise")
        expected = None if expected_next is None else (expected_next, "rowwise")
        assert prefetch == expected
        runner.record_reshard(layer)


def test_runner_divergence_retraces_and_recovers(distributed_setup):
    """A schedule divergence re-traces, then recovers once a full cycle matches.

    After divergence the runner is tracing again (demand-only); once a full
    cycle has been observed, complete_trace re-enables replay.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Batch 1 trace: 0,1,2.
    for layer in layers:
        _record_unshard_and_prefetch(runner, layer, "rowwise")
    runner.complete_trace()

    # Divergence: consume 0 then 2 (expected 1). Re-traces from 2.
    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[2], "rowwise") is None
    assert runner.is_tracing

    # The remainder of the divergent batch is traced: 2, then 0, 1 (a full
    # cycle completes and compiles again at the next boundary).
    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    _record_unshard_and_prefetch(runner, layers[1], "rowwise")
    runner.complete_trace()
    assert not runner.is_tracing

    # Replay the recovered cycle.
    assert _record_unshard_and_prefetch(runner, layers[2], "rowwise") == (layers[0], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None


def test_runner_tolerates_transient_mismatch_without_crashing(distributed_setup):
    """A single unexpected consume must not crash replay; it re-traces.

    Real schedules can deviate transiently (e.g. an extra or reordered
    occurrence). The runner treats any mismatch as a divergence, clears the
    trace, and re-traces from the offending occurrence, so training never
    aborts on a prefetch-order error.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Batch 1 trace: 0, 1, 2.
    for layer in layers:
        _record_unshard_and_prefetch(runner, layer, "rowwise")
    runner.complete_trace()

    # Replay 0 -> 1, then a transient error: consume 2 instead of 2 (same
    # module) but with the wrong orientation — a mismatch that must not raise.
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[2], "rowwise")
    # Transient error: expected rowwise consume of layer 2 comes in colwise.
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") is None
    assert runner.is_tracing  # re-traced, demand-only, no crash

    # The remainder re-traces; a complete cycle compiles again.
    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    _record_unshard_and_prefetch(runner, layers[1], "rowwise")
    runner.complete_trace()
    assert not runner.is_tracing

    # Recovered replay follows the new cycle (the cycle now starts with the
    # colwise consume of layer 2 that caused the divergence).
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") == (layers[0], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None


def test_runner_releases_abandoned_prefetch_on_divergence(distributed_setup):
    """A mismatched replay must release a prefetched module that is not consumed."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Trace a complete batch and return every unit to sharded storage.
    for layer in layers:
        layer.unshard_parameters("rowwise")
        layer._reshard_parameter_groups()
    runner.complete_trace()
    assert all(layer._unshard_event is None for layer in layers)

    # Replaying L0 prefetches L1. The real schedule then consumes L2 instead,
    # so replay diverges and L1's abandoned full-weight storage must be
    # released. L2 is demand-unsharded and remains available to its consumer.
    layers[0].unshard_parameters("rowwise")
    assert layers[1]._unshard_event is not None
    layers[0]._reshard_parameter_groups()
    layers[2].unshard_parameters("rowwise")

    assert runner.is_tracing
    assert layers[1]._unshard_event is None
    assert layers[2]._unshard_event is not None
    layers[2]._reshard_parameter_groups()


def test_runner_rematerializes_prefetch_when_orientation_diverges(distributed_setup, monkeypatch):
    """A differently oriented consume must replace its stale prefetched payload."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    first, second = model.layers
    runner = model.context.runner
    runner.record_unshard(first, "rowwise")
    runner.record_unshard(second, "rowwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        second, "_release_unsharded_parameter_groups", lambda: calls.append(("release", None))
    )
    monkeypatch.setattr(
        second,
        "_unshard_parameter_groups",
        lambda orientation="rowwise": calls.append(("unshard", orientation)),
    )

    # Replaying the first consume predicts a rowwise second consume and
    # materializes that payload. The actual colwise consume must release the
    # prediction before issuing its demand all-gather; MXFP8 orientations are
    # distinct payloads and cannot safely share this storage.
    prefetched_module, prefetched_orientation = _record_unshard_and_prefetch(
        runner, first, "rowwise"
    )
    prefetched_module._unshard_parameter_groups(prefetched_orientation)
    second.unshard_parameters("colwise")

    assert runner.is_tracing
    assert calls == [("unshard", "rowwise"), ("release", None), ("unshard", "colwise")]


def test_unshard_records_one_consume_per_module_per_pass(distributed_setup):
    """Repeated fine-grained hooks on one module record a single unshard.

    The 1F1B schedule fires one unshard hook per sub-module (dense, experts),
    so the same FsdpModule can be unsharded several times within a pass. The
    runner dedups them (first call records, later calls are no-ops) and
    record_reshard() clears the round so the next unshard records again.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Batch 1: each layer unshards once, then resharps. Repeated unshard
    # calls (extra sub-module hooks) must be no-ops for the trace.
    for layer in layers:
        runner.record_unshard(layer, "rowwise")
        runner.record_unshard(layer, "rowwise")  # duplicate hook — deduped
        runner.record_reshard(layer)  # clears the round for the next unshard
    runner.complete_trace()
    assert len(runner._trace) == 2 * len(layers)
    assert [e.kind for e in runner._trace] == [EventKind.UNSHARD, EventKind.RESHARD] * len(layers)

    # Batch 2 replay: consume in order without wrapping across the boundary.
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")
    runner.record_reshard(layers[0])
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[2], "rowwise")
    runner.record_reshard(layers[1])
    assert _record_unshard_and_prefetch(runner, layers[2], "rowwise") is None
    runner.record_reshard(layers[2])


def test_complete_trace_clears_dedup_so_replay_records(distributed_setup):
    """complete_trace must clear the per-round dedup set.

    The trace batch's final unshards are not followed by a reshard, so stale
    dedup entries would suppress the first replay unshards. The batch
    boundary clears the set; every replay event must record.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    runner = model.context.runner

    # Trace batch: unshard L0 -> reshard -> unshard L1 (no trailing reshard).
    runner.record_unshard(layers[0], "rowwise")
    runner.record_reshard(layers[0])
    runner.record_unshard(layers[1], "rowwise")
    runner.complete_trace()
    assert not runner._consumed_this_round

    # Replay: both unshards must record despite the trailing unshard of the
    # trace batch (no stale dedup suppression).
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")
    runner.record_reshard(layers[0])
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None
