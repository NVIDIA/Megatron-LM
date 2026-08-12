# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP runtime contexts."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
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
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


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
def test_fine_grained_hooks_preserve_registered_module_hierarchy(distributed_setup):
    """Fine-grained parent references must not become registered child modules."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)
    module_names = tuple(name for name, _ in model.named_modules())
    layer_keys = tuple(model.layers._modules)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    assert tuple(name for name, _ in model.named_modules()) == module_names
    assert tuple(model.layers._modules) == layer_keys


def test_sibling_roots_without_parent_keep_separate_contexts(distributed_setup):
    """Independent FSDP roots should not share runtime scheduling state."""
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
def test_post_backward_release_processes_nested_fsdp_modules_once(distributed_setup, monkeypatch):
    """Manual 1F1B release should include nested units without reducing twice."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel().to(device)

    with fully_shard_context(device=device):
        fully_shard(
            model.inner, mesh=mesh, placements=_flat_placements(), skip_backward_callback=True
        )
        fully_shard(model, mesh=mesh, placements=_flat_placements(), skip_backward_callback=True)

    calls = []
    for name, module in (("root", model), ("inner", model.inner)):
        monkeypatch.setattr(
            module,
            "_reshard_parameter_groups",
            lambda name=name: calls.append((name, "reshard")),
        )
        monkeypatch.setattr(
            module, "_reduce_gradient_groups", lambda name=name: calls.append((name, "reduce"))
        )

    model.post_backward_release_module()
    model.post_backward_release_module()

    # Each nested unit is resharded and reduced exactly once per backward;
    # the relative order of the two operations is not a contract.
    assert sorted(calls) == [
        ("inner", "reduce"),
        ("inner", "reshard"),
        ("root", "reduce"),
        ("root", "reshard"),
    ]


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
        # A different device must not be joined silently; the ambient context
        # is CUDA so requesting a CPU context keeps the nesting rejection.
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=torch.device("cpu"), reuse_existing=True):
                pass
        fully_shard(model, mesh=mesh, placements=_flat_placements())
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
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    assert runner.is_tracing

    # Batch 1 (trace): consume in schedule order F L0, B L2, F L1. No
    # prefetch during tracing.
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") is None
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") is None
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None
    assert runner.is_tracing

    # Batch boundary compiles the trace into the replay cycle.
    runner.complete_trace()
    assert not runner.is_tracing

    # Batch 2 (replay): consume in the same order; each call returns the
    # traced next consumer (with wrap-around at the batch boundary).
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[2], "colwise")
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") == (layers[1], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[0], "rowwise")

    # Divergence re-traces from the mismatching occurrence: the reshard
    # round of L0 is reset, then L0 is consumed with the wrong orientation.
    runner.record_reshard(layers[0])
    assert _record_unshard_and_prefetch(runner, layers[0], "colwise") is None
    assert runner.is_tracing


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


def test_runner_wrap_around_chunk_cycle_prefetches_first_module(distributed_setup):
    """Replay must prefetch across the cycle wrap: 0 -> 1 -> 2 -> 0.

    A VPP schedule walks the chunk cycle repeatedly. The traced successor of
    the last occurrence wraps to the first module, so consuming module 2
    must prefetch module 0 (not None).
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner

    # Batch 1 traces the chunk cycle 0 -> 1 -> 2, each unshard followed by
    # its reshard round.
    for layer in layers:
        assert _record_unshard_and_prefetch(runner, layer, "rowwise") is None
        runner.record_reshard(layer)
    runner.complete_trace()

    # Batch 2 replays two full cycles, prefetching the successor at every
    # step including the 2 -> 0 wrap.
    for _ in range(2):
        for i, layer in enumerate(layers):
            next_layer = layers[(i + 1) % len(layers)]
            assert _record_unshard_and_prefetch(runner, layer, "rowwise") == (next_layer, "rowwise")
            runner.record_reshard(layer)


def test_runner_wrap_within_batch_multiple_cycles(distributed_setup):
    """A single global batch can contain several full chunk cycles.

    The trace records every occurrence of the batch (e.g. two passes over
    the chunk cycle), and replay returns the exact next occurrence,
    including wraps inside the batch and at the batch boundary.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

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

    # Batch 2: each occurrence returns the exact next occurrence (the
    # prefetch skips reshard events, wrapping within and across cycles).
    expected_cycle = [layers[0], layers[1], layers[2], layers[0], layers[1], layers[2]]
    for i, layer in enumerate(expected_cycle):
        expected_next = expected_cycle[(i + 1) % len(expected_cycle)]
        assert _record_unshard_and_prefetch(runner, layer, "rowwise") == (expected_next, "rowwise")
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
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

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
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[2], "rowwise")


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
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

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
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[2], "colwise")


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
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

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
    assert [e.kind for e in runner._trace] == [
        EventKind.UNSHARD,
        EventKind.RESHARD,
    ] * len(layers)

    # Batch 2 replay: consume in order; each returns the traced next (wrap).
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[1], "rowwise")
    runner.record_reshard(layers[0])
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[2], "rowwise")
    runner.record_reshard(layers[1])
    assert _record_unshard_and_prefetch(runner, layers[2], "rowwise") == (layers[0], "rowwise")
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
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

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
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (layers[0], "rowwise")
