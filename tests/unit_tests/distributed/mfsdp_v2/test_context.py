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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


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


def _record_unshard_and_prefetch(runner, module, orientation, *, depth=1):
    """Record an unshard on the runner and return the suggested prefetch."""
    runner.record_unshard(module, orientation)
    suggestion = runner.suggest_prefetch_plan(module, orientation, depth=depth)
    if suggestion is None:
        return None
    return suggestion.module, suggestion.orientation


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


def test_vpp_chunks_share_one_context_via_reuse(distributed_setup):
    """Per-chunk adapter scopes should join one outer VPP construction context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    chunks = [MultiChildModel(dim=4, num_children=2).to(device) for _ in range(2)]

    with fully_shard_context(device=device) as outer:
        for chunk in chunks:
            with fully_shard_context(device=device, reuse_existing=True) as reused:
                assert reused is outer
                fully_shard(chunk, mesh=mesh, placements=_flat_placements())

    for chunk in chunks:
        assert chunk.context is outer
        assert chunk.is_root()
    assert list(outer.forward_order) == chunks
    assert list(outer.backward_order) == list(reversed(chunks))
    assert chunks[0].context.allgather_stream is chunks[1].context.allgather_stream
    assert chunks[0].context.reduce_scatter_stream is chunks[1].context.reduce_scatter_stream


def test_reuse_existing_requires_compatible_context(distributed_setup):
    """A reuse request must match the ambient context's device and memory mode."""
    device = distributed_setup.device

    with fully_shard_context(device=device):
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=torch.device("cpu"), reuse_existing=True):
                pass
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(
                device=device, reuse_existing=True, use_symmetric_memory=True
            ):
                pass
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=device, reuse_existing=True, use_trace_replay=True):
                pass
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=device, reuse_existing=True, prefetch_depth=2):
                pass


def test_prefetch_depth_validates_and_enables_trace_replay(distributed_setup):
    """A non-default depth is sufficient to select occurrence-based replay."""
    device = distributed_setup.device
    with pytest.raises(ValueError, match="prefetch_depth must be positive"):
        with fully_shard_context(device=device, prefetch_depth=0):
            pass

    with fully_shard_context(device=device, prefetch_depth=2) as context:
        assert context.prefetch_depth == 2
        assert context.runner.use_trace_replay


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
    """Manual 1F1B release should include nested units still in the backward phase."""
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

    # The schedule skipped the inner unit's per-module release; its post_backward
    # should still finalize it because it remains in the BACKWARD phase.
    model.phase = FsdpModule.Phase.BACKWARD
    model.inner.phase = FsdpModule.Phase.BACKWARD
    model.post_backward()

    # The root post_backward finalizes itself and any nested unit still in the
    # BACKWARD phase; the relative order of the two operations is not a contract.
    assert sorted(calls) == [
        ("inner", "reduce"),
        ("inner", "reshard"),
        ("root", "reduce"),
        ("root", "reshard"),
    ]


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

    # Batch 2 (replay): consume in the same order. The final occurrence has
    # no prefetch target before the optimizer boundary.
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (layers[2], "colwise")
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") == (layers[1], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None

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
    with pytest.raises(ValueError, match="greater than one requires trace replay"):
        ctx.runner.suggest_prefetch_plan(model, "rowwise", depth=2)

    # A full forward does not feed the runner in default mode.
    with torch.no_grad():
        model(torch.ones(2, 4, device=device))
    assert ctx.runner.is_tracing
    assert not ctx.runner._trace


def test_runner_global_batch_boundary_stops_prefetch(distributed_setup):
    """Replay must not prefetch parameters across an optimizer step."""
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

    # Each replay batch may prefetch within its own trace, but its last
    # occurrence must wait for the next batch to gather updated weights.
    for _ in range(2):
        for i, layer in enumerate(layers):
            expected = (layers[i + 1], "rowwise") if i + 1 < len(layers) else None
            assert _record_unshard_and_prefetch(runner, layer, "rowwise") == expected
            runner.record_reshard(layer)
            if i + 1 == len(layers):
                assert not runner.suggest_skip_reshard(layer)
        runner.complete_trace()


def test_runner_wrap_within_batch_multiple_cycles(distributed_setup):
    """A single global batch can contain several full chunk cycles."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    for _ in range(2):
        for layer in layers:
            assert _record_unshard_and_prefetch(runner, layer, "rowwise") is None
            runner.record_reshard(layer)
    runner.complete_trace()
    assert len(runner._trace) == 6 * 2

    expected_cycle = [layers[0], layers[1], layers[2], layers[0], layers[1], layers[2]]
    for i, layer in enumerate(expected_cycle):
        expected = (
            (expected_cycle[i + 1], "rowwise") if i + 1 < len(expected_cycle) else None
        )
        assert _record_unshard_and_prefetch(runner, layer, "rowwise") == expected
        runner.record_reshard(layer)


def test_runner_prefetch_depth_stops_at_global_batch_boundary(distributed_setup):
    """Depth two should return None when its target would cross the optimizer step."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=4).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    for layer in layers:
        assert _record_unshard_and_prefetch(runner, layer, "rowwise", depth=2) is None
        runner.record_reshard(layer)
    runner.complete_trace()

    for _ in range(2):
        for index, layer in enumerate(layers):
            expected = (layers[index + 2], "rowwise") if index + 2 < len(layers) else None
            assert _record_unshard_and_prefetch(
                runner, layer, "rowwise", depth=2
            ) == expected
            runner.record_reshard(layer)
        runner.complete_trace()


def test_runner_prefetch_depth_uses_occurrence_order_for_repeated_module(distributed_setup):
    """Repeated VPP module objects retain occurrence-order depth and orientation."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    occurrences = (
        (layers[0], "rowwise"),
        (layers[1], "rowwise"),
        (layers[0], "colwise"),
        (layers[2], "colwise"),
    )
    for layer, orientation in occurrences:
        assert _record_unshard_and_prefetch(runner, layer, orientation, depth=2) is None
        runner.record_reshard(layer)
    runner.complete_trace()

    for index, (layer, orientation) in enumerate(occurrences):
        expected = occurrences[index + 2] if index + 2 < len(occurrences) else None
        assert _record_unshard_and_prefetch(
            runner, layer, orientation, depth=2
        ) == expected
        runner.record_reshard(layer)


def test_runner_rejects_prefetch_depth_larger_than_trace(distributed_setup):
    """A depth without a corresponding traced occurrence is a configuration error."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    for layer in layers:
        assert _record_unshard_and_prefetch(runner, layer, "rowwise", depth=3) is None
        runner.record_reshard(layer)
    runner.complete_trace()

    runner.record_unshard(layers[0], "rowwise")
    with pytest.raises(ValueError, match="exceeds the 2 UNSHARD occurrences"):
        runner.suggest_prefetch_plan(layers[0], "rowwise", depth=3)


def test_runner_retains_deep_prefetch_through_intervening_reshard(distributed_setup):
    """A depth target reservation survives its earlier physical occurrence."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    source, target, middle = modules
    with fully_shard_context(device=device, prefetch_depth=3) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    runner.record_unshard(source, "rowwise")
    runner.record_unshard(target, "colwise")
    runner.record_reshard(target)
    runner.record_unshard(middle, "rowwise")
    runner.record_unshard(target, "rowwise")
    runner.complete_trace()

    assert runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise", depth=3)
    assert suggestion is not None
    assert suggestion.module is target
    assert suggestion.orientation == "rowwise"
    assert suggestion.release_after_reshard_index is not None
    runner.defer_prefetch(suggestion)

    assert runner.record_unshard(target, "colwise")
    reshard_index = runner.record_reshard(target)
    assert runner.retain_prefetches_across_reshard(target, reshard_index)
    assert runner._retained_prefetches

    assert runner.record_unshard(middle, "rowwise")
    assert runner.record_unshard(target, "rowwise")
    assert not runner._retained_prefetches


def test_finish_grad_sync_releases_unconsumed_prefetch(distributed_setup, monkeypatch):
    """Speculative full parameters must not survive into the optimizer step."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    source, target, middle = modules
    with fully_shard_context(device=device, prefetch_depth=3) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    runner.record_unshard(source, "rowwise")
    runner.record_unshard(target, "colwise")
    runner.record_reshard(target)
    runner.record_unshard(middle, "rowwise")
    runner.record_unshard(target, "rowwise")
    runner.complete_trace()

    runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise", depth=3)
    assert suggestion is not None
    runner.defer_prefetch(suggestion)
    runner.record_unshard(target, "colwise")
    reshard_index = runner.record_reshard(target)
    assert runner.retain_prefetches_across_reshard(target, reshard_index)

    releases = []
    waits = []
    monkeypatch.setattr(
        target,
        "_reshard_parameter_groups",
        lambda *, record_execution: releases.append(record_execution),
    )
    current_stream = type(
        "FakeStream", (), {"wait_stream": lambda _self, stream: waits.append(stream)}
    )()
    monkeypatch.setattr(context, "current_stream", lambda: current_stream)

    context.finish_grad_sync()

    assert releases == [False]
    assert waits == [context.reduce_scatter_stream]
    assert not runner._retained_prefetches


def test_incomplete_replay_flushes_immediate_prefetch_and_retraces(
    distributed_setup, monkeypatch
):
    """A truncated replay cannot carry eagerly prefetched stale weights forward."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    source, target = modules
    with fully_shard_context(device=device, use_trace_replay=True, prefetch_depth=1) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    runner.record_unshard(source, "rowwise")
    runner.record_unshard(target, "rowwise")
    runner.complete_trace()

    runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise")
    assert suggestion is not None and suggestion.module is target
    runner.track_prefetch(suggestion)

    releases = []
    monkeypatch.setattr(
        target,
        "_reshard_parameter_groups",
        lambda *, record_execution: releases.append(record_execution),
    )
    current_stream = type("FakeStream", (), {"wait_stream": lambda _self, _stream: None})()
    monkeypatch.setattr(context, "current_stream", lambda: current_stream)

    context.finish_grad_sync()
    runner.complete_trace()

    assert releases == [False]
    assert runner.is_tracing
    assert not runner._trace
    assert not runner._resident_prefetches


def test_module_reshard_honors_prefetch_reservation(distributed_setup, monkeypatch):
    """A retained depth target bypasses physical parameter reshard and release."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(device=device, prefetch_depth=2) as context:
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    calls = []
    monkeypatch.setattr(context.runner, "record_reshard", lambda target: 17)
    monkeypatch.setattr(context.runner, "suggest_skip_reshard", lambda target: False)
    monkeypatch.setattr(
        context.runner,
        "retain_prefetches_across_reshard",
        lambda target, trace_index: calls.append((target, trace_index)) or True,
    )
    for group in module.parameter_groups:
        monkeypatch.setattr(
            group,
            "reshard_parameters",
            lambda: pytest.fail("retained materialization was physically resharded"),
        )

    module._reshard_parameter_groups()

    assert calls == [(module, 17)]


def test_deep_prefetch_ignores_logically_skipped_reshard(distributed_setup):
    """An immediately reused materialization must not become a lifetime gate."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    source, target = modules
    with fully_shard_context(device=device, prefetch_depth=2) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    runner.record_unshard(source, "rowwise")
    runner.record_unshard(target, "rowwise")
    runner.record_reshard(target)
    runner.record_unshard(target, "rowwise")
    runner.complete_trace()

    assert runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise", depth=2)
    assert suggestion is not None
    assert suggestion.module is target
    assert suggestion.release_after_reshard_index is None


def test_runner_divergence_retraces_and_recovers(distributed_setup):
    """A divergent schedule should re-trace and recover at the next boundary."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    for layer in layers:
        _record_unshard_and_prefetch(runner, layer, "rowwise")
    runner.complete_trace()

    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    assert _record_unshard_and_prefetch(runner, layers[2], "rowwise") is None
    assert runner.is_tracing

    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    _record_unshard_and_prefetch(runner, layers[1], "rowwise")
    runner.complete_trace()
    assert not runner.is_tracing
    assert _record_unshard_and_prefetch(runner, layers[2], "rowwise") == (
        layers[0],
        "rowwise",
    )
    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (
        layers[1],
        "rowwise",
    )
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None


def test_runner_tolerates_transient_mismatch_without_crashing(distributed_setup):
    """A transient orientation mismatch should fall back to re-tracing."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    for layer in layers:
        _record_unshard_and_prefetch(runner, layer, "rowwise")
    runner.complete_trace()

    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (
        layers[1],
        "rowwise",
    )
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (
        layers[2],
        "rowwise",
    )
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") is None
    assert runner.is_tracing

    _record_unshard_and_prefetch(runner, layers[0], "rowwise")
    _record_unshard_and_prefetch(runner, layers[1], "rowwise")
    runner.complete_trace()
    assert not runner.is_tracing
    assert _record_unshard_and_prefetch(runner, layers[2], "colwise") == (
        layers[0],
        "rowwise",
    )


def test_unshard_records_one_consume_per_module_per_pass(distributed_setup):
    """Repeated fine-grained hooks should record one consume per module round."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    for layer in layers:
        runner.record_unshard(layer, "rowwise")
        runner.record_unshard(layer, "rowwise")
        runner.record_reshard(layer)
    runner.complete_trace()
    assert len(runner._trace) == 2 * len(layers)
    assert [event.kind for event in runner._trace] == [
        EventKind.UNSHARD,
        EventKind.RESHARD,
    ] * len(layers)

    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (
        layers[1],
        "rowwise",
    )
    runner.record_reshard(layers[0])
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") == (
        layers[2],
        "rowwise",
    )


def test_complete_trace_clears_dedup_so_replay_records(distributed_setup):
    """The batch boundary should clear fine-grained consume deduplication."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device, use_trace_replay=True):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    runner.record_unshard(layers[0], "rowwise")
    runner.record_reshard(layers[0])
    runner.record_unshard(layers[1], "rowwise")
    runner.complete_trace()
    assert not runner._consumed_this_round
    assert runner._complete_trace_calls == 1

    assert _record_unshard_and_prefetch(runner, layers[0], "rowwise") == (
        layers[1],
        "rowwise",
    )
    runner.record_reshard(layers[0])
    assert _record_unshard_and_prefetch(runner, layers[1], "rowwise") is None


def test_trace_pool_plans_after_first_execution_replay(distributed_setup):
    """Storage planning must observe the prefetch-enabled replay lifetime."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=1).to(device)

    with fully_shard_context(
        device=device, use_trace_replay=True, enable_trace_pool=True
    ) as context:
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements(), fine_grained=True)

    runner = context.runner
    allocator = context.trace_pool_allocator
    assert allocator is not None

    allocator.allocate("buffer", 8, torch.float32, device, arena="allgather")
    allocator.free("buffer")
    runner.record_unshard(model.layers[0], "rowwise")
    runner.record_reshard(model.layers[0])
    context.complete_trace()
    assert allocator.phase == "trace"

    allocator.allocate("buffer", 8, torch.float32, device, arena="allgather")
    allocator.free("buffer")
    runner.record_unshard(model.layers[0], "rowwise")
    runner.record_reshard(model.layers[0])
    context.complete_trace()
    assert allocator.phase == "optimized"
