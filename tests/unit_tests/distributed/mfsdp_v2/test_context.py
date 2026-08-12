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
