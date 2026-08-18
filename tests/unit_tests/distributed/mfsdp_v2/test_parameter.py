# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP parameter ownership and lifecycle."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
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


class TiedLM(nn.Module):
    """Tiny language model with shared input and output embedding weights."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute a scalar loss using both aliases of the shared weight."""
        return self.lm_head(self.embed_tokens(token_ids)).float().sum()


class SaveNonLeafWeightView(torch.autograd.Function):
    """Autograd function that saves a non-leaf parameter view for backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight_view: torch.Tensor) -> torch.Tensor:
        """Save the non-leaf weight view and run a simple elementwise op."""
        ctx.save_for_backward(x, weight_view)
        return x * weight_view

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the saved non-leaf weight view during backward."""
        x, weight_view = ctx.saved_tensors
        return grad_output * weight_view, grad_output * x


class NonLeafViewModel(nn.Module):
    """Model that saves a non-leaf parameter view across forward and backward."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run using a non-leaf view of the parameter."""
        weight_view = self.weight.view_as(self.weight)
        assert self.weight.is_leaf
        assert not weight_view.is_leaf
        return SaveNonLeafWeightView.apply(x, weight_view)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_nested_fully_shard_excludes_child_owned_parameters(distributed_setup):
    """An outer FsdpModule owns direct parameters but not nested child FsdpModule parameters."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NestedModel().to(device)

    with fully_shard_context(device=device):
        fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (inner_group,) = model.inner.parameter_groups
    (outer_group,) = model.parameter_groups

    assert [parameter.fqns for parameter in inner_group.fsdp_parameters] == [("weight",)]
    assert [parameter.fqns for parameter in outer_group.fsdp_parameters] == [("bias",)]


def test_tied_child_parameters_allocate_one_physical_weight(distributed_setup):
    """Tied registrations should allocate one DBuffer entry and optimizer parameter."""
    model = TiedLM()
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (parameter_group,) = model.parameter_groups
    (parameter,) = parameter_group.fsdp_parameters
    assert parameter.fqns == ("embed_tokens.weight", "lm_head.weight")
    assert parameter_group.main_weight.layout.size == 8 * 4
    # Both aliases must expose the same optimizer-visible sharded parameter.
    assert len(list(model.parameters())) == 1


def test_parameterless_parent_with_child_modules_trains(distributed_setup):
    """A parent with no unowned parameters should still root trainable child FsdpModules."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(5678)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False)).to(device)

    with fully_shard_context(device=device):
        fully_shard(model[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model[1], mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    assert model.parameter_groups == ()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(3, 4, device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()


def test_frozen_parameter_group_does_not_allocate_main_grad(distributed_setup):
    """A non-trainable parameter group should not allocate persistent main gradients."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    model.weight.requires_grad_(False)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups
    assert not group.requires_grad
    assert group.main_grad is None


def test_cpu_initialized_parameters_shard_to_mesh_device(distributed_setup):
    """A CPU model should support sharding a child before moving the full model to CUDA."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    nn.init.constant_(model[0].weight, 2.0)
    nn.init.constant_(model[1].weight, 3.0)
    x = torch.ones(1, 4)
    expected_output = model(x).to(device)

    # Shard the second layer's parameters onto the mesh device; the unwrapped
    # first layer's parameters remain on CPU until model.to(device) below.
    with fully_shard_context(device=device):
        fully_shard(model[1], mesh=mesh, placements=_flat_placements())

    assert model[0].weight.device.type == "cpu"
    assert isinstance(model[1].weight, DTensor)
    assert model[1].weight.device == device

    model.to(device)

    output = model(x.to(device))
    torch.testing.assert_close(output, expected_output)


def test_meta_parameters_shard_to_mesh_device(distributed_setup):
    """A sharded meta model should support initialization and forward."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16),
        nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16),
    )

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    nn.init.constant_(model[0].weight, 2.0)
    nn.init.constant_(model[1].weight, 3.0)
    # The exposed parameters update FP32 main weights, while forward uses separate BF16
    # model weights. This simulates load_checkpoint() until
    # https://github.com/NVIDIA/Megatron-LM/pull/6024 lands and syncs after loading.
    for parameter_group in model.parameter_groups:
        parameter_group.sync_model_weight_from_main_weight()

    output = model(torch.ones(1, 4, device=device, dtype=torch.bfloat16))
    torch.testing.assert_close(output, torch.full_like(output, 96.0))


def test_non_leaf_parameter_view_survives_storage_resize(distributed_setup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NonLeafViewModel().to(device)
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    group = model.parameter_groups[0]
    x = torch.randn(8, device=device, requires_grad=True)
    loss = model(x).sum()

    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0

    loss.backward()

    assert group.main_grad is not None
    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0
