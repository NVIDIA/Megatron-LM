# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import gc
import os
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    FsdpModule,
    ParameterGroup,
    Placements,
    Replicate,
    fully_shard,
)


@dataclass(frozen=True)
class DistributedSetup:
    """Per-rank distributed test setup."""

    rank: int
    world_size: int
    device: torch.device


@pytest.fixture(scope="module")
def setup() -> Iterator[DistributedSetup]:
    """Read torchrun rank state and set up this rank's local device."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    yield DistributedSetup(rank=rank, world_size=world_size, device=device)

    if dist.is_initialized():
        dist.destroy_process_group()


class TinyModel(nn.Module):
    """Small model with two separately shardable units."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(self.relu(self.fc1(x)))


class NestedModel(nn.Module):
    """Model with direct and child-owned parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(4))
        self.inner = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested model."""
        return self.inner(x) + self.bias


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
        return SaveNonLeafWeightView.apply(x, self.weight.view_as(self.weight))


class ConstantMetaModel(nn.Module):
    """Model whose meta parameter is initialized by reset_parameters()."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))

    def reset_parameters(self) -> None:
        """Initialize the weight to a deterministic value."""
        self.weight.fill_(3.0)


class MixedMetaRealModel(nn.Module):
    """Model with unsupported mixed meta and real direct parameters."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))
        self.bias = nn.Parameter(torch.ones(4, device=device))

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        self.weight.fill_(1.0)
        self.bias.zero_()


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _full_named_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for module_name, child in module.named_modules():
        if not isinstance(child, FsdpModule):
            continue
        prefix = f"{module_name}." if module_name else ""
        for group in child.parameter_groups():
            full_weight = group.main_weight.redistribute([Flat()]).allgather(0)
            for index, name in enumerate(group.sharded_parameters):
                result[f"{prefix}{name}"] = full_weight.get_tensor(index).detach().clone()
    return result


def _full_model_weight(group: ParameterGroup) -> torch.Tensor:
    return group.model_weight.redistribute([Flat()]).allgather(0).get_tensor(0)


@pytest.mark.distributed
def test_experimental_fully_shard_train_step_matches_baseline(setup: DistributedSetup):
    """A minimal per-module FSDP train step should match single-rank SGD."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    torch.manual_seed(1234)
    baseline = TinyModel().to(setup.device)
    model = TinyModel().to(setup.device)
    model.load_state_dict(baseline.state_dict())

    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    x = torch.randn(3, 8, device=setup.device)
    target = torch.randn(3, 4, device=setup.device)

    baseline_loss = torch.nn.functional.mse_loss(baseline(x), target)
    baseline_loss.backward()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.add_(parameter.grad, alpha=-0.05)

    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()

    sharded_params = _full_named_parameters(model)
    for name, expected in baseline.named_parameters():
        torch.testing.assert_close(sharded_params[name], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.distributed
def test_nested_fully_shard_excludes_child_owned_parameters(setup: DistributedSetup):
    """An outer FSDP unit owns direct parameters but not nested child-unit parameters."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = NestedModel().to(setup.device)

    fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    inner_names = [
        name for group in model.inner.parameter_groups() for name in group.sharded_parameters
    ]
    outer_names = [name for group in model.parameter_groups() for name in group.sharded_parameters]

    assert inner_names == ["weight"]
    assert outer_names == ["bias"]


@pytest.mark.distributed
def test_frozen_parameter_group_does_not_allocate_main_grad(setup: DistributedSetup):
    """A non-trainable parameter group should not allocate persistent main gradients."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(setup.device)
    model.weight.requires_grad_(False)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    assert not group.requires_grad
    assert group.main_grad is None


@pytest.mark.distributed
def test_default_main_buffer_dtypes_follow_policy_contract(setup: DistributedSetup):
    """Default main weights are FP32 while default main gradients match parameter dtype."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(4, 4, bias=False, dtype=torch.float64).to(setup.device)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    assert group.dtype is torch.float64
    assert group.main_weight.local_buffer.dtype is torch.float32
    assert group.main_grad is not None
    assert group.main_grad.local_buffer.dtype is torch.float64


@pytest.mark.distributed
def test_full_mesh_is_used_for_dbuffers(setup: DistributedSetup):
    """This version uses the full mesh for DBuffer storage."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (1, setup.world_size), mesh_dim_names=("tp", "dp"))
    placements = Placements(
        dp_axes=["tp", "dp"],
        parameter=[Replicate(), Flat()],
        gradient=[Replicate(), Flat()],
        optimizer=[Replicate(), Flat()],
    )
    model = nn.Linear(4, 4, bias=False).to(setup.device)

    fully_shard(model, mesh=mesh, placements=placements)

    (group,) = model.parameter_groups()
    assert group.mesh is mesh
    assert group.main_weight.mesh is mesh
    assert group.model_weight.mesh is mesh
    assert group.main_grad is not None
    assert group.main_grad.mesh is mesh
    assert isinstance(model.weight, DTensor)
    assert model.weight.device_mesh is mesh


@pytest.mark.distributed
def test_sharded_parameter_contract_uses_dtensors(setup: DistributedSetup):
    """Resting sharded parameters and gradients should be DTensors."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(setup.device)
    original_weight = model.weight

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    assert group.unsharded_parameters["weight"] is original_weight
    assert isinstance(model.weight, DTensor)
    assert isinstance(model.weight.data, DTensor)
    model(torch.randn(2, 4, device=setup.device)).sum().backward()
    assert isinstance(model.weight, DTensor)
    assert isinstance(model.weight.grad, DTensor)
    assert original_weight.grad is None


@pytest.mark.distributed
def test_backward_averages_across_dp_and_accumulates_across_calls(setup: DistributedSetup):
    """Each backward averages over DP ranks; repeated backwards accumulate by summing."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(1, setup.world_size, bias=False).to(setup.device)
    with torch.no_grad():
        model.weight.fill_(1.0)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.full((1, 1), float(setup.rank + 1), device=setup.device)
    model(x).sum().backward()
    model(x).sum().backward()

    assert isinstance(model.weight.grad, DTensor)
    local_grad = model.weight.grad.to_local()
    expected = torch.full_like(local_grad, float(setup.world_size + 1))
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)


@pytest.mark.distributed
def test_meta_parameters_initialize_with_reset_parameters(setup: DistributedSetup):
    """Meta parameters should be replaced by sharded DTensors and initialized in place."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = ConstantMetaModel()

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    assert isinstance(model.weight, DTensor)
    assert not model.weight.to_local().is_meta
    (group,) = model.parameter_groups()
    assert not group.model_weight.local_buffer.is_meta
    assert group.model_weight.local_buffer.numel() > 0
    full_weight = _full_model_weight(group)
    torch.testing.assert_close(full_weight, torch.full_like(full_weight, 3.0))


@pytest.mark.distributed
def test_model_weight_alias_is_rejected(setup: DistributedSetup):
    """Model weights cannot alias the unsharded parameter buffer."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Flat()], optimizer=[Flat()]
    )
    model = nn.Linear(4, 4, bias=False).to(setup.device)

    with pytest.raises(AssertionError, match="storage is resized and released"):
        fully_shard(model, mesh=mesh, placements=placements)


@pytest.mark.distributed
def test_meta_parameters_reject_mixed_direct_parameters(setup: DistributedSetup):
    """A module cannot mix meta and real direct parameters when reset is required."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MixedMetaRealModel(setup.device)

    with pytest.raises(ValueError, match="mixes meta and non-meta direct parameters"):
        fully_shard(model, mesh=mesh, placements=_flat_placements())


@pytest.mark.distributed
def test_non_leaf_parameter_view_survives_storage_resize(setup: DistributedSetup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if setup.device.type != "cuda":
        pytest.skip("Storage resize verification requires CUDA.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = NonLeafViewModel().to(setup.device)
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    group = model.parameter_groups()[0]
    x = torch.randn(8, device=setup.device, requires_grad=True)
    loss = model(x).sum()

    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0

    loss.backward()

    assert group.main_grad is not None
    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0


@pytest.mark.distributed
def test_experimental_fully_shard_reduces_peak_training_memory(setup: DistributedSetup):
    """Per-layer FSDP should reduce peak CUDA memory during a train step."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if setup.device.type != "cuda":
        pytest.skip("Peak memory verification requires CUDA.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    dim = 1024
    layers = 16
    batch = 8

    torch.manual_seed(4321)
    baseline = nn.Sequential(*[nn.Linear(dim, dim, bias=False) for _ in range(layers)]).to(
        setup.device
    )
    x = torch.randn(batch, dim, device=setup.device)
    torch.cuda.reset_peak_memory_stats(setup.device)
    baseline(x).sum().backward()
    torch.cuda.synchronize(setup.device)
    baseline_peak = torch.cuda.max_memory_allocated(setup.device)

    del baseline
    del x
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(4321)
    model = nn.Sequential(*[nn.Linear(dim, dim, bias=False) for _ in range(layers)]).to(
        setup.device
    )
    for layer in model:
        fully_shard(layer, mesh=mesh, placements=_flat_placements())
    gc.collect()
    torch.cuda.empty_cache()

    x = torch.randn(batch, dim, device=setup.device)
    torch.cuda.reset_peak_memory_stats(setup.device)
    model(x).sum().backward()
    torch.cuda.synchronize(setup.device)
    sharded_peak = torch.cuda.max_memory_allocated(setup.device)

    assert sharded_peak < baseline_peak
