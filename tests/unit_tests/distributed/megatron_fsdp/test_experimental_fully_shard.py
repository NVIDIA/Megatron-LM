# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


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
        weight_view = self.weight.view_as(self.weight)
        assert self.weight.is_leaf
        assert not weight_view.is_leaf
        return SaveNonLeafWeightView.apply(x, weight_view)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MB"


def _count_weight_syncs(nvtx_ranges: list[str]) -> int:
    return sum(message == "sync_model_weight_from_main_weight" for message in nvtx_ranges)


def _make_unwrapped_parent_microbatch_model(
    device: torch.device, mesh
) -> tuple[nn.Sequential, tuple]:
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False, dtype=torch.bfloat16),
        nn.Linear(1, 1, bias=False, dtype=torch.bfloat16),
    ).to(device)
    with torch.no_grad():
        for layer in model:
            layer.weight.fill_(1.0)

    for layer in model:
        fully_shard(
            layer,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )

    assert not hasattr(model, "parameter_groups")
    groups = tuple(layer.parameter_groups()[0] for layer in model)
    for group in groups:
        assert group.main_weight is not group.model_weight
    return model, groups


def test_fully_shard_losses_match_baseline(distributed_setup):
    """Minimal per-module FSDP training should match single-rank SGD."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    model = TinyModel().to(device)
    model.load_state_dict(baseline.state_dict())

    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    x = torch.randn(3, 8, device=device)
    target = torch.randn(3, 4, device=device)

    for step in range(5):
        baseline_optimizer.zero_grad()
        optimizer.zero_grad()

        baseline_loss = torch.nn.functional.mse_loss(baseline(x), target)
        loss = torch.nn.functional.mse_loss(model(x), target)
        logger.info(
            "FSDP train parity: rank=%s, step=%s, baseline_loss=%s, sharded_loss=%s",
            rank,
            step,
            baseline_loss.item(),
            loss.item(),
        )
        torch.testing.assert_close(loss, baseline_loss, msg=f"Loss mismatch at step {step}.")

        baseline_loss.backward()
        loss.backward()
        baseline_optimizer.step()
        optimizer.step()


def test_nested_fully_shard_excludes_child_owned_parameters(distributed_setup):
    """An outer FSDP unit owns direct parameters but not nested child-unit parameters."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NestedModel().to(device)

    fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    inner_names = [
        name for group in model.inner.parameter_groups() for name in group.parameter_names
    ]
    outer_names = [name for group in model.parameter_groups() for name in group.parameter_names]

    assert inner_names == ["weight"]
    assert outer_names == ["bias"]


def test_frozen_parameter_group_does_not_allocate_main_grad(distributed_setup):
    """A non-trainable parameter group should not allocate persistent main gradients."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    model.weight.requires_grad_(False)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    assert not group.requires_grad
    assert group.main_grad is None


def test_backward_averages_across_dp_and_accumulates_across_calls(distributed_setup):
    """Each backward averages over DP ranks; repeated backwards accumulate by summing."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(1, world_size, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.full((1, 1), float(rank + 1), device=device)
    model(x).sum().backward()
    model(x).sum().backward()

    assert isinstance(model.weight.grad, DTensor)
    local_grad = model.weight.grad.to_local()
    expected = torch.full_like(local_grad, float(world_size + 1))
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)


def test_next_forward_uses_optimizer_updated_weights(distributed_setup):
    """The next forward should observe weights updated by the previous optimizer step."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(1, world_size, bias=False, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)

    fully_shard(
        model,
        mesh=mesh,
        placements=_flat_placements(),
        mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
    )
    # SGD's foreach/fused CUDA paths require matching parameter and gradient dtypes.
    # Use the scalar path to exercise FP32 main weights with default BF16 main grads.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
    x = torch.ones(1, 1, device=device, dtype=torch.bfloat16)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        return loss.detach().float()

    first_loss = train_iteration()
    second_loss = train_iteration()

    with pytest.raises(AssertionError):
        torch.testing.assert_close(second_loss, first_loss)


def test_microbatch_false_scopes_unwrapped_parent_child_contexts(distributed_setup):
    """An unwrapped parent can set child FSDP contexts to a non-first microbatch."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model, _groups = _make_unwrapped_parent_microbatch_model(device, mesh)

    with microbatch(model, is_first=False):
        contexts = tuple(layer._context for layer in model)
        assert all(context is not None for context in contexts)
        assert all(not context.is_first_microbatch for context in contexts)
    assert tuple(layer._context for layer in model) == contexts
    assert all(context.is_first_microbatch for context in contexts)


def test_microbatch_training_syncs_once_per_minibatch(distributed_setup, monkeypatch):
    """Training with microbatches syncs main weights once per FSDP unit per minibatch."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False, dtype=torch.bfloat16),
        nn.Linear(1, 1, bias=False, dtype=torch.bfloat16),
    ).to(device)
    with torch.no_grad():
        for layer in model:
            layer.weight.fill_(1.0)
    fully_shard(
        model,
        mesh=mesh,
        placements=_flat_placements(),
        mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
    )
    groups = model.parameter_groups()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)
    x_microbatches = (
        torch.ones(1, 1, device=device, dtype=torch.bfloat16),
        torch.full((1, 1), 2.0, device=device, dtype=torch.bfloat16),
    )
    nvtx_ranges: list[str] = []

    original_range_push = torch.cuda.nvtx.range_push

    def range_push_spy(message: str) -> None:
        nvtx_ranges.append(message)
        original_range_push(message)

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push_spy)

    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        nvtx_ranges.clear()
        for microbatch_index, x in enumerate(x_microbatches):
            with microbatch(model, is_first=microbatch_index == 0):
                loss = model(x).float().sum() / len(x_microbatches)
            assert _count_weight_syncs(nvtx_ranges) == len(groups), (
                "main weights should sync once per FSDP group on the first microbatch, "
                "and not again on later microbatches in the same minibatch"
            )
            loss.backward()
        assert _count_weight_syncs(nvtx_ranges) == len(groups), (
            "main weights should sync once per FSDP group over the full minibatch"
        )
        optimizer.step()


def test_cpu_initialized_parameters_shard_to_mesh_device(distributed_setup):
    """CPU-initialized parameters should be sharded with their real values."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    expected_weight = model.weight.detach().to(device)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    full_weight = group.model_weight.allgather(0).get_local_tensor(0)
    assert full_weight.device.type == device.type
    torch.testing.assert_close(full_weight, expected_weight)


def test_non_leaf_parameter_view_survives_storage_resize(distributed_setup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NonLeafViewModel().to(device)
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    group = model.parameter_groups()[0]
    x = torch.randn(8, device=device, requires_grad=True)
    loss = model(x).sum()

    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0

    loss.backward()

    assert group.main_grad is not None
    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0


def test_fully_shard_reduces_peak_training_memory(distributed_setup):
    """Per-layer FSDP should reduce peak CUDA memory during training."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if device.type != "cuda":
        pytest.skip("Peak memory verification requires CUDA.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 1024
    layers = 16
    batch = 8
    steps = 2
    dtype = torch.bfloat16

    def train_steps(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor) -> None:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model(x).sum().backward()
            optimizer.step()

    torch.manual_seed(4321)
    baseline = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(
        device
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(baseline, baseline_optimizer, x)
    torch.cuda.synchronize(device)
    baseline_peak = torch.cuda.max_memory_allocated(device)

    del baseline_optimizer
    del baseline
    del x
    torch.cuda.empty_cache()

    torch.manual_seed(4321)
    model = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(
        device
    )
    for layer in model:
        fully_shard(
            layer,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=MixedPrecisionPolicy(
                main_params_dtype=dtype, main_grads_dtype=dtype
            ),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    torch.cuda.empty_cache()

    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(model, optimizer, x)
    torch.cuda.synchronize(device)
    sharded_peak = torch.cuda.max_memory_allocated(device)
    logger.info(
        "FSDP peak memory: rank=%s, baseline=%s, sharded=%s",
        rank,
        _mb(baseline_peak),
        _mb(sharded_peak),
    )

    assert sharded_peak < baseline_peak
