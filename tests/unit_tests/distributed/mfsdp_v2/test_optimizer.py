# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP optimizer behavior."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


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


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_adam_without_adapter_raises_precision_error(distributed_setup):
    """Raw Adam should fail on mixed-precision FSDP parameters without the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="dtype"):
        optimizer.step()


def test_fused_adam_adapter_accepts_mismatched_grads(distributed_setup):
    """TE FusedAdam should handle mixed-precision FSDP grads through the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    # These are the defaults, but spell them out so the test clearly exercises
    # mismatched parameter and gradient precision.
    mixed_precision_policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32, main_grads_dtype=torch.bfloat16
    )
    with fully_shard_context(device=device):
        fully_shard(
            model.fc1,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=mixed_precision_policy,
        )
        fully_shard(
            model.fc2,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=mixed_precision_policy,
        )
    optimizer = FusedAdam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer, precision_aware=True)

    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.dtype != parameter.grad.dtype

    params_before_step = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()

    assert any(
        not torch.equal(parameter_before, parameter.detach())
        for parameter_before, parameter in zip(params_before_step, model.parameters())
    )


def test_next_forward_uses_optimizer_updated_weights(distributed_setup):
    """The next forward should observe weights updated by the previous optimizer step."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(1, world_size, bias=False, dtype=torch.bfloat16).to(device)
    nn.init.constant_(model.weight, 1.0)

    with fully_shard_context(device=device):
        fully_shard(
            model,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )
    # SGD's foreach/fused CUDA paths require matching parameter and gradient dtypes.
    # Use the scalar path to exercise FP32 main weights with default BF16 main grads.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
    fully_shard_optimizer(optimizer)
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


def test_optimizer_post_step_syncs_once_per_parameter_group(distributed_setup, monkeypatch):
    """Optimizer synchronization should run once per group, not once per microbatch."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    parameter_groups = (*model.fc1.parameter_groups, *model.fc2.parameter_groups)
    sync_counts = {parameter_group: 0 for parameter_group in parameter_groups}

    def make_count_sync(parameter_group):
        sync_model_weight = parameter_group.sync_model_weight_from_main_weight

        def count_sync():
            sync_counts[parameter_group] += 1
            sync_model_weight()

        return count_sync

    for parameter_group in parameter_groups:
        monkeypatch.setattr(
            parameter_group, "sync_model_weight_from_main_weight", make_count_sync(parameter_group)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    inputs = torch.randn(3, 2, 8, device=device, dtype=torch.bfloat16)

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for microbatch_input in inputs:
            (model(microbatch_input).sum() / len(inputs)).backward()

        assert all(sync_count == step for sync_count in sync_counts.values())
        optimizer.step()
        assert all(sync_count == step + 1 for sync_count in sync_counts.values())
