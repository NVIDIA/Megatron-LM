# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP optimizer behavior."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)


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


def test_fully_shard_adam_without_adapter_raises_precision_error(distributed_setup):
    """Raw Adam should fail on mixed-precision FSDP parameters without the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    target = torch.randn(6, 4, device=device, dtype=torch.bfloat16)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss.backward()

    with pytest.raises(RuntimeError, match="same device and the same dtype"):
        optimizer.step()
