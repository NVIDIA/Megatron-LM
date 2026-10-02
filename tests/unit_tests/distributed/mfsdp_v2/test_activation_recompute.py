# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-unit activation recomputation of children with parent-owned parameters."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
)


class Unit(nn.Module):
    """Checkpoint a child whose weights normally belong to the parent FSDP unit."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor, manager: CheckpointWithoutOutputManager) -> torch.Tensor:
        """Checkpoint the norm, then compute with the parent's remaining weights."""
        x = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(self.norm, x)
        return self.linear(x.sin())


class Model(nn.Module):
    """Replay both units' norms from a hook on the final output."""

    def __init__(self) -> None:
        super().__init__()
        self.units = nn.ModuleList([Unit(), Unit()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replay earlier norms before their parent backward hooks are reached."""
        manager = CheckpointWithoutOutputManager()
        for unit in self.units:
            x = unit(x, manager)
        manager.discard_all_outputs_and_register_unified_recompute(x)
        return x.square().mean()


@pytest.mark.parametrize("shard_norm", [False, True], ids=["parent_only", "shard_norm"])
def test_direct_child_replay_matches_baseline(distributed_setup, shard_norm):
    """Direct child replay should match unsharded training for either unit boundary."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    torch.manual_seed(5678 + distributed_setup.rank)
    inputs = torch.randn(4, 8, device=device, requires_grad=True)

    def train(model, optimizer, inputs, *, reduce_wgrad: bool) -> list[torch.Tensor]:
        """Train five updates with explicit gradient averaging for the plain baseline."""
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            inputs.grad = None
            loss = model(inputs)
            losses.append(loss.detach())
            loss.backward()
            if reduce_wgrad:
                # The plain baseline has no DDP wrapper to average its gradients.
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
        return losses

    torch.manual_seed(1234)
    baseline = Model().to(device)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    baseline_losses = train(baseline, baseline_optimizer, inputs, reduce_wgrad=True)

    torch.manual_seed(1234)
    model = Model().to(device)
    with fully_shard_context(device=device):
        for unit in model.units:
            if shard_norm:
                fully_shard(unit.norm, mesh=mesh, placements=placements)
            fully_shard(unit, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)
    losses = train(model, optimizer, inputs, reduce_wgrad=False)

    torch.testing.assert_close(torch.stack(losses), torch.stack(baseline_losses))
