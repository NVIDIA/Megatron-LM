# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training parity when a sharded module is called more than once."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)


class CheckpointedBlock(nn.Module):
    """A layer whose normalization is recomputed in a nested backward task."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8, bias=False)
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Checkpoint the normalization before the linear operation."""
        return self.linear(checkpoint(self.norm, value, use_reentrant=True))


class RepeatedModule(nn.Module):
    """Reuse one layer across prediction depths, each contributing to the loss."""

    def __init__(self, checkpoint_mode: str) -> None:
        super().__init__()
        self.shared = (
            CheckpointedBlock()
            if checkpoint_mode == "nested_reentrant"
            else nn.Linear(8, 8, bias=False)
        )
        self.checkpoint_mode = checkpoint_mode

    def forward(self, value: torch.Tensor, num_uses: int) -> torch.Tensor:
        """Return every depth's output in one loss graph."""
        outputs = []
        for index in range(num_uses):
            if self.checkpoint_mode in ("none", "nested_reentrant") or (
                self.checkpoint_mode == "mixed" and index == 0
            ):
                value = self.shared(value)
            else:
                value = checkpoint(
                    self.shared, value, use_reentrant=self.checkpoint_mode != "non_reentrant"
                )
            value = value.tanh()
            outputs.append(value)
        return torch.stack(outputs)


@pytest.mark.parametrize(
    "checkpoint_mode", ["none", "reentrant", "non_reentrant", "mixed", "nested_reentrant"]
)
@pytest.mark.parametrize("wrap_root", [False, True])
def test_repeated_module_matches_unsharded(distributed_setup, checkpoint_mode, wrap_root):
    """Compare gradients and three SGD updates with two accumulated microbatches."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    torch.manual_seed(1234)
    reference = RepeatedModule(checkpoint_mode).to(device)
    model = copy.deepcopy(reference)
    mesh = init_device_mesh(device.type, (world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device):
        fully_shard(model.shared, mesh=mesh, placements=placements)
        if wrap_root:
            fully_shard(model, mesh=mesh, placements=placements)
    reference_parameters = dict(reference.named_parameters())
    reference_slices = {}
    for name, parameter in model.named_parameters():
        sizes = [None] * world_size
        dist.all_gather_object(sizes, parameter.to_local().shape[0])
        assert sum(sizes) == parameter.shape[0]
        offset = sum(sizes[:rank])
        reference_slices[name] = slice(offset, offset + sizes[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    for step in range(3):
        # Exercise state reset when a module changes between repeated and single use.
        num_uses = (2, 1, 3)[step]
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        for microbatch in range(2):
            torch.manual_seed(1000 + 100 * step + 10 * rank + microbatch)
            value = torch.randn(4, 8, device=device, requires_grad=True)
            reference_value = value.detach().clone().requires_grad_()
            target = torch.randn_like(value)
            loss = (model(value, num_uses) - target).square().mean()
            reference_loss = (reference(reference_value, num_uses) - target).square().mean()
            torch.testing.assert_close(loss, reference_loss)
            loss.backward()
            reference_loss.backward()
            torch.testing.assert_close(value.grad, reference_value.grad)

        for name, parameter in model.named_parameters():
            reference_grad = reference_parameters[name].grad
            dist.all_reduce(reference_grad)
            reference_grad.div_(world_size)
            torch.testing.assert_close(
                parameter.grad.to_local(), reference_grad[reference_slices[name]]
            )
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.to_local(), reference_parameters[name][reference_slices[name]]
            )
