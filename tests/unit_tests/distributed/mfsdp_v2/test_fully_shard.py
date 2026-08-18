# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Partial,
    Placements,
    Replicate,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import collect_linked_kernels

logger = logging.getLogger(__name__)


class TinyModel(nn.Module):
    """Small model with two separately shardable modules."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(self.relu(self.fc1(x)))


class CheckpointedTinyModel(TinyModel):
    """Tiny model that activation-checkpoints each shardable module."""

    def __init__(self, use_reentrant: bool) -> None:
        super().__init__()
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each linear layer through activation checkpointing."""
        x = checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)
        return checkpoint(self.fc2, self.relu(x), use_reentrant=self.use_reentrant)












def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])






@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_fully_shard_sgd_losses_match_baseline(distributed_setup, num_microbatches):
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

    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, 8, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, 4, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer, log_prefix) -> list[torch.Tensor]:
        losses = []
        for step in range(5):
            optimizer.zero_grad()

            for microbatch, (microbatch_x, microbatch_target) in enumerate(microbatches):
                loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                losses.append(loss.detach())
                logger.debug(
                    "%s train parity: rank=%s, step=%s, microbatch=%s, loss=%s",
                    log_prefix,
                    rank,
                    step,
                    microbatch,
                    loss,
                )

                (loss / num_microbatches).backward()

            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, "Baseline")
    sharded_losses = train(model, optimizer, "FSDP")

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        msg="Sharded losses did not match baseline losses.",
    )


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_fully_shard_activation_recompute_reshards_parameters(distributed_setup, use_reentrant):
    """Activation recomputation should leave every FSDP module resharded.

    Backward completes ``fc2`` before recomputing ``fc1``. Without suppressing
    forward prefetch during recomputation, ``fc1`` unshards ``fc2`` again after
    its backward hook has run, leaving ``fc2.weight`` as an unsharded Parameter
    instead of a sharded DTensor at the end of backward.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = CheckpointedTinyModel(use_reentrant=use_reentrant).to(device)
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.randn(2, 8, device=device, requires_grad=True)
    model(x).sum().backward()

    # Without the forward-prefetch suppression, ``fc1``'s recomputed forward
    # would unshard ``fc2`` after ``fc2``'s backward already resharded it,
    # leaving an unsharded Parameter here.
    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)

    # Backward completes each module before recomputing the previous one, so
    # every module-local phase must be cleared after its matching backward.
    assert model.phase is FsdpModule.Phase.RESTING
    assert model.fc1.phase is FsdpModule.Phase.RESTING
    assert model.fc2.phase is FsdpModule.Phase.RESTING

    # A second forward after backward runs in the forward phase again, so
    # forward-order prefetch resumes and the module phases return to resting.
    model(x).sum().backward()
    assert model.phase is FsdpModule.Phase.RESTING
    assert model.fc1.phase is FsdpModule.Phase.RESTING
    assert model.fc2.phase is FsdpModule.Phase.RESTING


















def test_backward_averages_across_dp_and_accumulates_across_calls(distributed_setup):
    """Each backward averages over DP ranks; repeated backwards accumulate by summing."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(1, world_size, bias=False).to(device)
    nn.init.constant_(model.weight, 1.0)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.full((1, 1), float(rank + 1), device=device)
    model(x).sum().backward()
    model(x).sum().backward()

    assert isinstance(model.weight.grad, DTensor)
    local_grad = model.weight.grad.to_local()
    expected = torch.full_like(local_grad, float(world_size + 1))
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)






def test_fully_shard_adam_mixed_precision_losses_match_baseline(distributed_setup):
    """Mixed-precision FSDP Adam should track an unsharded Adam baseline."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    baseline = TinyModel().to(device=device, dtype=torch.bfloat16)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(baseline.state_dict())
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())

    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)

    x = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
    target = torch.randn(3, 4, device=device, dtype=torch.bfloat16)

    for _ in range(3):
        baseline_optimizer.zero_grad()
        optimizer.zero_grad()

        baseline_loss = torch.nn.functional.mse_loss(baseline(x).float(), target.float())
        loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
        torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=3e-3)

        baseline_loss.backward()
        loss.backward()
        baseline_optimizer.step()
        optimizer.step()
