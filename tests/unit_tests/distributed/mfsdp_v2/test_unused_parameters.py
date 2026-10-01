# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unused trainable weights in MFSDP units, as in text-only multimodal batches."""

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


class ConditionalUnit(nn.Module):
    """A text path and an optional image projection owned by one FSDP unit."""

    def __init__(self):
        super().__init__()
        self.text = nn.Linear(8, 8)
        self.image = nn.Linear(8, 8)

    def forward(self, hidden, use_image):
        """The image path can be absent from an individual microbatch."""
        output = self.text(hidden)
        if use_image:
            output = output + self.image(hidden)
        return output.tanh()


class ConditionalModel(nn.Module):
    """Two always-executed units whose parameter usage can differ across ranks."""

    def __init__(self, num_units, recompute):
        super().__init__()
        self.layers = nn.ModuleList(ConditionalUnit() for _ in range(num_units))
        self.head = nn.Linear(8, 8)
        self.recompute = recompute

    def forward(self, hidden, use_image):
        """Run every unit, selecting its optional path separately."""
        for unit, enabled in zip(self.layers, use_image):
            if self.recompute is None:
                hidden = unit(hidden, enabled)
            else:
                hidden = checkpoint(unit, hidden, enabled, use_reentrant=self.recompute)
        return self.head(hidden)


@pytest.mark.parametrize(
    "num_units, shard_root",
    [(1, False), (2, False), (2, True)],
    ids=["single-unit", "two-units", "root-owner"],
)
@pytest.mark.parametrize("usage", ["never", "alternating", "rank_local", "frozen_text"])
@pytest.mark.parametrize(
    "recompute", [None, False, True], ids=["eager", "non_reentrant", "reentrant"]
)
def test_unused_parameters_match_zero_gradient_baseline(
    distributed_setup, num_units, shard_root, usage, recompute
):
    """Five AdamW steps exercise zero gradients, momentum, and changing usage."""
    device = distributed_setup.device
    rank, world_size = distributed_setup.rank, distributed_setup.world_size
    if world_size < 2:
        pytest.skip("Unused-parameter reduction requires at least two ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    torch.manual_seed(1234)
    baseline = ConditionalModel(num_units, recompute).to(device)
    model = ConditionalModel(num_units, recompute).to(device)
    model.load_state_dict(baseline.state_dict())
    if usage == "frozen_text":
        # Backward still traverses each unit, but none of its trainable weights are used.
        for candidate in (baseline, model):
            for unit in candidate.layers:
                unit.text.requires_grad_(False)
    with fully_shard_context(device=device):
        if shard_root:
            fully_shard(model.head, mesh=mesh, placements=placements)
        else:
            for unit in model.layers:
                fully_shard(unit, mesh=mesh, placements=placements, allow_unused_parameters=True)
        fully_shard(model, mesh=mesh, placements=placements, allow_unused_parameters=shard_root)

    torch.manual_seed(4321 + rank)
    inputs = torch.randn(
        5, 2, 4, 8, device=device, requires_grad=recompute is True or usage == "frozen_text"
    )
    targets = torch.randn_like(inputs)

    def train(model, inputs, targets, *, reduce_wgrad: bool) -> list[torch.Tensor]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        if not reduce_wgrad:
            fully_shard_optimizer(optimizer)
        losses = []
        for step, (step_inputs, step_targets) in enumerate(zip(inputs, targets)):
            optimizer.zero_grad()
            for chunk, (hidden, target) in enumerate(zip(step_inputs, step_targets)):
                use_image = [
                    usage not in ("never", "frozen_text")
                    and (step + index + (chunk + rank if usage == "rank_local" else 0)) % 2 == 0
                    for index in range(num_units)
                ]
                loss = torch.nn.functional.mse_loss(model(hidden, use_image), target)
                losses.append(loss.detach())
                (loss / len(step_inputs)).backward()
                if not reduce_wgrad:
                    for unit in model.modules():
                        if isinstance(unit, FsdpModule):
                            assert unit.phase is FsdpModule.Phase.RESTING
            if reduce_wgrad:
                # Match Megatron DDP's zero-filled buffers, including AdamW updates.
                for parameter in model.parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
        return losses

    baseline_losses = train(baseline, inputs, targets, reduce_wgrad=True)
    sharded_losses = train(model, inputs, targets, reduce_wgrad=False)
    torch.testing.assert_close(torch.stack(sharded_losses), torch.stack(baseline_losses))
    # Expose permanently unused projections too, so skipping their weight decay fails.
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs[-1, 0], [True] * num_units), baseline(inputs[-1, 0], [True] * num_units)
        )


class DelayedUnit(nn.Module):
    """One externally computed TE weight gradient and one unused autograd parameter."""

    def __init__(self, device):
        super().__init__()
        self.linear = te.Linear(
            16,
            16,
            bias=False,
            params_dtype=torch.float32,
            device=device,
            delay_wgrad_compute=True,
            fuse_wgrad_accumulation=False,
        )
        self.unused = nn.Parameter(torch.ones(16, device=device))

    def forward(self, hidden):
        """Run only the TE projection."""
        return self.linear(hidden).tanh()


@pytest.mark.parametrize("separate_backward", [False, True])
def test_unused_parameters_wait_for_delayed_te_gradients(distributed_setup, separate_backward):
    """Unused ordinary weights must not finish units with pending TE callbacks."""
    rank, world_size = distributed_setup.rank, distributed_setup.world_size
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    torch.manual_seed(1234)
    baseline = nn.Sequential(DelayedUnit(device), DelayedUnit(device))
    model = nn.Sequential(DelayedUnit(device), DelayedUnit(device))
    model.load_state_dict(baseline.state_dict())
    with fully_shard_context(device=device):
        for unit in model:
            fully_shard(unit, mesh=mesh, placements=placements, allow_unused_parameters=True)
    torch.manual_seed(4321 + rank)
    inputs = torch.randn(5, 4, 16, device=device, requires_grad=True)

    def train(model, inputs, *, reduce_wgrad: bool) -> list[torch.Tensor]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        if not reduce_wgrad:
            fully_shard_optimizer(optimizer)
        losses = []
        for hidden in inputs:
            optimizer.zero_grad()
            outputs = [unit(hidden) for unit in model] if separate_backward else [model(hidden)]
            for output in outputs:
                loss = output.square().mean()
                losses.append(loss.detach())
                loss.backward()
            if not reduce_wgrad:
                for unit in model:
                    assert unit.phase is FsdpModule.Phase.BACKWARD
                    assert unit.linear.weight.grad is None
                    torch.testing.assert_close(unit.unused.grad, torch.zeros_like(unit.unused))
            # Call TE in different orders across ranks. Collective order must still agree.
            for unit in (model if rank % 2 else reversed(model)):
                unit.linear.backward_dw()
            if reduce_wgrad:
                for parameter in model.parameters():
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            else:
                assert all(unit.phase is FsdpModule.Phase.RESTING for unit in model)
            optimizer.step()
        return losses

    baseline_losses = train(baseline, inputs, reduce_wgrad=True)
    sharded_losses = train(model, inputs, reduce_wgrad=False)
    torch.testing.assert_close(torch.stack(sharded_losses), torch.stack(baseline_losses))
