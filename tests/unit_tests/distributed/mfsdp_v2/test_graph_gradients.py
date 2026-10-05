# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for graph-derived gradient countdowns."""

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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.uneven_dtensor import (
    chunk_metadata_by_fqn,
)


class ConditionalUnit(nn.Module):
    """A text path and an optional image projection owned by one FSDP unit."""

    def __init__(self):
        super().__init__()
        self.text = nn.Linear(8, 8)
        self.image = nn.Linear(8, 8)

    def forward(self, hidden, image=None):
        """The image path can be absent from an individual microbatch."""
        output = self.text(hidden)
        if image is not None:
            output = output + self.image(image)
        return output.tanh()


@pytest.mark.parametrize("recompute", [None, False, True])
def test_unused_parameter_gradients_and_updates(distributed_setup, recompute):
    """Changing usage, shared calls, and checkpointing match a zero-filled baseline."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    torch.manual_seed(1234)
    baseline = ConditionalUnit().to(device)
    model = ConditionalUnit().to(device)
    model.load_state_dict(baseline.state_dict())
    with fully_shard_context(device=device):
        fully_shard(model, mesh, placements)
    reference_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    chunks = chunk_metadata_by_fqn(model)

    def forward(unit, hidden, image):
        if recompute is not None:
            return checkpoint(unit, hidden, image, use_reentrant=recompute)
        return unit(unit(hidden, image), image)

    torch.manual_seed(4321 + distributed_setup.rank)
    # Use, omit, then reuse the optional branch to catch stale gradients and momentum.
    for use_image in (True, False, True):
        reference_optimizer.zero_grad()
        optimizer.zero_grad()
        x = torch.randn(4, 8, device=device, requires_grad=True)
        sharded_x = x.detach().clone().requires_grad_()
        image = torch.randn_like(x) if use_image else None
        expected = forward(baseline, x, image).square().mean()
        actual = forward(model, sharded_x, image).square().mean()
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(sharded_x.grad, x.grad)
        assert model.phase is FsdpModule.Phase.RESTING
        for (name, reference), sharded in zip(baseline.named_parameters(), model.parameters()):
            if reference.grad is None:
                reference.grad = torch.zeros_like(reference)
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
            chunk = chunks[name]
            torch.testing.assert_close(
                sharded.grad.to_local(), reference.grad.narrow(0, chunk.offsets[0], chunk.sizes[0])
            )
        reference_optimizer.step()
        optimizer.step()
        for (name, reference), sharded in zip(baseline.named_parameters(), model.parameters()):
            chunk = chunks[name]
            torch.testing.assert_close(
                sharded.to_local(), reference.narrow(0, chunk.offsets[0], chunk.sizes[0])
            )


def test_parameter_count_stops_at_keyword_input(distributed_setup):
    """An upstream checkpoint must not make this unit fall back to its owned count."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    upstream = nn.Linear(8, 8).to(device)
    model = ConditionalUnit().to(device)
    with fully_shard_context(device=device):
        fully_shard(model, mesh, placements)

    x = torch.randn(4, 8, device=device, requires_grad=True)
    hidden = checkpoint(upstream, x, use_reentrant=True)
    # Keyword tensors are boundaries too. Walking past hidden would encounter
    # the checkpoint and incorrectly count the unused image weight and bias.
    output = model(hidden=hidden)
    assert model._trainable_parameter_countdown.initial_value == 2
    output.square().mean().backward()
    assert model.phase is FsdpModule.Phase.RESTING
    assert x.grad is not None
    for parameter in model.image.parameters():
        assert torch.count_nonzero(parameter.grad.to_local()).item() == 0


def test_zero_count_finishes_before_final_callback(distributed_setup):
    """An unused projection must not defer the unit's reduction to autograd completion."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    model = ConditionalUnit().to(device)
    model.text.requires_grad_(False)
    with fully_shard_context(device=device):
        fully_shard(model, mesh, placements)
    loss = model(torch.randn(4, 8, device=device, requires_grad=True)).square().mean()
    phases = []

    def queue_check(_grad):
        # Register ahead of FSDP's backward-pre hook and its final callback.
        torch.autograd.Variable._execution_engine.queue_callback(lambda: phases.append(model.phase))

    loss.register_hook(queue_check)
    loss.backward()
    assert phases == [FsdpModule.Phase.RESTING]


def test_zero_used_parameters_require_input_gradients(distributed_setup):
    """Apply the input-gradient guard even when owned trainable parameters are unused."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    trainable_linear = nn.Linear(4, 4, bias=False)
    frozen_linear = nn.Linear(4, 4, bias=False).requires_grad_(False)
    model = nn.Sequential(trainable_linear, frozen_linear).to(device)
    model.register_parameter("unused", nn.Parameter(torch.ones(4, device=device)))
    # The root owns an unused trainable parameter and a frozen weight. Its child
    # creates a backward graph despite the root's input not requiring gradients.
    with fully_shard_context(device=device):
        fully_shard(trainable_linear, mesh, placements)
        fully_shard(model, mesh, placements)

    def unpack(tensor):
        if tensor.untyped_storage().nbytes() == 0:
            pytest.fail("Backward read resharded weights", pytrace=False)
        return tensor

    # Preserve saved storage aliases and catch early release before CUDA reads it.
    with torch.autograd.graph.saved_tensors_hooks(
        pack_hook=lambda tensor: tensor, unpack_hook=unpack
    ):
        output = model(torch.ones(2, 4, device=device))
        with pytest.raises(NotImplementedError, match="MFSDP requires input gradients"):
            output.sum().backward()


def test_unused_parameters_wait_for_delayed_te_gradients(distributed_setup):
    """An unused weight must not prevent completion by TE's delayed-wgrad hook."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    model = te.Linear(
        16,
        16,
        bias=False,
        params_dtype=torch.float32,
        device=device,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=False,
    )
    model.register_parameter("unused", nn.Parameter(torch.ones(16, device=device)))
    with fully_shard_context(device=device):
        fully_shard(model, mesh, placements)
    model(torch.randn(4, 16, device=device, requires_grad=True)).sum().backward()
    assert model.phase is FsdpModule.Phase.BACKWARD
    assert model.weight.grad is None
    model.backward_dw()
    # Stream ordering is tested separately in #7854.
    torch.cuda.synchronize()
    assert model.phase is FsdpModule.Phase.RESTING
    assert model.weight.grad is not None
    assert torch.count_nonzero(model.unused.grad.to_local()).item() == 0
