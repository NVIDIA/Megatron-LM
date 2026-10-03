# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Conditionally used trainable weights with mixed text-only and text-vision microbatches."""

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


class ConditionalModel(nn.Module):
    """Always-executed units with uniform parameter usage across DP ranks."""

    def __init__(self, num_units, recompute):
        super().__init__()
        self.layers = nn.ModuleList(ConditionalUnit() for _ in range(num_units))
        self.head = nn.Linear(8, 8)
        self.recompute = recompute

    def forward(self, hidden, images):
        """Run every unit, selecting its optional path separately."""
        for unit, image in zip(self.layers, images):
            if self.recompute is None:
                hidden = unit(hidden, image)
            else:
                hidden = checkpoint(unit, hidden, image, use_reentrant=self.recompute)
        return self.head(hidden)


@pytest.mark.parametrize(
    "num_units, shard_root",
    [(1, False), (2, False), (2, True)],
    ids=["single-unit", "two-units", "root-owner"],
)
@pytest.mark.parametrize(
    "usage, recompute",
    [
        (usage, recompute)
        for usage in ("mixed", "never", "alternating", "frozen_text")
        for recompute in (None, False)
    ]
    + [("always", True)],
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
                fully_shard(unit, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)

    torch.manual_seed(4321 + rank)
    inputs = torch.randn(
        5, 4, 4, 8, device=device, requires_grad=recompute is True or usage == "frozen_text"
    )
    images = torch.randn_like(inputs)
    targets = torch.randn_like(inputs)
    # A validation forward must not change the next training countdown.
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs[0, 0], [images[0, 0]] * num_units),
            baseline(inputs[0, 0], [images[0, 0]] * num_units),
        )

    def train(model, inputs, images, targets, *, reduce_wgrad: bool) -> list[torch.Tensor]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        if not reduce_wgrad:
            fully_shard_optimizer(optimizer)
        losses = []
        for step, (step_inputs, step_images, step_targets) in enumerate(
            zip(inputs, images, targets)
        ):
            optimizer.zero_grad()
            for chunk, (hidden, image, target) in enumerate(
                zip(step_inputs, step_images, step_targets)
            ):
                if usage == "always":
                    unit_images = [image] * num_units
                elif usage == "mixed":
                    # Accumulate text, text+image, text, text+image before optimizer.step().
                    unit_images = [None if chunk % 2 == 0 else image] * num_units
                else:
                    unit_images = [
                        (
                            image
                            if usage not in ("never", "frozen_text") and (step + index) % 2 == 0
                            else None
                        )
                        for index in range(num_units)
                    ]
                loss = torch.nn.functional.mse_loss(model(hidden, unit_images), target)
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

    baseline_losses = train(baseline, inputs, images, targets, reduce_wgrad=True)
    sharded_losses = train(model, inputs, images, targets, reduce_wgrad=False)
    torch.testing.assert_close(torch.stack(sharded_losses), torch.stack(baseline_losses))
    # Expose permanently unused projections too, so skipping their weight decay fails.
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs[-1, 0], [images[-1, 0]] * num_units),
            baseline(inputs[-1, 0], [images[-1, 0]] * num_units),
        )


def test_reentrant_recompute_counts_visible_parameters(distributed_setup):
    """Recomputation exposes the unit's graph even though it runs during backward."""
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

    x = torch.randn(4, 8, device=device, requires_grad=True)
    sharded_x = x.detach().clone().requires_grad_()
    expected = baseline(x).square().mean()
    actual = checkpoint(model, sharded_x, use_reentrant=True).square().mean()
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sharded_x.grad, x.grad)
    assert model.phase is FsdpModule.Phase.RESTING
    chunks = chunk_metadata_by_fqn(model)
    for (name, reference), sharded in zip(baseline.named_parameters(), model.parameters()):
        grad = reference.grad if reference.grad is not None else torch.zeros_like(reference)
        dist.all_reduce(grad, op=dist.ReduceOp.AVG)
        chunk = chunks[name]
        torch.testing.assert_close(
            sharded.grad.to_local(), grad.narrow(0, chunk.offsets[0], chunk.sizes[0])
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


class SharedModel(nn.Module):
    """Reuse the same conditional layer at successive prediction depths."""

    def __init__(self, recompute):
        super().__init__()
        self.layer = ConditionalUnit()
        self.recompute = recompute

    def forward(self, hidden, image):
        """Both uses contribute to the same leaf gradient."""
        for optional_image in (None, image):
            if self.recompute is None:
                hidden = self.layer(hidden, optional_image)
            else:
                hidden = checkpoint(
                    self.layer, hidden, optional_image, use_reentrant=self.recompute
                )
        return hidden


@pytest.mark.parametrize("frozen_text", [False, True])
def test_used_gradients_finish_before_final_callback(distributed_setup, frozen_text):
    """An unused projection must not defer the unit's reduction to autograd completion."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    model = ConditionalUnit().to(device)
    if frozen_text:
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


@pytest.mark.parametrize("recompute", [None, False])
@pytest.mark.parametrize("frozen_text", [False, True])
def test_shared_layer_gradients_and_updates(distributed_setup, recompute, frozen_text):
    """Both shared-layer uses contribute before the countdown finishes."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    torch.manual_seed(1234)
    baseline = SharedModel(recompute).to(device)
    model = SharedModel(recompute).to(device)
    model.load_state_dict(baseline.state_dict())
    if frozen_text:
        baseline.layer.text.requires_grad_(False)
        model.layer.text.requires_grad_(False)
    with fully_shard_context(device=device):
        fully_shard(model.layer, mesh, placements)
        fully_shard(model, mesh, placements)
    reference_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    torch.manual_seed(4321 + distributed_setup.rank)
    for step in range(3):
        reference_optimizer.zero_grad()
        optimizer.zero_grad()
        x = torch.randn(4, 8, device=device, requires_grad=True)
        sharded_x = x.detach().clone().requires_grad_()
        image = torch.randn_like(x) if step % 2 and not frozen_text else None
        loss = baseline(x, image).square().mean()
        sharded_loss = model(sharded_x, image).square().mean()
        loss.backward()
        sharded_loss.backward()
        torch.testing.assert_close(sharded_loss, loss)
        torch.testing.assert_close(sharded_x.grad, x.grad)
        chunks = chunk_metadata_by_fqn(model)
        for (name, reference), sharded in zip(baseline.named_parameters(), model.parameters()):
            if not reference.requires_grad:
                continue
            if reference.grad is None:
                reference.grad = torch.zeros_like(reference)
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
            chunk = chunks[name]
            expected = reference.grad.narrow(0, chunk.offsets[0], chunk.sizes[0])
            torch.testing.assert_close(sharded.grad.to_local(), expected)
        assert model.layer.phase is FsdpModule.Phase.RESTING
        assert model.phase is FsdpModule.Phase.RESTING
        reference_optimizer.step()
        optimizer.step()


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


class DelayedUnit(nn.Module):
    """A used TE producer, an unused TE producer, and an unused ordinary weight."""

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
        self.unused_linear = te.Linear(
            16,
            16,
            bias=False,
            params_dtype=torch.float32,
            device=device,
            delay_wgrad_compute=True,
            fuse_wgrad_accumulation=False,
        )

    def forward(self, hidden):
        """Only one projection participates in forward."""
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
            fully_shard(unit, mesh=mesh, placements=placements)
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
                    assert unit.unused.grad is None
            # Corresponding microbatches use the same TE schedule on every rank.
            for unit in reversed(model):
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
