# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantized training tests for the experimental Megatron-FSDP path."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint
from transformer_engine.pytorch.optimizers import FusedAdam
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import BlockAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
)


def _make_mlp(device):
    kwargs = {"params_dtype": torch.bfloat16, "device": device}
    return nn.Sequential(te.Linear(64, 128, **kwargs), nn.GELU(), te.Linear(128, 32, **kwargs))


class _RecomputedMLP(nn.Module):
    """Two-layer MLP that optionally recomputes each linear layer in backward."""

    def __init__(self, device, use_reentrant: bool | None) -> None:
        super().__init__()
        kwargs = {"params_dtype": torch.bfloat16, "device": device}
        self.fc1 = te.Linear(64, 128, **kwargs)
        self.fc2 = te.Linear(128, 32, **kwargs)
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP, checkpointing each linear layer unless use_reentrant is None."""
        if self.use_reentrant is None:
            return self.fc2(F.gelu(self.fc1(x)))
        # TE's checkpoint restores the MXFP8 autocast state during recomputation.
        x = te_checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)
        return te_checkpoint(self.fc2, F.gelu(x), use_reentrant=self.use_reentrant)


_ZERO3_PLACEMENTS = Placements(
    dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
)

requires_mxfp8 = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 requires Blackwell-or-newer CUDA hardware.",
)


@pytest.mark.launch_on_gb200
@requires_mxfp8
@pytest.mark.parametrize(
    "placements",
    [
        Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]),
        Placements(dp_axes=[0], parameter=[Replicate()], gradient=[Shard(0)], optimizer=[Shard(0)]),
        Placements(
            dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[Replicate()]
        ),
    ],
    ids=["zero3", "zero2", "no_shard"],
)
def test_mxfp8_mlp_training_matches_reference(distributed_setup, placements):
    """MXFP8 MLP losses track an independently trained unsharded model."""
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Distributed MXFP8 coverage requires at least two ranks.")

    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        torch.manual_seed(2026)
        model = _make_mlp(device)
        torch.manual_seed(2026)
        reference = _make_mlp(device)
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=placements)

    # MLP weights share one quantized group; biases use a regular DBuffer.
    [weight_group] = [
        g for g in model.parameter_groups if isinstance(g.model_weight, QuantizedDBuffer)
    ]
    assert len(weight_group.fsdp_parameters) == 2
    if isinstance(placements.optimizer[0], Shard):
        assert weight_group.main_weight.placements == (BlockAtomic(32),)
    [bias_group] = [g for g in model.parameter_groups if g is not weight_group]
    assert isinstance(bias_group.model_weight, DBuffer)

    optimizer = FusedAdam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    # The reference has MXFP8 weights, so Adam needs separate FP32 weights for updates.
    # MFSDP doesn't need master_weights=True because it already passes FP32 master
    # weights to the optimizer.
    reference_optimizer = FusedAdam(reference.parameters(), lr=0.01, master_weights=True)
    # Match MFSDP's preserved initialization instead of dequantizing MXFP8 weights.
    for parameter in reference.parameters():
        # FP32 reconstruction from 16-bit remainders requires BF16 weight bits.
        # MXFP8 quantization loses some of those bits, so store full FP32 masters.
        reference_optimizer.initialize_state(parameter, store_param_remainders=False)
        if isinstance(parameter, MXFP8Tensor):
            reference_optimizer.set_scaled_state(
                parameter,
                "master_param",
                parameter.get_high_precision_init_val().to(device=device, dtype=torch.float32),
            )
            parameter.clear_high_precision_init_val()
    # Different rank inputs make an incorrect gradient reduction observable.
    torch.manual_seed(1234 + distributed_setup.rank)
    inputs = torch.randn(5, 32, 64, dtype=torch.bfloat16, device=device)
    targets = torch.randn(5, 32, 32, dtype=torch.bfloat16, device=device)

    def train(model, optimizer):
        losses = []
        for x, target in zip(inputs, targets):
            optimizer.zero_grad(set_to_none=True)
            with te.autocast(recipe=recipe):
                output = model(x)
            loss = (output.float() - target.float()).square().mean()
            losses.append(loss.detach())
            loss.backward()

            # The unwrapped reference needs explicit data-parallel averaging.
            if model is reference:
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)

            optimizer.step()
        return losses

    reference_losses = train(reference, reference_optimizer)
    sharded_losses = train(model, optimizer)
    # BF16 reduction order can make the independent optimizer updates differ.
    torch.testing.assert_close(
        torch.stack(sharded_losses), torch.stack(reference_losses), rtol=0, atol=3e-3
    )


@pytest.mark.launch_on_gb200
@requires_mxfp8
def test_mxfp8_unshard_gathers_rowwise_for_forward_and_columnwise_for_backward(distributed_setup):
    """Forward unshards only rowwise MXFP8 planes and backward only columnwise planes."""
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Distributed MXFP8 coverage requires at least two ranks.")

    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        model = _make_mlp(device)
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_ZERO3_PLACEMENTS)
    [weight_group] = [
        g for g in model.parameter_groups if isinstance(g.model_weight, QuantizedDBuffer)
    ]
    unsharded = weight_group._unsharded_model_weight
    rowwise_planes = unsharded.planes_for(rowwise=True, columnwise=False)
    columnwise_planes = unsharded.planes_for(rowwise=False, columnwise=True)

    def is_allocated(planes):
        return all(plane.local_buffer.untyped_storage().nbytes() > 0 for plane in planes)

    def is_released(planes):
        return all(plane.local_buffer.untyped_storage().nbytes() == 0 for plane in planes)

    first_linear = model[0]
    checked_phases = []

    def check_backward(_grad):
        # Runs after the root's pre-backward unshard and before this layer's
        # data-gradient GEMM, which reads columnwise planes.
        assert is_released(rowwise_planes)
        assert is_allocated(columnwise_planes)
        assert first_linear.weight._rowwise_data is None
        assert first_linear.weight._columnwise_data is not None
        checked_phases.append("backward")

    def check_forward(module, _args, output):
        # Forward GEMMs read rowwise planes.
        assert is_allocated(rowwise_planes)
        assert is_released(columnwise_planes)
        assert module.weight._rowwise_data is not None
        assert module.weight._columnwise_data is None
        checked_phases.append("forward")
        output.register_hook(check_backward)

    first_linear.register_forward_hook(check_forward)

    x = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    with te.autocast(recipe=recipe):
        output = model(x)
    output.float().square().mean().backward()

    assert checked_phases == ["forward", "backward"]
    assert is_released(unsharded.planes)


@pytest.mark.launch_on_gb200
@requires_mxfp8
@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_mxfp8_activation_recompute_matches_no_recompute(distributed_setup, use_reentrant):
    """Recomputation regathers rowwise planes in backward without changing numerics.

    Backward prefetch gathers only columnwise planes, so recomputed forwards must
    gather rowwise planes on demand. Shapes need padded scales, which covers
    reinstalling padded scale copies on the Parameter TE saved for backward.
    """
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Distributed MXFP8 coverage requires at least two ranks.")

    recipe = MXFP8BlockScaling()
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    recomputed_forwards = []

    def record_recompute(module, _args):
        # A forward running inside backward is an activation recomputation.
        if torch._C._current_graph_task_id() != -1:
            recomputed_forwards.append(module)

    def build(use_reentrant):
        with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
            torch.manual_seed(2026)
            model = _RecomputedMLP(device, use_reentrant)
        with fully_shard_context(device=device):
            fully_shard(model.fc1, mesh=mesh, placements=_ZERO3_PLACEMENTS)
            fully_shard(model.fc2, mesh=mesh, placements=_ZERO3_PLACEMENTS)
            fully_shard(model, mesh=mesh, placements=_ZERO3_PLACEMENTS)
        model.fc1.register_forward_pre_hook(record_recompute)
        model.fc2.register_forward_pre_hook(record_recompute)
        optimizer = FusedAdam(model.parameters(), lr=0.01)
        fully_shard_optimizer(optimizer)
        return model, optimizer

    torch.manual_seed(1234 + distributed_setup.rank)
    inputs = torch.randn(3, 32, 64, dtype=torch.bfloat16, device=device)

    def train(model, optimizer):
        """Return per-step losses and sharded gradients, and the final sharded weights."""
        losses, grads = [], []
        for x in inputs:
            optimizer.zero_grad(set_to_none=True)
            # Reentrant checkpointing produces gradients only for inputs requiring grad.
            x = x.clone().requires_grad_()
            with te.autocast(recipe=recipe):
                output = model(x)
            loss = output.float().square().mean()
            loss.backward()
            losses.append(loss.detach())
            grads.append([parameter.grad.to_local().clone() for parameter in model.parameters()])
            optimizer.step()
        weights = [parameter.to_local().clone() for parameter in model.parameters()]
        return losses, grads, weights

    expected_losses, expected_grads, expected_weights = train(*build(None))
    assert not recomputed_forwards
    losses, grads, weights = train(*build(use_reentrant))
    # fc1 and fc2 each recompute once per step.
    assert len(recomputed_forwards) == 2 * len(inputs)
    torch.testing.assert_close(losses, expected_losses, rtol=0, atol=0)
    torch.testing.assert_close(grads, expected_grads, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
