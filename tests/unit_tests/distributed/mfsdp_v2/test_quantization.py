# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantized training tests for the experimental Megatron-FSDP path."""

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from transformer_engine.common.recipe import MXFP8BlockScaling
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


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 requires Blackwell-or-newer CUDA hardware.",
)
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
