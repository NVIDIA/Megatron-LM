# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient accumulation and optimizer-update parity with unsharded training."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    RuntimePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


@pytest.mark.parametrize("sharding", ["replicated", "hsdp", "hfsdp"])
@pytest.mark.parametrize("set_to_none", [False, True])
def test_local_accumulation_matches_reference(distributed_setup, sharding, set_to_none):
    """Singleton-inner DP matches dense training, with replicated DP as a reduction control."""
    setup = distributed_setup
    if setup.world_size < 2:
        pytest.skip("Requires different contributions from at least two ranks.")
    if sharding == "replicated":
        mesh = init_device_mesh(setup.device.type, (setup.world_size,))
        placements = Placements([0], [Replicate()], [Replicate()], [Replicate()])
        grad_comm_dtype = None
    else:
        mesh = init_device_mesh(setup.device.type, (setup.world_size, 1))
        placements = Placements(
            [0, 1],
            [Replicate(), Shard(0)],
            [Partial("avg"), Shard(0)],
            [Shard(0) if sharding == "hfsdp" else Replicate(), Shard(0)],
        )
        # No inner communication: a BF16 communication policy must not round local sums.
        grad_comm_dtype = torch.bfloat16

    torch.manual_seed(1234)
    model = nn.Linear(
        8, 8 * setup.world_size, bias=False, device=setup.device, dtype=torch.bfloat16
    )
    reference = nn.Linear(8, 8 * setup.world_size, bias=False, device=setup.device)
    reference.load_state_dict(model.state_dict())
    scale = 1 / 3
    context = FsdpContext(
        device=setup.device, runtime_policy=RuntimePolicy(overlap_outer_communication=True)
    )
    with fully_shard_context(context):
        fully_shard(
            model,
            mesh,
            placements,
            mixed_precision_policy=MixedPrecisionPolicy(
                main_grads_dtype=torch.float32, grad_comm_dtype=grad_comm_dtype
            ),
            grad_scale=scale,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    generator = torch.Generator(device=setup.device).manual_seed(5678 + setup.rank)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=set_to_none)
        expected_grad = torch.zeros_like(reference.weight)
        for index in range(2):
            inputs = torch.randn(
                4, 8, device=setup.device, dtype=torch.bfloat16, generator=generator
            )
            reference_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(setup.device.type, dtype=torch.bfloat16):
                expected_output = reference(inputs)
            expected_loss = expected_output.float().square().mean()
            expected_loss.backward()
            expected_grad.add_(reference.weight.grad * scale)

            with microbatch(context, is_last=index == 1):
                loss = model(inputs).float().square().mean()
                loss.backward()
            torch.testing.assert_close(loss, expected_loss)

        context.finish_grad_sync()
        dist.all_reduce(expected_grad, op=dist.ReduceOp.AVG)
        actual_grad = model.weight.grad.full_tensor()
        torch.testing.assert_close(actual_grad, expected_grad)
        reference.weight.grad = expected_grad
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.weight.full_tensor(), reference.weight)
