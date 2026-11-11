# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss, gradient, and update parity for explicitly scheduled MFSDP training.

The caller materializes weights and submits completed gradients without execution
or gradient-completion hooks, for both independent tensor and packed sharding.
"""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    PerTensor,
    Placements,
    RuntimePolicy,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)


@pytest.mark.parametrize("packed", [False, True], ids=["per_tensor", "packed"])
def test_manual_schedule_matches_dense_training(distributed_setup, packed):
    """Three momentum-SGD steps match dense training on rank-local inputs."""
    setup = distributed_setup
    device = setup.device
    mesh = init_device_mesh(device.type, (setup.world_size,))
    torch.manual_seed(42)
    baseline = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3)).to(device)
    model = copy.deepcopy(baseline)
    placement = Shard(0) if packed else PerTensor(Shard(0))
    placements = Placements([0], [placement], [placement], [placement])
    context = FsdpContext(
        device=device, runtime_policy=RuntimePolicy(caller_managed_grad_sync=True)
    )
    with fully_shard_context(context):
        fully_shard(model, mesh, placements, schedule_policy=SchedulePolicy(register_hooks=False))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05, momentum=0.9)
    torch.manual_seed(123 + dist.get_rank())
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(4, 7, device=device, requires_grad=True)
        reference_loss = baseline(inputs).square().mean()
        reference_loss.backward()

        # Materialize forward weights, then discard them between phases.
        context.allgather_stream.wait_stream(context.current_stream())
        model.unshard()
        loss = model(inputs.detach().requires_grad_()).square().mean()
        torch.testing.assert_close(loss, reference_loss)
        model.reshard()

        # All saved weight views must be live again before autograd uses them.
        model.unshard()
        loss.backward()
        # With register_hooks=False, the scheduler declares gradients complete.
        model.post_accumulate_grad()
        context.post_backward()
        context.finish_grad_sync()

        for reference, parameter in zip(baseline.parameters(), model.parameters()):
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
            # Packed shards need not follow DTensor's ordinary chunk geometry.
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(shards, parameter.grad.to_local().cpu())
            torch.testing.assert_close(torch.cat(shards).to(device), reference.grad)
        optimizer.step()
        reference_optimizer.step()

        context.allgather_stream.wait_stream(context.current_stream())
        model.unshard()
        for reference, parameter in zip(baseline.parameters(), model.parameters()):
            torch.testing.assert_close(parameter, reference)
        model.reshard()
