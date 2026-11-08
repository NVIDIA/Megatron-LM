# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient and update parity across independently executed GraphTasks."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    RuntimePolicy,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def _sharded_unit(setup, model, grad_accumulation_count):
    """Shard one unit with caller-managed execution and gradient synchronization."""
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(
        FsdpContext(
            device=setup.device, runtime_policy=RuntimePolicy(caller_managed_grad_sync=True)
        )
    ):
        fully_shard(
            model,
            mesh=mesh,
            placements=placements,
            schedule_policy=SchedulePolicy(
                register_hooks=False, grad_accumulation_count=grad_accumulation_count
            ),
        )
    # The test drives materialization around each GraphTask; the completion hook
    # only reduces gradients once all declared contributions have accumulated.
    model.register_post_backward_hook(FsdpModule._reduce_gradient_groups)
    return model


def _graph_task(model, inputs, node):
    """Materialize one unit for an independently executed forward/backward node."""
    model.context.allgather_stream.wait_stream(model.context.current_stream())
    model.unshard()
    consumer = model if node is None else model[node]
    loss = consumer(inputs).square().mean()
    loss.backward()
    model.reshard()
    return loss.detach()


def _gather_parameter_shards(tensor):
    """Reconstruct a parameter whose rows are distributed by the packed layout."""
    # Packed FSDP shards can own unequal row counts, including empty shards.
    # DTensor.full_tensor() assumes ordinary per-parameter Shard(0) partitions.
    shards = [None] * dist.get_world_size()
    dist.all_gather_object(shards, tensor.to_local().detach().cpu())
    return torch.cat(shards, dim=0).to(tensor.device)


@pytest.mark.parametrize(
    "nodes",
    [(None,), (None, None), (None, None, None), (0, 1, 0)],
    ids=["one_graph_task", "two_graph_tasks", "three_graph_tasks", "unequal_multiplicities"],
)
def test_gradients_and_updates_match_baseline(distributed_setup, nodes):
    """Losses, public gradients, and updated weights match dense training."""
    setup = distributed_setup
    torch.manual_seed(0)
    baseline = nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 16, bias=False)).to(
        setup.device
    )
    model = _sharded_unit(
        setup, copy.deepcopy(baseline), sum(2 if node is None else 1 for node in nodes)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)

    torch.manual_seed(123 + setup.rank)
    for _ in range(2):
        optimizer.zero_grad()
        baseline_optimizer.zero_grad()
        # Two windows per step exercise countdown re-arming and gradient accumulation.
        for _ in range(2):
            for node in nodes:
                inputs = torch.randn(4, 16, device=setup.device)
                consumer = baseline if node is None else baseline[node]
                baseline_loss = consumer(inputs).square().mean()
                baseline_loss.backward()
                torch.testing.assert_close(_graph_task(model, inputs, node), baseline_loss.detach())
        model.context.finish_grad_sync()
        for reference, parameter in zip(baseline.parameters(), model.parameters()):
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
            torch.testing.assert_close(_gather_parameter_shards(parameter.grad), reference.grad)
        optimizer.step()
        baseline_optimizer.step()
        for reference, parameter in zip(baseline.parameters(), model.parameters()):
            torch.testing.assert_close(_gather_parameter_shards(parameter), reference)
