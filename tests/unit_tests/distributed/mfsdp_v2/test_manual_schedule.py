# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss, gradient, and update parity for explicitly scheduled MFSDP training.

The caller materializes weights and submits completed gradients without execution
or gradient-completion hooks, with both per-parameter and packed grouping.
"""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    ParameterGrouping,
    Placements,
    RuntimePolicy,
    SchedulePolicy,
    ShardLayout,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)


@pytest.mark.parametrize(
    "parameter_grouping, shard_layout, mesh_ndim",
    [
        pytest.param(ParameterGrouping.PER_PARAMETER, ShardLayout.PADDED, 1, id="per_parameter"),
        pytest.param(ParameterGrouping.PACKED, ShardLayout.PADDED, 1, id="packed"),
        pytest.param(
            ParameterGrouping.PER_PARAMETER, ShardLayout.UNPADDED, 1, id="per_parameter_unpadded"
        ),
        pytest.param(ParameterGrouping.PER_PARAMETER, ShardLayout.PADDED, 2, id="per_parameter_2d"),
    ],
)
def test_manual_schedule_matches_dense_training(
    distributed_setup, parameter_grouping, shard_layout, mesh_ndim
):
    """Three momentum-SGD steps match dense training on rank-local inputs."""
    setup = distributed_setup
    device = setup.device
    if mesh_ndim == 2 and (setup.world_size < 2 or setup.world_size % 2):
        pytest.skip("2-D mesh coverage requires an even number of ranks.")
    mesh_shape = (setup.world_size,) if mesh_ndim == 1 else (2, setup.world_size // 2)
    mesh = init_device_mesh(device.type, mesh_shape)
    torch.manual_seed(42)
    baseline = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3)).to(device)
    model = copy.deepcopy(baseline)
    placement = Shard(0)
    placements = Placements(list(range(mesh_ndim)), *[[placement] * mesh_ndim for _ in range(3)])
    if mesh_ndim == 2:
        placements = Placements(
            [0, 1], [Replicate(), placement], [Partial("avg"), placement], [Replicate(), placement]
        )
    context = FsdpContext(
        device=device, runtime_policy=RuntimePolicy(caller_managed_grad_sync=True)
    )
    with fully_shard_context(context):
        fully_shard(
            model,
            mesh,
            placements,
            schedule_policy=SchedulePolicy(register_hooks=False),
            shard_layout=shard_layout,
            parameter_grouping=parameter_grouping,
        )
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

        for reference in baseline.parameters():
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
        # MFSDP shard geometry need not match DTensor's ordinary chunk geometry.
        references = dict(baseline.named_parameters())
        for group in model.parameter_groups:
            full_grad = group.pre_optimizer_main_grad.redistribute([Replicate()] * mesh_ndim)
            for index, parameter in enumerate(group.fsdp_parameters):
                torch.testing.assert_close(
                    full_grad.get_tensor_view(index), references[parameter.fqns[0]].grad
                )
        optimizer.step()
        reference_optimizer.step()

        context.allgather_stream.wait_stream(context.current_stream())
        model.unshard()
        for reference, parameter in zip(baseline.parameters(), model.parameters()):
            torch.testing.assert_close(parameter, reference)
        model.reshard()
