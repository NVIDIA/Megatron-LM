# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss, gradient, and update parity for module-scheduled MFSDP training.

Each linear's custom autograd implementation materializes weights and submits
completed gradients without execution or gradient-completion hooks.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
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


class ScheduledLinearFunction(torch.autograd.Function):
    """Compute with materialized weights and drive their backward lifecycle."""

    @staticmethod
    def forward(ctx, inputs, weight, bias, module):
        """Run the forward GEMM with weights materialized by the module."""
        ctx.module = module
        ctx.save_for_backward(inputs)
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        """Gather for backward GEMMs, submit completed gradients, and reshard."""
        module = ctx.module
        (inputs,) = ctx.saved_tensors
        module.context.allgather_stream.wait_stream(module.context.current_stream())
        module.unshard()
        grad_input = grad_output.matmul(module.weight) if ctx.needs_input_grad[0] else None
        module.weight.grad = grad_output.t().matmul(inputs)
        module.bias.grad = grad_output.sum(0)
        module.post_accumulate_grad()
        module.reshard()
        # post_accumulate_grad() installs reduced gradients on optimizer parameters;
        # autograd must not accumulate another copy through their graph edges.
        return grad_input, None, None, None


class _ScheduledLinear(nn.Linear):
    """Linear whose autograd implementation explicitly drives its FSDP lifecycle."""

    def forward(self, inputs):
        """Gather, compute with full weights, and reshard without execution hooks."""
        self.context.allgather_stream.wait_stream(self.context.current_stream())
        self.unshard()
        output = ScheduledLinearFunction.apply(inputs, self.weight, self.bias, self)
        self.reshard()
        return output


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
    model = nn.Sequential(_ScheduledLinear(7, 5), nn.Tanh(), _ScheduledLinear(5, 3)).to(device)
    model.load_state_dict(baseline.state_dict())
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
        for layer_index in (0, 2):
            fully_shard(
                model[layer_index],
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
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(4, 7, device=device, requires_grad=step > 0)
        reference_loss = baseline(inputs).square().mean()
        reference_loss.backward()

        model_inputs = inputs.detach().requires_grad_(inputs.requires_grad)
        loss = model(model_inputs).square().mean()
        torch.testing.assert_close(loss, reference_loss)
        loss.backward()
        context.finish_grad_sync()
        if inputs.requires_grad:
            torch.testing.assert_close(model_inputs.grad, inputs.grad)

        for reference in baseline.parameters():
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
        # MFSDP shard geometry need not match DTensor's ordinary chunk geometry.
        references = dict(baseline.named_parameters())
        for layer_index in (0, 2):
            for group in model[layer_index].parameter_groups:
                full_grad = group.pre_optimizer_main_grad.redistribute([Replicate()] * mesh_ndim)
                for index, parameter in enumerate(group.fsdp_parameters):
                    reference = references[f"{layer_index}.{parameter.fqns[0]}"]
                    torch.testing.assert_close(full_grad.get_tensor_view(index), reference.grad)
        optimizer.step()
        reference_optimizer.step()

        for layer_index in (0, 2):
            for group in model[layer_index].parameter_groups:
                full_weight = group.main_weight.redistribute([Replicate()] * mesh_ndim)
                for index, parameter in enumerate(group.fsdp_parameters):
                    reference = references[f"{layer_index}.{parameter.fqns[0]}"]
                    torch.testing.assert_close(full_weight.get_tensor_view(index), reference)
