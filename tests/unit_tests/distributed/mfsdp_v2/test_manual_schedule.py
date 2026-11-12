# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Loss, gradient, and update parity for parameter-group-scheduled MFSDP training.

One FSDP unit schedules two parameters at distinct compute stages, prefetching
later weights and reducing completed gradients without execution hooks.
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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    get_containing_parameter_group,
)


class MLP(nn.Module):
    """Two matrix parameters consumed at different stages, separated by tanh."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.weight2 = nn.Parameter(torch.empty(output_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, inputs):
        """Run the two projection stages without distributed scheduling."""
        hidden = torch.tanh(F.linear(inputs, self.weight1))
        return F.linear(hidden, self.weight2)


def prefetch_group(group, allgather_stream):
    """Start a gather without making compute wait for it."""
    with torch.cuda.stream(allgather_stream):
        group.unshard_parameters()


def reshard_group(group, allgather_stream):
    """Release this group's full weights after their last consumer."""
    allgather_stream.wait_stream(torch.cuda.current_stream(allgather_stream.device))
    with torch.cuda.stream(allgather_stream):
        group.reshard_parameters()


def reduce_group(group, reduction_stream):
    """Pack completed gradients and reduce on the separate reduction stream."""
    reduction_stream.wait_stream(torch.cuda.current_stream(reduction_stream.device))
    with torch.cuda.stream(reduction_stream):
        group.reduce_gradients()


class ScheduledMLPFunction(torch.autograd.Function):
    """Schedule parameter groups using tensor handles and shared FSDP context."""

    @staticmethod
    def forward(ctx, inputs, weight1, weight2, fsdp_context):
        """Compute stage one while stage two's weights are being gathered."""
        allgather_stream = fsdp_context.allgather_stream
        ready2 = allgather_stream.record_event()
        group1 = get_containing_parameter_group(weight1)
        group2 = get_containing_parameter_group(weight2)
        ctx.groups = (group1, group2)
        ctx.weights = (weight1, weight2)
        ctx.fsdp_context = fsdp_context
        hidden = torch.tanh(F.linear(inputs, weight1))
        reshard_group(group1, allgather_stream)
        # The first GEMM and activation did not wait for this later gather.
        fsdp_context.current_stream().wait_event(ready2)
        output = F.linear(hidden, weight2)
        ctx.save_for_backward(inputs, hidden)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Prefetch stage one while computing and reducing stage two's gradients."""
        group1, group2 = ctx.groups
        weight1, weight2 = ctx.weights
        inputs, hidden = ctx.saved_tensors
        allgather_stream = ctx.fsdp_context.allgather_stream
        reduction_stream = ctx.fsdp_context.inner_reduction_stream
        compute_stream = ctx.fsdp_context.current_stream()
        allgather_stream.wait_stream(compute_stream)
        prefetch_group(group2, allgather_stream)
        ready2 = allgather_stream.record_event()
        compute_stream.wait_event(ready2)
        prefetch_group(group1, allgather_stream)
        ready1 = allgather_stream.record_event()

        grad_hidden = grad_output.matmul(weight2)
        weight2.grad = grad_output.t().matmul(hidden)
        reduce_group(group2, reduction_stream)
        reshard_group(group2, allgather_stream)

        compute_stream.wait_event(ready1)
        grad_projected = grad_hidden * (1 - hidden.square())
        grad_input = grad_projected.matmul(weight1) if ctx.needs_input_grad[0] else None
        weight1.grad = grad_projected.t().matmul(inputs)
        reduce_group(group1, reduction_stream)
        reshard_group(group1, allgather_stream)
        # The group methods install optimizer gradients directly.
        return grad_input, None, None, None


class ScheduledMLP(MLP):
    """Pass parameter tensors and FSDP context to the scheduled computation."""

    def forward(self, inputs, fsdp_context):
        """Wait for the first matrix and prefetch the second before stage one."""
        allgather_stream = fsdp_context.allgather_stream
        group1 = get_containing_parameter_group(self.weight1)
        group2 = get_containing_parameter_group(self.weight2)
        compute_stream = fsdp_context.current_stream()
        allgather_stream.wait_stream(compute_stream)
        prefetch_group(group1, allgather_stream)
        ready1 = allgather_stream.record_event()
        compute_stream.wait_event(ready1)
        prefetch_group(group2, allgather_stream)
        output = ScheduledMLPFunction.apply(inputs, self.weight1, self.weight2, fsdp_context)
        reshard_group(group2, allgather_stream)
        return output


@pytest.mark.parametrize(
    "shard_layout, mesh_ndim",
    [
        pytest.param(ShardLayout.PADDED, 1, id="padded"),
        pytest.param(ShardLayout.UNPADDED, 1, id="unpadded"),
        pytest.param(ShardLayout.PADDED, 2, id="2d_hsdp"),
    ],
)
def test_manual_schedule_matches_dense_training(distributed_setup, shard_layout, mesh_ndim):
    """Three momentum-SGD steps match dense training on rank-local inputs."""
    setup = distributed_setup
    device = setup.device
    if mesh_ndim == 2 and (setup.world_size < 2 or setup.world_size % 2):
        pytest.skip("2-D mesh coverage requires an even number of ranks.")
    mesh_shape = (setup.world_size,) if mesh_ndim == 1 else (2, setup.world_size // 2)
    mesh = init_device_mesh(device.type, mesh_shape)
    torch.manual_seed(42)
    baseline = MLP(1024, 2048, 1024).to(device)
    model = ScheduledMLP(1024, 2048, 1024).to(device)
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
        fully_shard(
            model,
            mesh,
            placements,
            schedule_policy=SchedulePolicy(register_hooks=False),
            shard_layout=shard_layout,
            parameter_grouping=ParameterGrouping.PER_PARAMETER,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05, momentum=0.9)
    torch.manual_seed(123 + dist.get_rank())
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(512, 1024, device=device, requires_grad=step > 0)
        reference_loss = baseline(inputs).square().mean()
        reference_loss.backward()

        model_inputs = inputs.detach().requires_grad_(inputs.requires_grad)
        loss = model(model_inputs, context).square().mean()
        torch.testing.assert_close(loss, reference_loss)
        loss.backward()
        context.finish_grad_sync()
        if inputs.requires_grad:
            torch.testing.assert_close(model_inputs.grad, inputs.grad)

        for reference in baseline.parameters():
            dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
        # MFSDP shard geometry need not match DTensor's ordinary chunk geometry.
        references = dict(baseline.named_parameters())
        for group in model.parameter_groups:
            full_grad = group.pre_optimizer_main_grad.redistribute([Replicate()] * mesh_ndim)
            for index, parameter in enumerate(group.fsdp_parameters):
                reference = references[parameter.fqns[0]]
                torch.testing.assert_close(full_grad.get_tensor_view(index), reference.grad)
        optimizer.step()
        reference_optimizer.step()

        for group in model.parameter_groups:
            full_weight = group.main_weight.redistribute([Replicate()] * mesh_ndim)
            for index, parameter in enumerate(group.fsdp_parameters):
                reference = references[parameter.fqns[0]]
                torch.testing.assert_close(full_weight.get_tensor_view(index), reference)
