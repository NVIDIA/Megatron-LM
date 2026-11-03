# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check bounded gradient staging against a reference, with and without graph replay."""

from copy import deepcopy
from weakref import ref

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class MixedLayer(nn.Module):
    """Two dtype groups in one owner, both computing in FP32."""

    def __init__(self, device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(8, device=device) / 2)
        self.bias = nn.Parameter(torch.zeros(8, device=device, dtype=torch.bfloat16))

    def forward(self, x):
        """Round the compute bias to BF16 while retaining an FP32 master in the reference."""
        return nn.functional.linear(x, self.weight, self.bias.to(torch.bfloat16).float())


@pytest.mark.parametrize(
    "overlap_outer", [False, True], ids=["shared-reduction", "split-reduction"]
)
@pytest.mark.parametrize("strategy", ["optim", "optim_grads_params"])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_grad_buffer_lifetime_matches_reference(
    distributed_setup, monkeypatch, strategy, capture, overlap_outer
):
    """Exercise dtype groups, delayed local accumulation, allocator reuse, and final draining."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Sequential(*(MixedLayer(device) for _ in range(3)))
    reference = deepcopy(model).float()
    sharded = strategy == "optim_grads_params"
    placements = Placements(
        dp_axes=[0],
        parameter=[Shard(0) if sharded else Replicate()],
        gradient=[Shard(0) if sharded else Partial("avg")],
        optimizer=[Shard(0)],
    )
    with fully_shard_context(
        FsdpContext(
            device=device,
            prefetch_order=PrefetchOrder.RECORDED,
            runtime_policy=RuntimePolicy(
                caller_managed_grad_sync=True, overlap_outer_communication=overlap_outer
            ),
        )
    ) as context:
        for layer in model:
            fully_shard(
                layer,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=MixedPrecisionPolicy(
                    main_params_dtype=torch.float32,
                    main_grads_dtype=torch.float32,
                    grad_comm_dtype=torch.float32,
                ),
            )
        fully_shard(model, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    torch.manual_seed(73 + distributed_setup.rank)
    inputs = torch.randn(3, 4, 8, device=device)
    reduce_gradients = FsdpParameterGroup.reduce_inner_gradients

    def delayed_reduce(group, partial_grad):
        # ZeRO-1's intermediate microbatches consume this input without any NCCL call.
        torch.cuda._sleep(100_000)
        return reduce_gradients(group, partial_grad)

    monkeypatch.setattr(FsdpParameterGroup, "reduce_inner_gradients", delayed_reduce)

    def train():
        optimizer.zero_grad(set_to_none=False)
        losses = []
        for mb, x in enumerate(inputs):
            with microbatch(context, is_last=mb == len(inputs) - 1):
                loss = model(x).square().mean() / len(inputs)
                loss.backward()
            losses.append(loss.detach())
            assert len(context._pending_grad_buffers) == 2
            assert all(len(batch.buffers) == 2 for batch in context._pending_grad_buffers)
            # Reuse small compute-stream blocks while reduction inputs are still in flight.
            for _ in range(8):
                torch.empty(64, device=device).fill_(1.0e6)
        context.finish_grad_sync()
        assert not context._pending_grad_buffers
        optimizer.step()
        return torch.stack(losses)

    def check(actual_losses):
        reference_optimizer.zero_grad(set_to_none=False)
        expected_losses = []
        for x in inputs:
            loss = reference(x).square().mean() / len(inputs)
            loss.backward()
            expected_losses.append(loss.detach())
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, group=mesh.get_group())
        torch.testing.assert_close(
            actual_losses, torch.stack(expected_losses), rtol=1e-5, atol=1e-6
        )
        reference_optimizer.step()
        for layer, expected_layer in zip(model, reference):
            for group in layer.parameter_groups:
                for index, parameter in enumerate(group.fsdp_parameters):
                    (name,) = parameter.fqns
                    expected = getattr(expected_layer, name)
                    owned = group.main_weight._get_owned_range(index)
                    if owned is None:
                        continue
                    for actual, baseline in (
                        (parameter.sharded.grad.to_local(), expected.grad),
                        (group.main_weight.get_tensor_view(index), expected.detach()),
                    ):
                        expected_shard = baseline.flatten().narrow(
                            0, owned.tensor_relative_offset, owned.numel
                        )
                        torch.testing.assert_close(
                            actual.flatten(), expected_shard, rtol=1e-5, atol=1e-6
                        )

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    for _ in range(2):
        with torch.cuda.stream(stream):
            losses = train()
        torch.cuda.current_stream(device).wait_stream(stream)
        check(losses)
        stream.wait_stream(torch.cuda.current_stream(device))

    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            losses = train()
    for _ in range(3):
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            if capture:
                graph.replay()
            else:
                losses = train()
        torch.cuda.current_stream(device).wait_stream(stream)
        check(losses)


def test_grad_buffer_release_orders_reuse_on_allocation_stream(distributed_setup):
    """Finalizing on another stream must still protect the input's original allocator."""
    device = distributed_setup.device
    context = FsdpContext(device=device)
    allocation_stream = torch.cuda.Stream(device)
    allocation_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(allocation_stream):
        source = torch.full((4096,), 7.0, device=device)
        result = torch.empty_like(source)
    context.inner_reduction_stream.wait_stream(allocation_stream)
    with torch.cuda.stream(context.inner_reduction_stream):
        torch.cuda._sleep(100_000)
        result.copy_(source)
    context._retain_grad_buffers([source], allocation_stream)
    source_ref = ref(source)
    del source
    assert source_ref() is not None

    context.finish_grad_sync()
    assert not context._pending_grad_buffers
    assert source_ref() is None
    with torch.cuda.stream(allocation_stream):
        replacement = torch.empty_like(result)
        replacement.fill_(-9.0)
    torch.cuda.current_stream(device).wait_stream(allocation_stream)
    torch.testing.assert_close(result, torch.full_like(result, 7.0), rtol=0, atol=0)
    torch.testing.assert_close(replacement, torch.full_like(replacement, -9.0), rtol=0, atol=0)
