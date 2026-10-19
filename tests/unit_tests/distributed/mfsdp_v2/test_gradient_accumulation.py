"""Local-gradient reuse across reduce-scatter windows."""

import copy
import math

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import collect_linked_event_groups


@pytest.mark.parametrize("sharding", ["fsdp", "hsdp", "hfsdp"])
@pytest.mark.parametrize(
    "interval,set_to_none,comm_dtype",
    [(1, True, None), (2, False, None), (8, True, None), (2, True, torch.bfloat16)],
)
def test_rs_reuse_matches_gradients_and_updates(
    distributed_setup, monkeypatch, sharding, interval, set_to_none, comm_dtype
):
    """Windows preserve rank-distinct gradient sums, scaling, and repeated SGD updates."""
    setup = distributed_setup
    if setup.world_size < 4 or setup.world_size % 2:
        pytest.skip("Requires an even number of at least four ranks.")
    if sharding == "fsdp":
        mesh = init_device_mesh(setup.device.type, (setup.world_size,))
        placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    else:
        mesh = init_device_mesh(setup.device.type, (2, setup.world_size // 2))
        placements = Placements(
            [0, 1],
            [Replicate(), Shard(0)],
            [Partial("avg"), Shard(0)],
            [Replicate() if sharding == "hsdp" else Shard(0), Shard(0)],
        )
    torch.manual_seed(2026)
    reference = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16)).to(setup.device)
    model = copy.deepcopy(reference)
    with fully_shard_context(device=setup.device) as context:
        for layer in (model[0], model[2]):
            fully_shard(
                layer,
                mesh=mesh,
                placements=placements,
                grad_scale=0.25,
                mixed_precision_policy=MixedPrecisionPolicy(
                    main_grads_dtype=torch.float32, grad_comm_dtype=comm_dtype
                ),
                schedule_policy=SchedulePolicy(reduce_scatter_interval=interval),
            )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    allocations = []
    reductions = []
    original_allocate = FsdpParameterGroup.allocate_partial_grad_buffer
    original_reduce = FsdpParameterGroup.reduce_partial_gradients

    def allocate(group, **kwargs):
        buffer = original_allocate(group, **kwargs)
        allocations.append(buffer.dtype)
        return buffer

    def reduce(group, buffer, **kwargs):
        reductions.append(buffer.dtype)
        return original_reduce(group, buffer, **kwargs)

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", allocate)
    monkeypatch.setattr(FsdpParameterGroup, "reduce_partial_gradients", reduce)
    torch.manual_seed(123 + setup.rank)
    tolerance = dict(atol=3e-4, rtol=2e-2) if comm_dtype else dict(atol=1e-6, rtol=1e-5)

    def check_buffers(gradients):
        for actual_layer, expected_layer in zip((model[0], model[2]), (reference[0], reference[2])):
            for group in actual_layer.parameter_groups:
                expected = DBuffer(
                    mesh,
                    [Replicate()] * mesh.ndim,
                    group.main_weight.layout,
                    torch.float32,
                    setup.device,
                )
                for index, parameter in enumerate(group.fsdp_parameters):
                    value = expected_layer.get_parameter(parameter.fqns[0])
                    expected.get_tensor_view(index).copy_(value.grad if gradients else value)
                expected_shard = expected.view(group.main_weight.placements)
                for index, parameter in enumerate(group.fsdp_parameters):
                    value = parameter.sharded.grad if gradients else parameter.sharded
                    torch.testing.assert_close(
                        value.to_local(), expected_shard.get_tensor_view(index), **tolerance
                    )

    for step in range(3):
        optimizer.zero_grad(set_to_none=set_to_none)
        reference_optimizer.zero_grad(set_to_none=set_to_none)
        for microbatch_index in range(5):
            inputs = torch.randn(4, 16, device=setup.device)
            reference(inputs).square().mean().backward()
            with microbatch(context, is_last=microbatch_index == 4):
                model(inputs).square().mean().backward()
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            parameter.grad.mul_(0.25)
        check_buffers(gradients=True)
        expected_reductions = (step + 1) * 2 * math.ceil(5 / interval)
        assert len(reductions) == len(allocations) == expected_reductions
        assert reductions == [comm_dtype or torch.float32] * expected_reductions
        assert allocations == [torch.float32] * expected_reductions
        assert all(not layer._pending_gradients for layer in (model[0], model[2]))
        optimizer.step()
        reference_optimizer.step()
        check_buffers(gradients=False)


def test_rs_reuse_waits_for_delayed_wgrad(distributed_setup):
    """Only completed delayed weight gradients advance a reduction window."""
    setup = distributed_setup
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    torch.manual_seed(2026)
    model = te.Linear(
        16,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        device=setup.device,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=False,
    )
    with fully_shard_context(device=setup.device, caller_managed_grad_sync=True) as context:
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements([0], [Shard(0)], [Shard(0)], [Shard(0)]),
            schedule_policy=SchedulePolicy(reduce_scatter_interval=2),
        )
    for microbatch_index in range(2):
        with microbatch(context, is_last=microbatch_index == 1):
            inputs = torch.ones(4, 16, device=setup.device, dtype=torch.bfloat16)
            model(inputs).float().sum().backward()
            assert model._gradient_reduce_count == microbatch_index
            assert model.weight.grad is None
            model.backward_dw()
            context.finish_grad_sync()
        if microbatch_index == 0:
            assert model.weight.grad is None
            assert len(model._pending_gradients) == 1
    assert not model._pending_gradients
    torch.testing.assert_close(
        model.weight.grad.to_local(), torch.full_like(model.weight.grad.to_local(), 8)
    )


def test_rs_reuse_full_iteration_graph_and_collectives(distributed_setup):
    """Two reductions replace five, and a complete accumulation step replays correctly."""
    setup = distributed_setup
    if setup.world_size < 2:
        pytest.skip("Requires at least two ranks for collective profiling.")
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    torch.manual_seed(2026)
    reference = nn.Linear(16, 16, bias=False, device=setup.device)
    model = copy.deepcopy(reference)
    with fully_shard_context(device=setup.device) as context:
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements([0], [Shard(0)], [Shard(0)], [Shard(0)]),
            schedule_policy=SchedulePolicy(reduce_scatter_interval=3),
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    inputs = torch.ones(4, 16, device=setup.device)

    def iteration(layer, opt):
        opt.zero_grad(set_to_none=False)
        for microbatch_index in range(5):
            with microbatch(context, is_last=microbatch_index == 4):
                loss = layer(inputs).square().mean()
                loss.backward()
        opt.step()
        return loss.detach()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            iteration(model, optimizer)
            iteration(reference, reference_optimizer)
    torch.cuda.current_stream().wait_stream(stream)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        iteration(model, optimizer)
        torch.cuda.synchronize()
    iteration(reference, reference_optimizer)
    assert len(collect_linked_event_groups(prof, "reduce_scatter")) == 2
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        captured_loss = iteration(model, optimizer)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
        reference_loss = iteration(reference, reference_optimizer)
        torch.testing.assert_close(captured_loss, reference_loss)
    assert not model._pending_gradients
