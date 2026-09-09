# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Communication-overlap tests for the minimal Megatron-FSDP path."""

from itertools import chain

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
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
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import (
    collect_linked_event_groups,
    event_groups_overlap,
)


class MultiChildModel(nn.Module):
    """Model with direct parameters and multiple child FsdpModules."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _zero1_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[Shard(0)]
    )


def _zero2_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Shard(0)], optimizer=[Shard(0)]
    )


# CPU ops that a device event chains up to via cpu_parent, used to attribute the device
# work to its enclosing collective or matmul operation.
_ALL_GATHER_OP_NAME_SUBSTRING = "allgather"
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"
_GEMM_OP_NAME_SUBSTRING = "aten::mm"


@pytest.mark.parametrize(
    "placements_factory",
    [
        pytest.param(_zero1_placements, id="zero1"),
        pytest.param(_zero2_placements, id="zero2"),
        pytest.param(_flat_placements, id="zero3"),
    ],
)
@pytest.mark.parametrize(
    "use_symmetric_memory",
    [
        # Both variants' all-gathers are launch-timing sensitive, but default-CTA
        # kernels occupy more compute CTAs, making the profiler overlap count less
        # stable across ranks (see
        # https://github.com/NVIDIA/Megatron-LM/actions/runs/31615942188).
        # The symmetric-memory variant uses zero-CTA all-gather kernels, so its overlap
        # measurement is more stable and remains enabled.
        pytest.param(False, marks=(pytest.mark.flaky, pytest.mark.flaky_in_dev)),
        pytest.param(True),
    ],
    ids=["default", "symmetric_memory"],
)
@pytest.mark.parametrize(
    "unify_communication_stream", [False, True], ids=["separate_streams", "unified_stream"]
)
def test_overlaps_communication_and_compute(
    distributed_setup, placements_factory, use_symmetric_memory, unify_communication_stream
):
    """ZeRO-1/2/3 communication should overlap GEMM compute."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    # A large hidden size keeps the per-layer GEMMs long enough that the
    # collectives reliably overlap them. The overlap count is otherwise
    # launch-bound: the host issues kernels with gaps (amplified by CI's
    # `coverage run` wrapper), so with short GEMMs a collective can land in a
    # gap between GEMMs instead of running alongside one, making the count jitter
    # run to run. At dim=16384 the GEMMs dominate that launch jitter and the
    # overlap becomes deterministic. (dim=8192 was flaky under coverage.)
    dim = 16384
    num_children = 4
    dtype = torch.bfloat16
    placements = placements_factory()

    # new_group requires a default process group. Initialize it here so this test works
    # in isolation. Do not eagerly initialize it with device_id in the shared fixture:
    # that can hang teardown after communicator splits; see
    # https://github.com/pytorch/pytorch/issues/190396.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if use_symmetric_memory:
        # Dedicated communicator with NCCL's zero-CTA policy. cta_policy is a
        # per-communicator property, so scoping it to this group leaves the rest of the
        # bucket on default-CTA symmetric-memory kernels (test_symmetric_memory.py asserts
        # ncclSymk all-gather kernel counts, which zero-CTA would turn into copy-engine
        # memcpys). This 1-D group models the DP (FSDP) sub-mesh that mfsdp is handed in
        # production: with EP/TP the full device mesh is multi-dimensional, but mfsdp
        # requires an all-FSDP mesh (see experimental/module.py) and never sees the TP/EP
        # axes, so only the DP communicator needs the zero-CTA policy.
        zero_cta_options = dist.ProcessGroupNCCL.Options()
        zero_cta_options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        dp_group = dist.new_group(backend="nccl", pg_options=zero_cta_options)
        # NCCL window registration can fail when symmetric-memory rendezvous is the first
        # operation on a communicator, so initialize this communicator explicitly.
        dist.barrier(group=dp_group, device_ids=[device.index])
    else:
        dp_group = dist.new_group(backend="nccl")

    mesh = DeviceMesh.from_group(dp_group, device.type)
    model = MultiChildModel(dim=dim, num_children=num_children).to(device=device, dtype=dtype)
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(
        device=device,
        use_symmetric_memory=use_symmetric_memory,
        unify_communication_stream=unify_communication_stream,
    ) as context:
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    fully_shard_optimizer(optimizer)
    x = torch.randn(8192, dim, device=device, dtype=dtype, requires_grad=True)

    def train_one_step() -> None:
        """Run one optimizer step consuming two microbatches."""
        optimizer.zero_grad(set_to_none=True)
        for microbatch_index in range(2):
            with microbatch(context, is_last=microbatch_index == 1):
                model(x).sum().backward()
        optimizer.step()

    train_one_step()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_step()
        # Synchronize inside the profiler context so in-flight device kernels
        # complete and get recorded before the profiler stops on __exit__.
        # Synchronizing after the context would finalize the trace first and
        # drop the CUDA events.
        torch.cuda.synchronize(device)

    gemm_groups = collect_linked_event_groups(prof, _GEMM_OP_NAME_SUBSTRING)
    # Each child Linear runs one forward and two backward matmuls per microbatch.
    # aten::mm may also launch auxiliary kernels, so check only the matmul lower bound.
    expected_gemm_count = 6 * num_children
    assert len(gemm_groups) >= expected_gemm_count, (
        f"Expected at least {expected_gemm_count} groups linked to GEMMs, got "
        f"{len(gemm_groups)}: {gemm_groups}"
    )

    allgather_groups = collect_linked_event_groups(prof, _ALL_GATHER_OP_NAME_SUBSTRING)
    reduce_scatter_groups = collect_linked_event_groups(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    # ZeRO-1/2 all-gather once per optimizer step. ZeRO-1 reduces only the second
    # microbatch, while ZeRO-2 reduces both. ZeRO-3 gathers for every forward and
    # backward and reduces every microbatch.
    expected_collective_counts = {
        _zero1_placements: (num_children, num_children, num_children - 1, num_children - 1),
        _zero2_placements: (
            num_children,
            2 * num_children,
            num_children - 1,
            2 * (num_children - 1),
        ),
        _flat_placements: (
            4 * num_children,
            2 * num_children,
            4 * (num_children - 1),
            2 * (num_children - 1),
        ),
    }
    (
        expected_allgather_count,
        expected_reduce_scatter_count,
        expected_allgather_overlap_count,
        expected_reduce_scatter_overlap_count,
    ) = expected_collective_counts[placements_factory]

    assert len(allgather_groups) == expected_allgather_count, (
        f"Expected {expected_allgather_count} all-gather groups, got "
        f"{len(allgather_groups)}: {allgather_groups}"
    )
    assert len(reduce_scatter_groups) == expected_reduce_scatter_count, (
        f"Expected {expected_reduce_scatter_count} reduce-scatter groups, got "
        f"{len(reduce_scatter_groups)}: {reduce_scatter_groups}"
    )

    allgather_events = list(chain.from_iterable(allgather_groups))
    reduce_scatter_events = list(chain.from_iterable(reduce_scatter_groups))
    gemm_events = list(chain.from_iterable(gemm_groups))
    allgather_streams = {event.device_resource_id for event in allgather_events}
    reduce_scatter_streams = {event.device_resource_id for event in reduce_scatter_events}
    gemm_streams = {event.device_resource_id for event in gemm_events}
    if allgather_events:
        assert len(allgather_streams) == 1
        if unify_communication_stream:
            assert allgather_streams == reduce_scatter_streams
        else:
            assert allgather_streams.isdisjoint(reduce_scatter_streams)
    assert len(reduce_scatter_streams) == 1
    assert allgather_streams.isdisjoint(gemm_streams)
    assert reduce_scatter_streams.isdisjoint(gemm_streams)

    allgather_overlap_count = sum(
        any(event_groups_overlap(group, gemm_group) for gemm_group in gemm_groups)
        for group in allgather_groups
    )
    reduce_scatter_overlap_count = sum(
        any(event_groups_overlap(group, gemm_group) for gemm_group in gemm_groups)
        for group in reduce_scatter_groups
    )
    if not use_symmetric_memory:
        assert allgather_overlap_count == expected_allgather_overlap_count, (
            f"Expected exactly {expected_allgather_overlap_count} all-gathers to "
            f"overlap compute, got {allgather_overlap_count}/{len(allgather_groups)}."
        )
    assert reduce_scatter_overlap_count == expected_reduce_scatter_overlap_count, (
        f"Expected exactly {expected_reduce_scatter_overlap_count} reduce-scatters to overlap "
        f"compute, got {reduce_scatter_overlap_count}/{len(reduce_scatter_groups)}."
    )

    # Release the dedicated communicator so it does not leak into the shared session.
    dist.destroy_process_group(dp_group)


def test_prefetch_size_zero_disables_allgather_overlap(distributed_setup):
    """Zero per-module prefetch budgets should launch all-gathers before compute."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    dim = 16384
    num_children = 4
    dtype = torch.bfloat16
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=dim, num_children=num_children).to(device=device, dtype=dtype)
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    placements = _flat_placements()
    with fully_shard_context(device=device) as context:
        for layer in model.layers:
            fully_shard(
                layer,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=policy,
                schedule_policy=SchedulePolicy(forward_prefetch_size=0, backward_prefetch_size=0),
            )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    fully_shard_optimizer(optimizer)
    x = torch.randn(8192, dim, device=device, dtype=dtype, requires_grad=True)

    def train_one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for microbatch_index in range(2):
            with microbatch(context, is_last=microbatch_index == 1):
                model(x).sum().backward()
        optimizer.step()

    train_one_step()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_step()
        torch.cuda.synchronize(device)

    gemm_groups = collect_linked_event_groups(prof, _GEMM_OP_NAME_SUBSTRING)
    allgather_groups = collect_linked_event_groups(prof, _ALL_GATHER_OP_NAME_SUBSTRING)
    assert len(allgather_groups) == 4 * num_children
    assert all(
        not any(event_groups_overlap(allgather, gemm) for gemm in gemm_groups)
        for allgather in allgather_groups
    ), "All-gathers overlapped GEMM compute despite zero prefetch budgets."
