# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Communication-overlap tests for the minimal Megatron-FSDP path."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import (
    collect_linked_kernels,
    events_overlap,
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
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


# CPU ops that a device event chains up to via cpu_parent, used to attribute the device
# work to its enclosing collective or matmul operation.
_ALL_GATHER_OP_NAME_SUBSTRING = "allgather"
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"
_GEMM_OP_NAME_SUBSTRING = "aten::mm"


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
def test_overlaps_communication_and_compute(distributed_setup, use_symmetric_memory):
    """Forward and backward communication should overlap GEMM compute."""
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
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(device=device, use_symmetric_memory=use_symmetric_memory):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(4096, dim, device=device, dtype=dtype, requires_grad=True)

    def train_one_iteration() -> None:
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()

    train_one_iteration()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_iteration()
        # Synchronize inside the profiler context so in-flight device kernels
        # complete and get recorded before the profiler stops on __exit__.
        # Synchronizing after the context would finalize the trace first and
        # drop the CUDA events.
        torch.cuda.synchronize(device)

    gemm_kernels = collect_linked_kernels(prof, _GEMM_OP_NAME_SUBSTRING)
    # Each child Linear runs one forward and two backward matmuls. aten::mm may also
    # launch auxiliary kernels, so check only the matmul lower bound.
    assert len(gemm_kernels) >= 3 * num_children, (
        f"Expected at least {3 * num_children} kernels linked to GEMMs, got "
        f"{len(gemm_kernels)}: "
        f"{[kernel.name for kernel in gemm_kernels]}"
    )

    allgather_kernels = collect_linked_kernels(prof, _ALL_GATHER_OP_NAME_SUBSTRING)
    reduce_scatter_kernels = collect_linked_kernels(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    # Each child layer does a forward and a backward all-gather and one
    # reduce-scatter. Zero-CTA moves the all-gather to copy-engine memcpys, so it
    # should not emit all-gather kernels.
    expected_allgather_kernel_count = 0 if use_symmetric_memory else 2 * num_children
    assert len(allgather_kernels) == expected_allgather_kernel_count, (
        f"Expected {expected_allgather_kernel_count} all-gather kernels, got "
        f"{len(allgather_kernels)}: {[kernel.name for kernel in allgather_kernels]}"
    )
    assert len(reduce_scatter_kernels) == num_children, (
        f"Expected {num_children} reduce-scatter kernels, got "
        f"{len(reduce_scatter_kernels)}: {[kernel.name for kernel in reduce_scatter_kernels]}"
    )

    allgather_streams = {kernel.device_resource_id for kernel in allgather_kernels}
    reduce_scatter_streams = {kernel.device_resource_id for kernel in reduce_scatter_kernels}
    gemm_streams = {kernel.device_resource_id for kernel in gemm_kernels}
    if allgather_kernels:
        assert len(allgather_streams) == 1
    assert len(reduce_scatter_streams) == 1
    assert allgather_streams.isdisjoint(reduce_scatter_streams)
    assert allgather_streams.isdisjoint(gemm_streams)
    assert reduce_scatter_streams.isdisjoint(gemm_streams)

    allgather_overlap_count = sum(
        any(events_overlap(kernel, gemm) for gemm in gemm_kernels) for kernel in allgather_kernels
    )
    reduce_scatter_overlap_count = sum(
        any(events_overlap(kernel, gemm) for gemm in gemm_kernels)
        for kernel in reduce_scatter_kernels
    )
    expected_allgather_overlap = 2 * (num_children - 1)
    expected_reduce_scatter_overlap = num_children - 1
    if not use_symmetric_memory:
        assert allgather_overlap_count >= expected_allgather_overlap, (
            f"Expected at least {expected_allgather_overlap} all-gathers to "
            f"overlap compute, got {allgather_overlap_count}/{len(allgather_kernels)}."
        )
    assert reduce_scatter_overlap_count >= expected_reduce_scatter_overlap, (
        f"Expected at least {expected_reduce_scatter_overlap} reduce-scatters to overlap "
        f"compute, got {reduce_scatter_overlap_count}/{len(reduce_scatter_kernels)}."
    )

    # Release the dedicated communicator so it does not leak into the shared session.
    dist.destroy_process_group(dp_group)
