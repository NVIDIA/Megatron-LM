# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental FSDP symmetric-memory staging."""

from itertools import chain

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    RuntimePolicy,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import collect_linked_event_groups

# Each sharded Linear's collective must be large enough that NCCL selects its
# symmetric-memory (ncclSymk*) kernels over ring. Sub-KB collectives fall back to
# ring on some platforms (e.g. CI with NCCL_NVLS_ENABLE=0), which would make the
# symmetric-kernel assertions below fail; 1024-wide layers (a few-MiB bf16 weight)
# reliably engage the symmetric kernels.
_HIDDEN = 1024
_ALL_GATHER_OP_NAME_SUBSTRING = "allgather"
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"


class TinyModel(nn.Module):
    """Two separately shardable Linear modules, sized so NCCL selects symmetric-memory kernels."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _HIDDEN)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""
        return self.fc2(self.relu(self.fc1(x)))


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def test_dbuffer_allgather_registers_separate_symmetric_buffers(distributed_setup):
    """Separate send and receive allocations must both support symmetric all-gather."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if device.type != "cuda" or world_size < 2:
        pytest.skip("Requires at least 2 CUDA ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dist.barrier(group=mesh.get_group(), device_ids=[device.index])
    # Separate pools prevent output registration from also covering the input's segment.
    allocator = symm_mem.get_mempool_allocator(device)
    pools = [torch.cuda.MemPool(allocator, no_split=True) for _ in range(2)]
    with torch.cuda.use_mem_pool(pools[0]):
        source = DBuffer.empty(
            mesh, [RowAtomic()], [(world_size * _HIDDEN, _HIDDEN)], torch.bfloat16, device
        )
    with torch.cuda.use_mem_pool(pools[1]):
        out = DBuffer(mesh, [Replicate()], source.layout, source.dtype, device)
    assert source.is_symmetric_memory and out.is_symmetric_memory
    source.local_buffer.fill_(mesh.get_local_rank() + 1)

    with torch.profiler.profile() as prof:
        assert source.allgather(0, out=out) is out
        torch.cuda.synchronize()

    expected = torch.arange(1, world_size + 1, dtype=source.dtype, device=device)
    torch.testing.assert_close(
        out.local_buffer.view(world_size, -1),
        expected[:, None].expand(world_size, source.local_buffer.numel()),
        rtol=0,
        atol=0,
    )
    # Retire registered allocations while their communicator is still alive.
    del source, out, pools

    events = list(
        chain.from_iterable(collect_linked_event_groups(prof, _ALL_GATHER_OP_NAME_SUBSTRING))
    )
    assert events, "No all-gather CUDA events were recorded."
    assert all(
        "ncclSymk" in event.name for event in events
    ), f"Expected symmetric all-gather for separate input/output buffers, observed: {events}"


@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_fully_shard_symmetric_memory_matches_default_and_profiles_nccl(
    distributed_setup, num_microbatches
):
    """NCCL symmetric-memory staging should preserve training parity and hit symmetric kernels."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    num_training_steps = 5

    def train(use_symmetric_memory: bool) -> list[torch.Tensor]:
        torch.manual_seed(1234)
        model = TinyModel().to(device=device, dtype=torch.bfloat16)
        mixed_precision_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
        with fully_shard_context(
            FsdpContext(
                device=device,
                runtime_policy=RuntimePolicy(use_symmetric_memory=use_symmetric_memory),
            )
        ):
            fully_shard(
                model.fc1,
                mesh=mesh,
                placements=_default_placements(),
                mixed_precision_policy=mixed_precision_policy,
            )
            fully_shard(
                model.fc2,
                mesh=mesh,
                placements=_default_placements(),
                mixed_precision_policy=mixed_precision_policy,
            )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)

        micro_batch_size = 2
        x = torch.randn(
            num_microbatches, micro_batch_size, _HIDDEN, device=device, dtype=torch.bfloat16
        )
        target = torch.randn(
            num_microbatches, micro_batch_size, _HIDDEN, device=device, dtype=torch.bfloat16
        )
        microbatches = tuple(zip(x.unbind(), target.unbind()))

        losses = []
        for _ in range(num_training_steps):
            optimizer.zero_grad()
            for microbatch_x, microbatch_target in microbatches:
                loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                losses.append(loss.detach())
                (loss / num_microbatches).backward()
            optimizer.step()

        return losses

    with torch.profiler.profile() as prof_without_symm_mem:
        losses_without_symm_mem = train(use_symmetric_memory=False)
        torch.cuda.synchronize()

    with torch.profiler.profile() as prof_with_symm_mem:
        losses_with_symm_mem = train(use_symmetric_memory=True)
        torch.cuda.synchronize()

    torch.testing.assert_close(
        torch.stack(losses_with_symm_mem),
        torch.stack(losses_without_symm_mem),
        msg="Symmetric-memory FSDP losses did not match default FSDP losses.",
    )

    allgather_groups_without_symm_mem = collect_linked_event_groups(
        prof_without_symm_mem, _ALL_GATHER_OP_NAME_SUBSTRING
    )
    reduce_scatter_groups_without_symm_mem = collect_linked_event_groups(
        prof_without_symm_mem, _REDUCE_SCATTER_OP_NAME_SUBSTRING
    )
    assert all(
        "ncclSymk" not in event.name
        for event in chain.from_iterable(allgather_groups_without_symm_mem)
    )
    assert all(
        "ncclSymk" not in event.name
        for event in chain.from_iterable(reduce_scatter_groups_without_symm_mem)
    )

    allgather_groups_with_symm_mem = collect_linked_event_groups(
        prof_with_symm_mem, _ALL_GATHER_OP_NAME_SUBSTRING
    )
    reduce_scatter_groups_with_symm_mem = collect_linked_event_groups(
        prof_with_symm_mem, _REDUCE_SCATTER_OP_NAME_SUBSTRING
    )
    # 2 sharded modules (fc1, fc2), one reduce-scatter each per microbatch step.
    expected_reduce_scatter_group_count = num_training_steps * num_microbatches * 2
    assert len(reduce_scatter_groups_with_symm_mem) == expected_reduce_scatter_group_count, (
        "Unexpected NCCL symmetric-memory reduce-scatter group count. "
        f"Observed reduce-scatter events: {reduce_scatter_groups_with_symm_mem[:20]}"
    )
    assert all(
        "ncclSymk" in event.name
        for event in chain.from_iterable(reduce_scatter_groups_with_symm_mem)
    ), (
        "Expected all symmetric-memory reduce-scatter events to be ncclSymk kernels. "
        f"Observed reduce-scatter events: {reduce_scatter_groups_with_symm_mem[:20]}"
    )

    expected_allgather_group_count = 2 * expected_reduce_scatter_group_count
    assert len(allgather_groups_with_symm_mem) == expected_allgather_group_count, (
        "Unexpected NCCL symmetric-memory all-gather group count. "
        f"Observed all-gather events: {allgather_groups_with_symm_mem[:20]}"
    )
    assert all(
        "ncclSymk" in event.name for event in chain.from_iterable(allgather_groups_with_symm_mem)
    ), (
        "Expected all symmetric-memory all-gather events to be ncclSymk kernels. "
        f"Observed all-gather events: {allgather_groups_with_symm_mem[:20]}"
    )


def test_fully_shard_zero_cta_moves_all_gather_to_copy_engine(distributed_setup):
    """NCCL's zero-CTA policy runs the all-gather on the copy engine.

    Zero-CTA offloads only pure data movement, so the all-gather emits no ``ncclSymk``
    kernel (it becomes a copy-engine memcpy). The reduce-scatter's reduction cannot run on
    the copy engine, so it stays a symmetric-memory kernel -- an SM-launched NVLS multicast
    reduce (ncclSymkDevKernel_ReduceScatter_LDMC/LL).
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    # new_group requires a default process group. Initialize it here so this test works
    # in isolation. Do not eagerly initialize it with device_id in the shared fixture:
    # that can hang teardown after communicator splits; see
    # https://github.com/pytorch/pytorch/issues/190396.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Dedicated communicator with NCCL's zero-CTA policy, scoped to this test so the rest
    # of the bucket keeps default-CTA symmetric-memory kernels.
    zero_cta_options = dist.ProcessGroupNCCL.Options()
    zero_cta_options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dp_group = dist.new_group(backend="nccl", pg_options=zero_cta_options)
    # NCCL window registration can fail when symmetric-memory rendezvous is the first
    # operation on a communicator, so initialize this communicator explicitly.
    dist.barrier(group=dp_group, device_ids=[device.index])
    mesh = DeviceMesh.from_group(dp_group, device.type)

    num_training_steps = 5
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    mixed_precision_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
    with fully_shard_context(
        FsdpContext(device=device, runtime_policy=RuntimePolicy(use_symmetric_memory=True))
    ):
        fully_shard(
            model.fc1,
            mesh=mesh,
            placements=_default_placements(),
            mixed_precision_policy=mixed_precision_policy,
        )
        fully_shard(
            model.fc2,
            mesh=mesh,
            placements=_default_placements(),
            mixed_precision_policy=mixed_precision_policy,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)
    x = torch.randn(2, _HIDDEN, device=device, dtype=torch.bfloat16)
    target = torch.randn(2, _HIDDEN, device=device, dtype=torch.bfloat16)

    with torch.profiler.profile() as prof:
        for _ in range(num_training_steps):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(x), target).backward()
            optimizer.step()
        torch.cuda.synchronize()

    allgather_groups = collect_linked_event_groups(prof, _ALL_GATHER_OP_NAME_SUBSTRING)
    reduce_scatter_groups = collect_linked_event_groups(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    # Two sharded modules each all-gather in forward and backward on every training step.
    assert len(allgather_groups) == num_training_steps * 2 * 2, (
        "Expected zero-CTA all-gather groups. "
        f"Observed all-gather events: {allgather_groups[:20]}"
    )
    # The reduce-scatter's reduction cannot run on the copy engine, so it stays a
    # symmetric-memory kernel (an SM-launched NVLS multicast reduce): one per sharded
    # module (fc1, fc2) per training step.
    expected_reduce_scatter_group_count = num_training_steps * 2
    assert len(reduce_scatter_groups) == expected_reduce_scatter_group_count, (
        f"Expected {expected_reduce_scatter_group_count} symmetric-memory reduce-scatter "
        f"groups under zero-CTA. Observed reduce-scatter events: {reduce_scatter_groups[:20]}"
    )
    assert all("ncclSymk" in event.name for event in chain.from_iterable(reduce_scatter_groups)), (
        "Expected all zero-CTA reduce-scatter events to be ncclSymk kernels. "
        f"Observed reduce-scatter events: {reduce_scatter_groups[:20]}"
    )
