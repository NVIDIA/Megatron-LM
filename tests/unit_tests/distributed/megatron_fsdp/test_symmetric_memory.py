# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental FSDP symmetric-memory staging."""

import math

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)


class TinyModel(nn.Module):
    """Small model with two separately shardable units."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(self.relu(self.fc1(x)))


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _kernels(prof: torch.profiler.profile) -> list[str]:
    return [event.name for event in prof.events()]


def _is_symmetric_kernel(kernel: str) -> bool:
    return "ncclSymk" in kernel


def _count_symmetric_kernels(kernels: list[str], subname: str) -> int:
    return sum(1 for kernel in kernels if _is_symmetric_kernel(kernel) and subname in kernel)


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
    num_sharded_modules = 2
    num_training_steps = 5

    def train(use_symm_mem: bool) -> list[torch.Tensor]:
        torch.manual_seed(1234)
        model = TinyModel().to(device=device, dtype=torch.bfloat16)
        mixed_precision_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
        fully_shard(
            model.fc1,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=mixed_precision_policy,
            use_symm_mem=use_symm_mem,
        )
        fully_shard(
            model.fc2,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=mixed_precision_policy,
            use_symm_mem=use_symm_mem,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)

        micro_batch_size = 2
        x = torch.randn(num_microbatches, micro_batch_size, 8, device=device, dtype=torch.bfloat16)
        target = torch.randn(
            num_microbatches, micro_batch_size, 4, device=device, dtype=torch.bfloat16
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

    with profile(activities=[ProfilerActivity.CUDA]) as prof_without_symm_mem:
        losses_without_symm_mem = train(use_symm_mem=False)
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof_with_symm_mem:
        losses_with_symm_mem = train(use_symm_mem=True)
        torch.cuda.synchronize()

    torch.testing.assert_close(
        torch.stack(losses_with_symm_mem),
        torch.stack(losses_without_symm_mem),
        msg="Symmetric-memory FSDP losses did not match default FSDP losses.",
    )

    kernels_without_symm_mem = _kernels(prof_without_symm_mem)
    assert _count_symmetric_kernels(kernels_without_symm_mem, "AllGather") == 0
    assert _count_symmetric_kernels(kernels_without_symm_mem, "ReduceScatter") == 0

    kernels_with_symm_mem = _kernels(prof_with_symm_mem)
    expected_reduce_scatter_kernel_count = (
        num_training_steps * num_microbatches * num_sharded_modules
    )
    nccl_kernels_with_symm_mem = [
        kernel for kernel in kernels_with_symm_mem if "nccl" in kernel.lower()
    ]
    assert (
        _count_symmetric_kernels(kernels_with_symm_mem, "ReduceScatter")
        == expected_reduce_scatter_kernel_count
    ), (
        "Unexpected NCCL symmetric-memory reduce-scatter kernel count. "
        f"Observed NCCL kernels: {nccl_kernels_with_symm_mem[:20]}"
    )

    expected_all_gather_kernel_count = 2 * expected_reduce_scatter_kernel_count
    assert (
        _count_symmetric_kernels(kernels_with_symm_mem, "AllGather")
        == expected_all_gather_kernel_count
    ), (
        "Unexpected NCCL symmetric-memory all-gather kernel count. "
        f"Observed NCCL kernels: {nccl_kernels_with_symm_mem[:20]}"
    )


# ---------------------------------------------------------------------------
# Communication-time benchmark (runs by default under torchrun).
#
# For each hidden size (a pytest parameter), shards a single Linear(hidden, hidden)
# and reports the communication time of its all-gather (weights) and
# reduce-scatter (grads) for default (ring) vs symmetric memory, using bf16
# params + fp32 main weights. The collective sizes depend only on the weight
# (hidden^2), so hidden is the only knob; batch / warmup / step counts don't
# change what is measured and are fixed internally.
#
# "Communication time" is the MIN device time across ranks for each recorded collective:
# NCCL LL/symm kernels spin-wait for peers, so a rank's device time = transfer + wait. The
# least-waited rank approximates the intrinsic transfer time -- it matches a synchronized
# pure-NCCL microbench. The across-ranks min does most of the work, so a single profiled
# step suffices.
#
# Run (single node, 8 GPUs):
#   torchrun --nproc_per_node=8 -m pytest -q -s \
#     tests/unit_tests/distributed/megatron_fsdp/test_symmetric_memory.py -k benchmark
# ---------------------------------------------------------------------------


# CPU-side c10d collective ops -> communication type. With record_shapes=True these
# carry the buffer shapes (so we get the message size), and their device_time_total is
# the launched NCCL kernel's time -- for both the ring and symmetric backends.
_EVENT_NAME_TO_COLLECTIVE = {
    "c10d::_allgather_base_": "all_gather",
    "c10d::_reduce_scatter_base_": "reduce_scatter",
}


def _collective_records(prof, *, element_size: int):
    """One record per collective instance observed in the profiled step, in program order.

    Size comes from the c10d op's recorded shapes (full buffer = max numel); time from its
    device_time_total (= the launched NCCL kernel's time). The c10d ops are CPU-side, so they
    appear in dispatcher call order -- identical across ranks for this symmetric step -- which
    is the order the caller's positional cross-rank min relies on.
    """
    records = []
    for ev in prof.events():
        # prof.events() includes every op in the step; keep only the collective ops.
        ctype = _EVENT_NAME_TO_COLLECTIVE.get(ev.name)
        if ctype is None:
            continue
        # A collective with no attributed device time means its NCCL kernel wasn't
        # captured -- fail loudly rather than drop the record and skew the min.
        assert ev.device_time_total > 0, (
            f"{ev.name} recorded no device time (device_time_total="
            f"{ev.device_time_total}); its NCCL kernel was not captured."
        )
        numel = max((math.prod(s) for s in ev.input_shapes if s), default=0)
        records.append(
            {"collective": ctype, "bytes": numel * element_size, "us": ev.device_time_total}
        )
    return records


@pytest.mark.parametrize("hidden", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("use_symm_mem", [False, True])
def test_symmetric_memory_comm_time_benchmark(benchmark, distributed_setup, hidden, use_symm_mem):
    """Benchmark one sharded Linear(hidden, hidden) step (bf16 params + fp32 main weights).

    pytest-benchmark's ``benchmark`` fixture times the wall-clock per step; one record per
    collective instance -- each with the cross-rank min kernel time (the wait-free
    communication time) -- is attached as ``benchmark.extra_info["communications"]``.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This benchmark requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    # Warm the communicator with a barrier so the symmetric rendezvous below is never the
    # first NCCL op on this group -- otherwise NCCL window registration fails
    # (pytorch/pytorch#188567).
    torch.distributed.barrier(device_ids=[device.index])

    torch.manual_seed(1234)
    model = nn.Linear(hidden, hidden, bias=False).to(device=device, dtype=torch.bfloat16)
    # The default MixedPrecisionPolicy() already keeps fp32 main weights + param-dtype
    # (bf16) grads, so no policy is passed. foreach=False so stock SGD tolerates the
    # fp32-param / bf16-grad pairing (the foreach path rejects mixed dtypes).
    fully_shard(model, mesh=mesh, placements=_flat_placements(), use_symm_mem=use_symm_mem)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    # batch only drives a fwd/bwd; it does not affect the collective sizes (those depend on
    # the weight, hidden^2).
    x = torch.randn(512, hidden, device=device, dtype=torch.bfloat16)

    def step():
        optimizer.zero_grad()
        model(x).sum().backward()  # any backward produces the weight grad -> reduce-scatter
        optimizer.step()
        torch.cuda.synchronize()

    # Wall-clock per step FIRST. pedantic's warmup_rounds (untimed) absorb the one-time costs --
    # symm window registration on the first symm collective, allocator growth, NCCL channel
    # setup -- so both its timed rounds AND the profiling below are steady-state. Fixed
    # rounds/iterations keep every rank issuing the same number of collectives in lockstep;
    # auto-calibrated benchmark(step) would pick per-rank counts and deadlock.
    benchmark.pedantic(step, rounds=5, iterations=1, warmup_rounds=2)

    # Profile a SINGLE step (model fully warm from pedantic above). The cross-rank min below
    # reduces over this step's ~2 all-gather + 1 reduce-scatter instances x ranks -- enough
    # for the least-waited (wait-free) time without looping.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        step()

    communications = _collective_records(prof, element_size=x.element_size())
    # The cross-rank min below is positional, so records[i] must be the same collective on
    # every rank. Program order gives that for this symmetric step; the one way it breaks is a
    # rank dropping an instance (e.g. a zero-duration event filtered out), which shifts every
    # later position -- so assert the counts match before reducing.
    n = len(communications)
    n_max = torch.tensor(n, device=device)
    n_min = torch.tensor(n, device=device)
    torch.distributed.all_reduce(n_max, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(n_min, op=torch.distributed.ReduceOp.MIN)
    assert n_min.item() == n == n_max.item(), (
        f"ranks recorded different collective counts (this rank {n}, "
        f"range {n_min.item()}..{n_max.item()}); records are misaligned."
    )
    # Reduce each record's time across ranks: within a collective the least-waited rank's
    # kernel time ~= the intrinsic transfer (NCCL LL/symm kernels spin-wait for peers, so a
    # rank's device time = transfer + wait-for-slowest-peer), so min over ranks is the
    # tightest wait-free estimate.
    mins = torch.tensor([c["us"] for c in communications], device=device)
    torch.distributed.all_reduce(mins, op=torch.distributed.ReduceOp.MIN)
    for c, v in zip(communications, mins.tolist()):
        c["us"] = v
    benchmark.extra_info["communications"] = communications
