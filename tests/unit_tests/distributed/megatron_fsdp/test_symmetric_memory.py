# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental FSDP symmetric-memory staging."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

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
    return sum(
        1
        for kernel in kernels
        if _is_symmetric_kernel(kernel) and subname in kernel
    )


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
        model = TinyModel().to(device)
        fully_shard(
            model.fc1, mesh=mesh, placements=_flat_placements(), use_symm_mem=use_symm_mem
        )
        fully_shard(
            model.fc2, mesh=mesh, placements=_flat_placements(), use_symm_mem=use_symm_mem
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        micro_batch_size = 2
        x = torch.randn(num_microbatches, micro_batch_size, 8, device=device)
        target = torch.randn(num_microbatches, micro_batch_size, 4, device=device)
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
    expected_reduce_scatter_kernel_count = num_training_steps * num_microbatches * num_sharded_modules
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
