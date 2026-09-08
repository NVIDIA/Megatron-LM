# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclasses.dataclass(frozen=True)
class DistributedSetup:
    """Per-rank distributed test setup."""

    rank: int
    world_size: int
    device: torch.device


@pytest.fixture(scope="function")
def distributed_setup() -> Iterator[DistributedSetup]:
    """Read torchrun rank state and set up this rank's local device."""
    # Some MFSDP v2 tests are sensitive to NCCL algorithm/channel choices. Clear
    # the suite-wide NCCL defaults (set in the top-level conftest.py) before
    # init_device_mesh initializes NCCL communicators so this bucket uses NCCL
    # settings closer to production.
    os.environ.pop("NCCL_MAX_NCHANNELS", None)
    os.environ.pop("NCCL_NVLS_ENABLE", None)

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # is_symm_mem_tensor() marks the current symmetric-memory backend as in use,
        # even for ordinary tensors, so select NCCL before any DBuffer test calls it.
        symm_mem.set_backend("NCCL")
    else:
        device = torch.device("cpu")

    yield DistributedSetup(rank=rank, world_size=world_size, device=device)

    if dist.is_initialized():
        # Keep the default process group alive for later distributed tests.
        if device.type == "cuda":
            # Pass the device explicitly to suppress PyTorch's NCCL barrier warning.
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()
