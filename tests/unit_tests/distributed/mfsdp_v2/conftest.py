# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _world


@dataclasses.dataclass(frozen=True)
class DistributedSetup:
    """Per-rank distributed test setup."""

    rank: int
    world_size: int
    device: torch.device


@pytest.fixture(scope="function")
def distributed_setup() -> Iterator[DistributedSetup]:
    """Set up this rank's local device and clean up per-test process groups."""
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

        # Centralize cleanup so tests do not need to track and destroy every subgroup.
        # Fixture teardown also runs if a test fails or skips after setup, preventing
        # communicator resources from accumulating across tests.
        #
        # Destruction order is the main risk: ranks must destroy overlapping groups in a
        # consistent order to avoid NCCL hangs. pg_map preserves local insertion order,
        # so this relies on consistent group creation order across ranks.
        #
        # Destroying WORLD adds a full distributed restart: fast ranks could reinitialize
        # while peers are still shutting down, causing NCCL connection failures. PyTorch
        # requires synchronization outside torch.distributed between destruction and
        # reinitialization. Later test teardown also uses WORLD, so leave its destruction
        # to session cleanup.
        # https://docs.pytorch.org/docs/stable/distributed.html#reinitialization
        for group in list(_world.pg_map):
            if group is not dist.group.WORLD:
                dist.destroy_process_group(group)
