# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared fixtures for the Megatron-FSDP test directory.

Tests in this directory launch under ``torchrun`` and share a common
distributed bring-up. The ``distributed_setup`` fixture is module-scoped
so each test file gets a fresh process group, and is exported here so
test files don't each need to redefine it.
"""

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def distributed_setup() -> dict[str, Any]:
    """Setup torch.distributed and CUDA device for torchrun + pytest.

    Initializes the default process group (NCCL on CUDA, gloo on CPU) and
    yields a dict of rank/world_size/device info. Tests that require CUDA
    should add their own `pytest.mark.skipif(not torch.cuda.is_available())`.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        device_type = "cuda"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device_type = "cpu"
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    yield {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device_type": device_type,
        "device": device,
    }

    if dist.is_initialized():
        dist.destroy_process_group()
