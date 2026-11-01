# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bring up torch.distributed for hetero MIMO without initializing MPU."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.utils import destroy_global_memory_buffer
from megatron.training.utils import print_rank_0

__all__ = ["initialize_distributed", "print_rank_0", "shutdown_distributed"]


def initialize_distributed() -> None:
    """Bring up torch.distributed without MPU init.

    The global memory buffer needs no setup: megatron.core.utils creates it on first use.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert_parallel_state_uninitialized()
    dist.barrier()


def assert_parallel_state_uninitialized() -> None:
    """Assert no Megatron model-parallel globals are set so the hetero path owns setup.

    initialize_model_parallel() and register_process_groups() set the global groups by
    registering a collection, so a registered collection means that some of them are set.
    """
    registered = parallel_state.get_registered_process_groups()
    if registered is not None:
        raise RuntimeError(
            "Hetero MIMO bootstrap expects Megatron parallel_state process groups to be "
            f"uninitialized, but found: {registered}"
        )
    if parallel_state.is_initialized():
        raise RuntimeError(
            "Hetero MIMO bootstrap expects Megatron parallel_state process groups to be "
            "uninitialized, but found: data_parallel"
        )


def shutdown_distributed() -> None:
    """Tear down the global memory buffer and torch.distributed (thin stock teardown)."""
    destroy_global_memory_buffer()
    if dist.is_initialized():
        dist.destroy_process_group()
