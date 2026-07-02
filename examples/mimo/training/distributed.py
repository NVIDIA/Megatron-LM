# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bring up torch.distributed and the global memory buffer for hetero MIMO without initializing MPU."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.training.utils import print_rank_0

__all__ = ["initialize_distributed", "print_rank_0", "shutdown_distributed"]


def initialize_distributed() -> None:
    """Bring up torch.distributed + the global memory buffer without MPU init."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert_parallel_state_uninitialized()
    try:
        parallel_state.get_global_memory_buffer()
    except AssertionError:
        parallel_state._set_global_memory_buffer()
    dist.barrier()


def assert_parallel_state_uninitialized() -> None:
    """Assert no Megatron model-parallel globals are set so the hetero path owns setup."""
    # Defensive parallel_state reads (allowed compatibility point); check each global MIMO
    # constructs, since model_parallel_is_initialized() omits CP/embedding.
    checks = (
        ("data_parallel", parallel_state.is_initialized),
        ("tensor_model_parallel", _grp(parallel_state.get_tensor_model_parallel_group)),
        ("pipeline_model_parallel", _grp(parallel_state.get_pipeline_model_parallel_group)),
        ("context_parallel", _grp(parallel_state.get_context_parallel_group)),
        ("embedding", _grp(parallel_state.get_embedding_group)),
        ("position_embedding", _grp(parallel_state.get_position_embedding_group)),
    )
    initialized = [label for label, predicate in checks if predicate()]
    if initialized:
        raise RuntimeError(
            "Hetero MIMO bootstrap expects Megatron parallel_state process groups to be "
            f"uninitialized, but found: {', '.join(initialized)}"
        )


def _grp(getter):
    """Turn a group getter into a no-arg "is it set?" predicate."""
    return lambda: getter(check_initialized=False) is not None


def shutdown_distributed() -> None:
    """Tear down the global memory buffer and torch.distributed (thin stock teardown)."""
    parallel_state.destroy_global_memory_buffer()
    if dist.is_initialized():
        dist.destroy_process_group()
