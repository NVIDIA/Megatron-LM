# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Utilities for offloading grad_data buffers from GPU memory."""

from typing import Dict, Any

import torch

from megatron.core.distributed import DistributedDataParallel


def get_grad_buffer_memory_usage(ddp_model: DistributedDataParallel) -> Dict[str, Any]:
    """
    Get memory usage information for grad_data buffers.

    Args:
        ddp_model: DistributedDataParallel model instance

    Returns:
        Dictionary containing:
        - total_bytes: Total bytes used by all grad buffers
        - buffers: Dict mapping buffer names to their info (numel, dtype, device)
    """
    all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers

    total_bytes = 0
    buffer_info = {}

    for idx, buffer in enumerate(all_buffers):
        storage_size = buffer.grad_data.storage().size()
        if storage_size > 0:
            numel = buffer.grad_data.numel()
            dtype = buffer.grad_dtype
            device = str(buffer.grad_data.device)
            bytes_used = numel * buffer.grad_data.element_size()
        else:
            # Buffer is offloaded
            numel = 0
            dtype = buffer.grad_dtype
            device = "offloaded"
            bytes_used = 0

        buffer_info[f"buffer_{idx}"] = {
            "numel": numel,
            "dtype": str(dtype),
            "device": device,
        }
        total_bytes += bytes_used

    return {"total_bytes": total_bytes, "buffers": buffer_info}


def offload_grad_data(
    ddp_model: DistributedDataParallel,
    synchronize: bool = True,
    empty_cache: bool = True,
) -> None:
    """
    Free all grad_data tensors from a DistributedDataParallel model to release GPU memory.

    Uses storage().resize_(0) to release memory while keeping tensor views intact.
    All bucket.grad_data and param.main_grad views remain valid tensor objects
    (though accessing them during offload is undefined behavior).

    The storage size is tracked internally by each buffer, so no state needs to be
    passed to onload_grad_data.

    Args:
        ddp_model: DistributedDataParallel model instance
        synchronize: Whether to call torch.cuda.synchronize() before freeing
        empty_cache: Whether to call torch.cuda.empty_cache() after freeing
    """
    if synchronize:
        torch.cuda.synchronize()

    all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers

    for buffer in all_buffers:
        buffer.offload_to_cpu(move_params=False, move_grads=True)

    if empty_cache:
        torch.cuda.empty_cache()


def onload_grad_data(
    ddp_model: DistributedDataParallel,
    synchronize: bool = True,
) -> None:
    """
    Reallocate grad_data tensors on GPU.

    All existing views (bucket.grad_data, param.main_grad) automatically
    become valid again since they share the same storage. The grad_data
    is zeroed after reallocation.

    Args:
        ddp_model: DistributedDataParallel model instance
        synchronize: Whether to call torch.cuda.synchronize() after allocation
    """
    all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers

    for buffer in all_buffers:
        buffer.reload_from_cpu(move_params=False, move_grads=True)

    if synchronize:
        torch.cuda.synchronize()
