# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utilities for freeing grad_data buffers and reallocating them.

This module provides memory management for grad_data buffers without preserving
their contents. The offload operation frees GPU memory, and the onload operation
allocates fresh buffers. This is useful when the buffer contents are not needed
(e.g., gradients will be recomputed anyway).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .param_and_grad_buffer import _ParamAndGradBuffer, _ParamAndGradBucket, _ParamAndGradBucketGroup


@dataclass
class BucketMetadata:
    """Metadata needed to restore a bucket's grad_data view."""
    offset: int
    numel: int


@dataclass
class ParamGradMetadata:
    """Metadata needed to restore a parameter's main_grad view."""
    start_index: int
    end_index: int
    shape: torch.Size


@dataclass
class GradBufferOffloadState:
    """
    State container holding metadata needed to reallocate grad_data buffers.
    
    This is returned by offload_grad_data and should be passed to onload_grad_data.
    Note: The actual tensor data is NOT preserved - only metadata for reallocation.
    """
    # Total number of elements in the buffer (for reallocation)
    numel: int
    # Original dtype and device info
    grad_dtype: torch.dtype
    original_device: torch.device
    # Bucket metadata for restoring bucket.grad_data views
    bucket_metadata: List[BucketMetadata]
    # Parameter metadata for restoring param.main_grad views
    param_grad_metadata: Dict[torch.nn.Parameter, ParamGradMetadata]
    # Buffer reference (for validation during onload)
    buffer_id: int


def offload_grad_data(
    ddp_model,
    optimizer=None,
    synchronize: bool = True,
    empty_cache: bool = True,
) -> List[GradBufferOffloadState]:
    """
    Free all grad_data tensors from a DistributedDataParallel model to release GPU memory.
    
    This function handles all references to the underlying grad_data tensors:
    1. param.main_grad views on each parameter
    2. bucket.grad_data views in each bucket
    3. cached_grad_buffer_shard_list in bucket groups
    4. The main buffer.grad_data tensor
    
    Note: The tensor data is NOT preserved. Only metadata needed to reallocate
    the buffers is stored. Use this when the buffer contents don't matter
    (e.g., gradients will be recomputed anyway).
    
    Args:
        ddp_model: DistributedDataParallel model instance
        optimizer: Optional DistribOptimizer instance. If provided, ensures
                   consistency between DDP and optimizer buffer references.
        synchronize: Whether to call torch.cuda.synchronize() before freeing
        empty_cache: Whether to call torch.cuda.empty_cache() after freeing
    
    Returns:
        List of GradBufferOffloadState objects needed to reallocate the buffers.
        One state object per buffer (including expert parallel buffers).
    """
    if synchronize:
        torch.cuda.synchronize()
    
    offload_states = []
    
    # Get all buffers (dense + expert parallel)
    all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers
    all_bucket_groups = ddp_model.bucket_groups + ddp_model.expert_parallel_bucket_groups
    
    # Validate optimizer references point to same buffers if optimizer provided
    if optimizer is not None and hasattr(optimizer, 'buffers') and not getattr(optimizer, 'is_stub_optimizer', False):
        for opt_buffer in optimizer.buffers:
            assert opt_buffer in all_buffers, (
                "Optimizer buffer not found in DDP buffers. "
                "Ensure optimizer and DDP share the same buffer references."
            )
    
    for buffer_idx, buffer in enumerate(all_buffers):
        offload_state = _offload_single_buffer(buffer, all_bucket_groups, buffer_idx)
        offload_states.append(offload_state)
    
    if empty_cache:
        torch.cuda.empty_cache()
    
    return offload_states


def _offload_single_buffer(
    buffer: _ParamAndGradBuffer,
    bucket_groups: List[_ParamAndGradBucketGroup],
    buffer_id: int,
) -> GradBufferOffloadState:
    """Free a single buffer's grad_data from GPU memory.
    
    The tensor data is NOT preserved - only metadata for reallocation is stored.
    """
    
    # Step 1: Collect metadata before clearing references
    bucket_metadata = []
    for bucket in buffer.buckets:
        bucket_metadata.append(BucketMetadata(
            offset=bucket.offset,
            numel=bucket.grad_data.numel(),
        ))
    
    param_grad_metadata = {}
    for param in buffer.params:
        if hasattr(param, 'main_grad') and param.main_grad is not None:
            start_index, end_index, _ = buffer.param_index_map[param]
            param_grad_metadata[param] = ParamGradMetadata(
                start_index=start_index,
                end_index=end_index,
                shape=param.main_grad.shape,
            )
    
    # Step 2: Clear cached shard lists in bucket groups that reference this buffer's buckets
    buffer_bucket_set = set(buffer.buckets)
    for bucket_group in bucket_groups:
        # Check if this bucket group contains buckets from this buffer
        if any(b in buffer_bucket_set for b in bucket_group.buckets):
            for i in range(len(bucket_group.cached_grad_buffer_shard_list)):
                bucket_group.cached_grad_buffer_shard_list[i] = None
    
    # Step 3: Clear bucket grad_data views
    for bucket in buffer.buckets:
        bucket.grad_data = None
    
    # Step 4: Clear param.main_grad views
    for param in buffer.params:
        if hasattr(param, 'main_grad'):
            param.main_grad = None
    
    # Step 5: Store metadata for reallocation (NOT the tensor data)
    numel = buffer.grad_data.numel()
    original_device = buffer.grad_data.device
    grad_dtype = buffer.grad_dtype
    
    # Step 6: Free the GPU tensor
    buffer.grad_data = None
    
    return GradBufferOffloadState(
        numel=numel,
        grad_dtype=grad_dtype,
        original_device=original_device,
        bucket_metadata=bucket_metadata,
        param_grad_metadata=param_grad_metadata,
        buffer_id=buffer_id,
    )


def onload_grad_data(
    ddp_model,
    offload_states: List[GradBufferOffloadState],
    optimizer=None,
    synchronize: bool = True,
) -> None:
    """
    Reallocate grad_data tensors on GPU.
    
    This function allocates fresh buffers and recreates all references:
    1. The main buffer.grad_data tensor (freshly allocated, uninitialized)
    2. bucket.grad_data views in each bucket
    3. param.main_grad views on each parameter
    
    Note: The tensor contents are NOT restored - fresh memory is allocated.
    cached_grad_buffer_shard_list will be recreated lazily on next use.
    
    Args:
        ddp_model: DistributedDataParallel model instance
        offload_states: List of GradBufferOffloadState objects from offload_grad_data
        optimizer: Optional DistribOptimizer instance (for validation)
        synchronize: Whether to call torch.cuda.synchronize() after allocation
    """
    # Get all buffers (dense + expert parallel)
    all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers
    
    assert len(offload_states) == len(all_buffers), (
        f"Number of offload states ({len(offload_states)}) must match "
        f"number of buffers ({len(all_buffers)})"
    )
    
    for offload_state in offload_states:
        buffer = all_buffers[offload_state.buffer_id]
        _onload_single_buffer(buffer, offload_state)
    
    if synchronize:
        torch.cuda.synchronize()


def _onload_single_buffer(
    buffer: _ParamAndGradBuffer,
    offload_state: GradBufferOffloadState,
) -> None:
    """Reallocate a single buffer's grad_data on GPU.
    
    Allocates fresh memory - the tensor contents are NOT restored.
    """
    
    # Step 1: Allocate fresh tensor on GPU (contents are uninitialized)
    buffer.grad_data = torch.empty(
        offload_state.numel,
        dtype=offload_state.grad_dtype,
        device=offload_state.original_device,
    )
    
    # Step 2: Recreate bucket grad_data views
    for bucket, metadata in zip(buffer.buckets, offload_state.bucket_metadata):
        start_index = metadata.offset
        end_index = metadata.offset + metadata.numel
        bucket.grad_data = buffer.grad_data[start_index:end_index]
    
    # Step 3: Recreate param.main_grad views
    for param, metadata in offload_state.param_grad_metadata.items():
        param.main_grad = buffer.grad_data[
            metadata.start_index:metadata.end_index
        ].view(metadata.shape)


def offload_all_grad_data_to_cpu(
    ddp_model,
    optimizer=None,
) -> List[GradBufferOffloadState]:
    """
    Convenience function to free all grad buffers from GPU.
    
    Alias for offload_grad_data with default parameters.
    Note: Despite the name, this does NOT copy data to CPU - it only frees GPU memory.
    """
    return offload_grad_data(ddp_model, optimizer)


def onload_all_grad_data_from_cpu(
    ddp_model,
    offload_states: List[GradBufferOffloadState],
    optimizer=None,
) -> None:
    """
    Convenience function to reallocate all grad buffers on GPU.
    
    Alias for onload_grad_data with default parameters.
    Note: Despite the name, this allocates fresh memory - it does NOT restore from CPU.
    """
    onload_grad_data(ddp_model, offload_states, optimizer)


def get_grad_buffer_memory_usage(ddp_model) -> Dict[str, int]:
    """
    Get the current GPU memory usage of grad buffers.
    
    Args:
        ddp_model: DistributedDataParallel model instance
    
    Returns:
        Dictionary with memory usage information in bytes.
    """
    all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers
    
    total_bytes = 0
    buffer_info = {}
    
    for i, buffer in enumerate(all_buffers):
        if buffer.grad_data is not None:
            numel = buffer.grad_data.numel()
            element_size = buffer.grad_data.element_size()
            buffer_bytes = numel * element_size
            total_bytes += buffer_bytes
            buffer_info[f"buffer_{i}"] = {
                "numel": numel,
                "dtype": str(buffer.grad_dtype),
                "bytes": buffer_bytes,
                "device": str(buffer.grad_data.device),
            }
        else:
            buffer_info[f"buffer_{i}"] = {
                "numel": 0,
                "dtype": str(buffer.grad_dtype),
                "bytes": 0,
                "device": "offloaded",
            }
    
    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "buffers": buffer_info,
    }
