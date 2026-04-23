# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
import operator
import traceback
from contextlib import nullcontext
from functools import reduce
from typing import Any, Callable, Optional

import torch

from .utils import log_single_rank

logger = logging.getLogger(__name__)

NCCL_ALLOCATOR = None

try:
    # Try to import the MCore NCCL nccl_allocator first.
    # If it fails, try to import the APEX NCCL nccl_allocator.
    from megatron.core import nccl_allocator

    NCCL_ALLOCATOR = "MCORE"
except ImportError:
    try:
        from apex.contrib import nccl_allocator

        NCCL_ALLOCATOR = "APEX"
    except ImportError:
        nccl_allocator = None


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context: Optional[Callable] = None):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
            with mem_alloc_context():
                self.buffer[(name, dtype)] = torch.empty(
                    required_len,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    global _GLOBAL_MEMORY_BUFFER
    if "_GLOBAL_MEMORY_BUFFER" not in globals() or _GLOBAL_MEMORY_BUFFER is None:
        _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
    return _GLOBAL_MEMORY_BUFFER


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error
    message ``s`` since otherwise, it is swallowed.
    """
    if not cond:
        logger.error(s)
        logger.error(''.join(traceback.format_stack()))
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDEr",
                )
                tensor._typed_storage()._resize_(size.numel())


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                _p_assert(
                    tensor.storage_offset() == 0,
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}",
                )
                tensor._typed_storage()._resize_(0)


class MultiGroupUBRAllocator:
    """
    A custom allocator class that registers a single memory pool with multiple different
    communication groups, which is not natively supported by apex's nccl_allocator.

    This is particularly useful for Mixture of Experts (MoE) models where:
    - Non-expert parameters/gradients use the data-parallel + context-parallel group (dp_cp_group)
    - Expert parameters/gradients use the expert-parallel + data-parallel group (ep_dp_group)

    Since Megatron-Core FSDP uses a contiguous single tensor for the entire model's parameters, we
    need to register the same memory pool with both communication groups to enable nccl algorithms
    that is relying on the user buffer registration for both expert and non-expert parameters.

    Implementation:
        It uses apex nccl_allocator internally to create a Tensor using ncclMemAlloc
        and register to the `group` and then registers the Mempool also for the `additional_group`

    Example:
        ```
        import apex.contrib.nccl_allocator as nccl_allocator
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool()
        group_1 = torch.distributed.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7], backend="nccl")
        group_2 = torch.distributed.new_group(ranks=[0, 2, 4, 6], backend="nccl")
        with MultiGroupUBRAllocator(pool, groups=[group_1, group_2]):
            a = torch.zeros(1024, dtype=torch.float32, device="cuda")
            b = torch.zeros(1024, dtype=torch.float32, device="cuda")
        ```
    """

    def __init__(self, pool, groups):  # torch.cuda.MemPool  # torch.distributed.ProcessGroup
        self.pool = pool
        self.groups = groups
        self.mem_allocator = nccl_allocator.nccl_mem(self.pool, group=self.groups[0])
        assert len(self.groups) > 1, "MultiGroupUBRAllocator requires at least two groups"

    def __enter__(self):
        for group in self.groups[1:]:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            try:
                # Since the registration is done in mempool granularity, we need to deregister
                # the tensors in the mempool and re-register the mempool including the newly created
                # tensors after the context is exited.
                backend.deregister_mem_pool(self.pool)
            except RuntimeError:
                pass
        self.mem_allocator.__enter__()

    def __exit__(self, *args):
        self.mem_allocator.__exit__(*args)
        for group in self.groups[1:]:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            log_single_rank(
                logger,
                logging.INFO,
                f"[MultiGroupUBRAllocator] Registering mem pool to group {group}, "
                f"group.group_desc:{group.group_desc}",
            )
            backend.register_mem_pool(self.pool)


@dataclasses.dataclass
class Bucket:
    """
    A container for holding data in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        data (torch.Tensor): A tensor containing the data elements
            grouped together in a bucket.
            used to synchronize data operations.

    Note:
        Buckets are used to optimize communication in FSDP training by
            grouping small tensors together.
    """

    data: torch.Tensor


class TemporaryBucketAllocator:
    """
    A utility class for managing temporary buckets (buffers) used in FSDP
    operations like parameters unshard and gradients reduction.

    This allocator handles the dynamic allocation and deallocation of temporary memory buffers
    needed during FSDP (Fully Sharded Data Parallel) operations, particularly for parameters
    unshard and gradients reduction. It helps optimize memory usage by allowing temporary
    buckets to be released when no longer needed.

    Key Features:
        - Dynamic allocation of temporary buckets for FSDP operations
        - Memory-efficient management of temporary buffers
        - Support for both parameters unshard and gradients reduction operations
        - Automatic cleanup of unused buckets to save memory

    Usage:
        ```python
        # Create an allocator instance
        allocator = TemporaryBucketAllocator(name="gpt_parameters")

        # Allocate a temporary bucket
        temp_bucket = allocator.allocate(size=1024, dtype=torch.float32)

        # Use the temporary bucket for FSDP operations
        # ... perform all-gather or reduce-scatter ...

        # Free the bucket when done
        allocator.free(temp_bucket)
        ```

    Note:
        It's important to release temporary buckets after use to prevent memory leaks
        and optimize memory usage during training.
    """

    def __init__(self):
        self.buckets = {}

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        return self.buckets[bucket_id]

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)
            del self.buckets[bucket_id]


class StorageResizeBasedBucketAllocator(TemporaryBucketAllocator):
    """
    A specialized temporary bucket allocator that resizes the storage of temporary buckets
    based on the required size.
    """

    def __init__(self):
        super().__init__()

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        bucket = self.buckets[bucket_id]
        _alloc_storage(bucket.data, torch.Size([size]))
        return bucket

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)


class RotaryBucketAllocator(TemporaryBucketAllocator):
    """A specialized temporary bucket allocator that implements a circular buffer recycling strategy
    to minimize memory fragmentation in FSDP operations.

    RotaryBucketAllocator extends TemporaryBucketAllocator by maintaining a limited pool of
    pre-allocated buffers that are reused in a circular manner. This approach helps prevent
    memory fragmentation that typically occurs with frequent allocation and deallocation of
    temporary buffers during FSDP operations.

    Key Features:
        - Circular buffer recycling strategy for memory efficiency
        - Reduced memory fragmentation compared to dynamic allocation
        - Pre-allocated buffer pool for faster access
        - Automatic buffer reuse without explicit deallocation

    Usage:
        ```python
        # Create a rotary allocator
        allocator = RotaryBucketAllocator(name="gpt_parameters")

        # Get a temporary buffer from the pool
        temp_bucket = allocator.allocate(dtype=torch.float32)

        # Use the temporary bucket for FSDP operations
        # ... perform all-gather or reduce-scatter ...

        # Free the bucket when done, make it in idle buffer pool
        allocator.free(temp_bucket)
        ```
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.num_global_buffer = 0
        self.idle_buffer = []  # [buffer_id]
        self.using_buffer = {}  # {bucket_id: buffer_id}

    def allocate(
        self,
        bucket_id: int,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_alloc_context: Optional[Callable] = None,
    ) -> Bucket:
        """
        allocate a temporary bucket.
        """

        def _get_global_buffer(buffer_id: int):
            return get_global_memory_buffer().get_tensor(
                [size],
                dtype=dtype,
                name=self._get_gbuf_name(buffer_id),
                mem_alloc_context=mem_alloc_context,
            )

        if bucket_id in self.using_buffer:
            buffer_id = self.using_buffer[bucket_id]
            return Bucket(data=_get_global_buffer(buffer_id))

        if len(self.idle_buffer) == 0:
            # allocate new buffer
            buffer_id = self.num_global_buffer
            self.num_global_buffer += 1
            self.idle_buffer.append(buffer_id)

        buffer_id = self.idle_buffer.pop(0)
        self.using_buffer[bucket_id] = buffer_id
        return Bucket(data=_get_global_buffer(buffer_id))

    def _get_gbuf_name(self, buffer_id: int):
        return f"{self.name}_{buffer_id}"

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        if bucket_id in self.using_buffer:
            buffer_id = self.using_buffer.pop(bucket_id)
            self.idle_buffer.append(buffer_id)
