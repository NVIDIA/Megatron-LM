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
import traceback
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from .utils import get_global_memory_buffer, log_single_rank

if TYPE_CHECKING:
    from .param_and_grad_buffer import ParameterGroup

logger = logging.getLogger(__name__)

NCCL_ALLOCATOR = None

try:
    # Try to import the MCore NCCL nccl_allocator first.
    # If it fails, try to import the APEX NCCL nccl_allocator.
    import megatron.core.nccl_allocator as nccl_allocator

    NCCL_ALLOCATOR = "MCORE"
except ImportError:
    try:
        import apex.contrib.nccl_allocator as nccl_allocator

        NCCL_ALLOCATOR = "APEX"
    except ImportError:
        nccl_allocator = None


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


class FixedPoolAllocator(TemporaryBucketAllocator):
    """
    A specialized temporary bucket allocator that implements a buffer recycling strategy
    to minimize memory fragmentation in FSDP operations.

    This allocator maintains a fixed pool of pre-allocated buffers, reusing them
    to reduce the overhead and fragmentation caused by frequent allocation and
    deallocation of temporary buffers during FSDP operations.
    """

    def __init__(
        self,
        name: str,
        fsdp_param_groups: List["ParameterGroup"],
        size: int = 2,
        fallback_to_persistent_buffer: bool = False,
    ):
        self.name = name
        self.fsdp_param_groups = fsdp_param_groups
        self.size = size  # Number of buffers in the pool (default is 2 for double buffering)
        self.allocation_tracker = {}  # tracking the global buffer allocation status

        # Build a mapping from FSDP unit id to its associated bucket ids.
        fsdp_unit_buckets = defaultdict(list)
        for bucket_id, param_group in enumerate(fsdp_param_groups):
            if param_group.fsdp_unit_id == -1 or param_group.fsdp_unit_id is None:
                continue
            fsdp_unit_buckets[param_group.fsdp_unit_id].append(bucket_id)
        self.fsdp_unit_buckets = fsdp_unit_buckets

        # Identify the largest group of FSDP units that share the same buffer storage.
        fsdp_units_to_double_buffer = []
        for fsdp_unit_id, bucket_ids in fsdp_unit_buckets.items():
            same_storage_fsdp_units = []
            for i in fsdp_unit_buckets:
                if self._is_two_bucket_group_equal(fsdp_unit_buckets[i], bucket_ids):
                    same_storage_fsdp_units.append(i)
            # Track the largest group of FSDP units sharing the same buffer storage
            if len(same_storage_fsdp_units) > len(fsdp_units_to_double_buffer):
                fsdp_units_to_double_buffer = same_storage_fsdp_units

        # --- Fixed Pool Buffering Check ---
        # Ensure there is at least one group of FSDP units eligible for fixed pool buffering.
        # If not, the allocator cannot provide its intended memory recycling benefits.
        assert (
            len(fsdp_units_to_double_buffer) > 0
        ), "Found no FSDP units to use fixed-size buffering"
        self.fsdp_double_buffer_units = fsdp_units_to_double_buffer

        if torch.distributed.get_rank() == 0:
            for bucket_id, param_group in enumerate(fsdp_param_groups):
                if (
                    param_group.fsdp_unit_id == -1
                    or param_group.fsdp_unit_id is None
                    or param_group.fsdp_unit_id not in self.fsdp_double_buffer_units
                ):
                    logging.info(
                        f"FSDP unit (id={param_group.fsdp_unit_id}) does not fit "
                        "in FixedPoolAllcator"
                    )
                    if fallback_to_persistent_buffer is False:
                        logging.info(
                            "It will fall back to dynamic memory allocator, NCCL user "
                            "buffer is not supported"
                        )
                    else:
                        logging.info(
                            "It will be allocated a persistent buffer. If the memory "
                            "budget is tight, set "
                            "trainer.strategy.ddp.fsdp_db_use_persist_buf_on_alloc_fail to False."
                        )

        # Initialize buffer group status.
        # Each buffer group represents a set of buffers associated with an FSDP unit's bucket group.
        self.idle_buffer = []  # List of available (buf_group_id, offset) tuples.
        self.using_buffer = {}  # Map from bucket_id to (buf_group_id, offset) in use.

        # Populate the idle buffer pool with all buffer group and bucket offset combinations.
        for buf_group_id in range(self.size):  # Iterate over each buffer group in the pool.
            num_bucket = len(self.fsdp_unit_buckets[self.fsdp_double_buffer_units[0]])
            for bucket_offset in range(num_bucket):
                self.idle_buffer.append((buf_group_id, bucket_offset))

        # Fallback allocator used if the fixed pool allocator cannot fulfill a request.
        self.fallback_to_persistent_buffer = fallback_to_persistent_buffer
        self.backup_allocator = TemporaryBucketAllocator()

    def _is_two_bucket_group_equal(self, group_a, group_b):
        # Check if two bucket groups are equivalent in dtype and size.
        if len(group_a) != len(group_b):
            return False

        for a, b in zip(group_a, group_b):
            pg_a = self.fsdp_param_groups[a]
            pg_b = self.fsdp_param_groups[b]
            a_size = sum(p.numel() for p in pg_a.params)
            b_size = sum(p.numel() for p in pg_b.params)
            if pg_a.dtype != pg_b.dtype or a_size != b_size:
                return False
        return True

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
        fsdp_unit_id = self.fsdp_param_groups[bucket_id].fsdp_unit_id
        if fsdp_unit_id in self.fsdp_double_buffer_units:
            # Try to allocate from the buffer pool.
            bucket_offset = self.fsdp_unit_buckets[fsdp_unit_id].index(bucket_id)
            buffer_name = None
            if bucket_id in self.using_buffer:
                # If this bucket is already using a buffer, reuse it.
                buf_group_id, bucket_offset = self.using_buffer[bucket_id]
                buffer_name = self._get_gbuf_name(buf_group_id, bucket_offset)
            else:
                # Otherwise, find an available buffer group for this bucket offset.
                for buf_group_id in range(self.size):
                    if (buf_group_id, bucket_offset) in self.idle_buffer:
                        self.using_buffer[bucket_id] = (buf_group_id, bucket_offset)
                        buffer_name = self._get_gbuf_name(buf_group_id, bucket_offset)
                        self.idle_buffer.remove((buf_group_id, bucket_offset))
                        break

            assert buffer_name is not None, (
                f"[FSDP][Rank {torch.distributed.get_rank()}][{self.name}] "
                f"No buffer found for bucket_id: {bucket_id}, fsdp_unit_id: {fsdp_unit_id}, "
                f"bucket_offset: {bucket_offset} \n"
                f"current using_buffer: {self.using_buffer} \n"
                f"current idle_buffer: {self.idle_buffer}"
            )
        elif self.fallback_to_persistent_buffer is True:
            buffer_name = f"{self.name}_not_fit_in_fixed_pool_{bucket_id}_{size}_{dtype}_{device}"
        else:
            # If the bucket is not eligible for fixed pool buffering, or no buffer is available,
            # fall back to dynamic allocation via the backup allocator. This means that we
            # will do dynamic memory allocation.
            logging.debug(f"[FSDP] Using backup allocator for {bucket_id} {fsdp_unit_id}")
            return self.backup_allocator.allocate(
                bucket_id=bucket_id, size=size, dtype=dtype, device=device
            )

        # Use buffer_name to get memory from global memory.
        if mem_alloc_context is not None and mem_alloc_context != nullcontext:
            # Check if a new buffer allocation is required
            if (
                self.allocation_tracker.get((buffer_name, dtype), None) is None
                or self.allocation_tracker[(buffer_name, dtype)] < size
            ):
                # Requires synchronization for new buffer allocation
                self.allocation_tracker[(buffer_name, dtype)] = size
                torch.cuda.synchronize()
        return Bucket(
            data=get_global_memory_buffer().get_tensor(
                [size], dtype=dtype, name=buffer_name, mem_alloc_context=mem_alloc_context
            )
        )

    def _get_gbuf_name(self, buf_group_id: int, bucket_index: int):
        return f"{self.name}_{buf_group_id}_{bucket_index}"

    def free(self, bucket_id: int):
        """
        free a temporary bucket.
        """
        fsdp_unit_id = self.fsdp_param_groups[bucket_id].fsdp_unit_id
        if fsdp_unit_id in self.fsdp_double_buffer_units:
            if bucket_id not in self.using_buffer:
                # This bucket is not allocated by fixed pool allocator.
                return
            # Return the buffer to the idle pool.
            self.idle_buffer.append(self.using_buffer[bucket_id])
            del self.using_buffer[bucket_id]
            return
        if self.fallback_to_persistent_buffer is False:
            # If not managed by fixed pool allocator, delegate to the backup allocator.
            logging.debug(f"[FSDP] Free from the backup allocator for {bucket_id} {fsdp_unit_id}")
            self.backup_allocator.free(bucket_id)
