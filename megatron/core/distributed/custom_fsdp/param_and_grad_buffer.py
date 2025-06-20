# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import functools
import gc
import inspect
import logging
import math
import traceback
import warnings
from collections import defaultdict, namedtuple
from contextlib import ExitStack, nullcontext
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.distributed import _coalescing_manager

from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor, modify_underlying_storage, quantize_param_shard
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.utils import is_submodule, is_te_min_version, log_on_each_pipeline_stage

try:
    from transformer_engine.pytorch import fp8_model_init
except:
    pass

try:
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
except:
    pass

try:
    import apex.contrib.nccl_allocator as nccl_allocator
except ImportError:
    nccl_allocator = None
NCCL_MEMORY_POOL = None


logger = logging.getLogger(__name__)


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error
    message ``s`` since otherwise, it is swallowed.
    """
    if not cond:
        print(s)
        traceback.print_stack()
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


TensorItemIndex = namedtuple(
    'TensorItemIndex', ['global_data_index', 'size', 'item_id', 'bucket_id', 'shape']
)
BucketIndex = namedtuple('BucketIndex', ['bucket_id', 'global_data_index', 'size', 'items'])
ShardBucketIndex = namedtuple(
    'ShardBucketIndex',
    ['bucket_id', 'global_data_index', 'local_data_index', 'bucket_data_index', 'size'],
)


class DualUBRAllocator:
    """
    A custom allocator class that registers a single memory pool with two different
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
        with DualUBRAllocator(pool, group_1, group_2):
            a = torch.zeros(1024, dtype=torch.float32, device="cuda")
            b = torch.zeros(1024, dtype=torch.float32, device="cuda")
        ```
    """

    def __init__(
        self,
        pool,  # torch.cuda.MemPool
        group,  # torch.distributed.ProcessGroup
        additional_group,  # torch.distributed.ProcessGroup
    ):
        self.pool = pool
        self.group = group
        self.additional_group = additional_group
        self.mem_allocator = nccl_allocator.nccl_mem(self.pool, group=self.group)

    def __enter__(self):
        backend = self.additional_group._get_backend(
            torch.device("cuda", torch.cuda.current_device())
        )
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
        backend = self.additional_group._get_backend(
            torch.device("cuda", torch.cuda.current_device())
        )
        backend.register_mem_pool(self.pool)


@dataclasses.dataclass
class BucketingPolicy:
    """
    A policy for bucketing in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        suggested_bucket_size (int): The suggested size of each bucket in num of elements.
        fsdp_unit_modules (list): A list of module classes that are treated as a
            single unit for FSDP bucketing.
        data_parallel_sharding_strategy (str): The strategy used for sharding
            data parallel modules.

    Note:
        This policy is used to configure the bucketing behavior in FSDP training.
    """

    suggested_bucket_size: Optional[int] = 40_000_000
    fsdp_unit_modules: List[torch.nn.Module] = dataclasses.field(default_factory=list)
    data_parallel_sharding_strategy: str = 'no_shard'


def _pad(number_to_be_padded: int, divisor: int) -> int:
    return int(math.ceil(number_to_be_padded / divisor) * divisor)


def build_data_parallel_buffer_index(
    elements: List[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    is_data_distributed: bool,
    ddp_config: DistributedDataParallelConfig,
    bucket_id: int = 0,
) -> Tuple[List[TensorItemIndex], BucketIndex, ShardBucketIndex]:
    """
    Assuming that all input tensor elements are consecutively compose a global
    buffer, give the index range of every tensor,  every bucket and every in
    bucket local buffer.

    Args:
        elements (List[torch.Size]): List of input tensor.
        data_parallel_rank (int): Rank of the current process in the data parallel group.
        data_parallel_world_size (int): World size of the data parallel group.
        bucket_id (int, optional): The id of the bucket. Defaults to 0.

    Returns:
        Tuple[List[TensorItemIndex], BucketIndex, ShardBucketIndex]:  The index
            range of every tensor, every bucket and every in bucket local buffer.
    """

    def _pad_if_needed(data_index: int) -> int:
        """
        Pads data indices if using distributed optimizer (to ensure uniform sharding).
        """
        if ddp_config.data_parallel_sharding_strategy != 'no_shard':
            # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
            # This also helps cuBLAS pick more efficient algorithms for GEMMs.
            # We now ensure that all buckets start at a memory address that is 256-byte
            # aligned (128 values since params and grads use >= 16-bit precision).
            return _pad(data_index, math.lcm(data_parallel_world_size, 128))
        return data_index

    def add_item(item_id, item, bucket, item_index_map, bucket_id):
        bucket.append(item)
        bucket_size = sum([it.numel() for it in bucket])
        item_index_map.append(
            TensorItemIndex(
                data_index + bucket_size - item.numel(),
                item.numel(),
                item_id=item_id,
                bucket_id=bucket_id,
                shape=item,
            )
        )

    item_index_map = []
    bucket = []
    data_index = 0
    for item_id, item in enumerate(elements):
        add_item(item_id, item, bucket, item_index_map, bucket_id)

    bucket_size = sum([it.numel() for it in bucket])
    bucket_size = _pad_if_needed(bucket_size)
    bucket_index = BucketIndex(
        bucket_id,
        data_index,
        bucket_size,
        items=list(filter(lambda x: x.bucket_id == bucket_id, item_index_map)),
    )

    shard_size = bucket_index.size // data_parallel_world_size
    bucket_data_index = shard_size * data_parallel_rank
    global_data_index = bucket_index.global_data_index + bucket_data_index

    if is_data_distributed:
        shard_bucket_index = ShardBucketIndex(
            bucket_id, global_data_index, 0, bucket_data_index, shard_size
        )
    else:
        shard_bucket_index = ShardBucketIndex(
            bucket_id, global_data_index, global_data_index, bucket_data_index, shard_size
        )

    return item_index_map, bucket_index, shard_bucket_index


@dataclasses.dataclass
class Bucket:
    """
    A container for holding data in Fully Sharded Data Parallel (FSDP) training.

    Attributes:
        data (torch.Tensor): A tensor containing the data elements
            grouped together in a bucket.
        data_operation_event (Optional[torch.cuda.Event]): An optional CUDA event
            used to synchronize data operations.
        status (Any): An optional status object used to track the state of the bucket.

    Note:
        Buckets are used to optimize communication in FSDP training by
            grouping small tensors together.
    """

    data: torch.Tensor
    data_operation_event: Optional[torch.cuda.Event] = None
    status: Any = None


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
        self.buckets = {}  # {bucket_id: Bucket}

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
        self.name = name
        self.num_global_buffer = 0
        self.idle_buffer = []  # [buffer_id]
        self.using_buffer = {}  # {bucket_id: buffer_id}
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

        def _get_global_buffer(buffer_id: int):
            return parallel_state.get_global_memory_buffer().get_tensor(
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

    def __init__(self, name: str, fsdp_param_groups: List["ParameterGroup"], size: int = 2):
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
                f"[FSDP][Rank {torch.distributed.get_rank()}][{self.name}]"
                f"No buffer found for bucket_id: {bucket_id}, fsdp_unit_id: {fsdp_unit_id}, "
                f"bucket_offset: {bucket_offset} \n"
                f"current using_buffer: {self.using_buffer} \n"
                f"current idle_buffer: {self.idle_buffer}"
            )
            # Synchronization is required before the allocation for the user buffer
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
                data=parallel_state.get_global_memory_buffer().get_tensor(
                    [size], dtype=dtype, name=buffer_name, mem_alloc_context=mem_alloc_context
                )
            )

        # If the bucket is not eligible for fixed pool buffering, or no buffer is available,
        # fall back to dynamic allocation via the backup allocator. This means that we
        # will do dynamic memory allocation.
        logging.debug(f"[FSDP] Using backup allocator for {bucket_id} {fsdp_unit_id}")
        return self.backup_allocator.allocate(
            bucket_id=bucket_id, size=size, dtype=dtype, device=device
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
        # If not managed by fixed pool allocator, delegate to the backup allocator.
        logging.debug(f"[FSDP] Free from the backup allocator for {bucket_id} {fsdp_unit_id}")
        self.backup_allocator.free(bucket_id)


class DataParallelBuffer:
    """
    A class that manages the data parallel buffer for Fully Sharded Data Parallel (FSDP) training.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        is_data_distributed: bool,
        bucket_id: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        inter_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        temporary_bucket_allocator: Optional[TemporaryBucketAllocator] = None,
        init_meta_only: bool = False,
        is_dtype_float8: bool = False,
        gradient_scaling_factor: Optional[float] = None,
        mem_alloc_context: Optional[Callable] = None,
    ) -> None:
        self.ddp_config = ddp_config
        self.params = params
        _param_dtype = {p.dtype for p in self.params}
        assert len(_param_dtype) == 1, f'params have different dtypes: {_param_dtype}'
        self.is_data_distributed = is_data_distributed
        self.bucket_id = bucket_id
        self.dtype = dtype if dtype else next(iter(_param_dtype))
        self.device = device
        self.data_parallel_group = data_parallel_group
        self.inter_data_parallel_group = inter_data_parallel_group
        self.dp_rank = self.data_parallel_group.rank()
        self.dp_world_size = self.data_parallel_group.size()
        self.temporary_bucket_allocator = (
            temporary_bucket_allocator if temporary_bucket_allocator else TemporaryBucketAllocator()
        )
        self.is_dtype_float8 = is_dtype_float8
        self.gradient_scaling_factor = gradient_scaling_factor
        self.mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext

        (self.item_index_map, self.bucket_index, self.shard_bucket_index) = (
            build_data_parallel_buffer_index(
                [p.shape for p in self.params],
                self.dp_rank,
                self.dp_world_size,
                is_data_distributed,
                ddp_config,
                bucket_id=bucket_id,
            )
        )

        self.data_size = (
            self.bucket_index.size if not is_data_distributed else self.shard_bucket_index.size
        )
        if init_meta_only:
            self.data = None
        else:
            self.data = torch.empty(self.data_size, dtype=self.dtype, device=device)

        self.param_idx = {p: i for i, p in enumerate(self.params)}
        self.placeholder_bucket = None
        self.placeholder_items = {}

    def fetch_bucket(
        self, dtype: Optional[torch.dtype] = None, and_allocate_params_data: bool = False
    ) -> Bucket:
        """
        Fetch a communication buffer for data-parallel operations.

        The size of the bucket is defined by the `DataParallelBuffer` instance.
        If `and_allocate_params_data` is True, this method resets the parameter
        data stored in the `DataParallelBuffer` instance.

        Args:
            dtype (Optional[torch.dtype], optional): The data type of the tensor
                to fetch a buffer for. Defaults to None.
            and_allocate_params_data (bool, optional): Whether to allocate and
                reset parameter data. Defaults to False.

        Returns:
            Bucket: The communication buffer for the specified data type.
        """
        if dtype is None:
            dtype = self.dtype
        bucket_index = self.bucket_index

        if not self.is_data_distributed and dtype == self.dtype:
            bucket = Bucket(
                data=self.data[
                    bucket_index.global_data_index : bucket_index.global_data_index
                    + bucket_index.size
                ]
            )
        else:
            bucket = self.temporary_bucket_allocator.allocate(
                bucket_id=bucket_index.bucket_id,
                size=bucket_index.size,
                dtype=dtype,
                device=self.device,
                mem_alloc_context=self.mem_alloc_context,
            )

            if and_allocate_params_data:
                for p in self.params:
                    item_id = self.param_idx[p]
                    if is_float8tensor(p):
                        p._data = self.get_item_from_bucket(bucket, item_id).view(p.shape)
                    else:
                        p.data = self.get_item_from_bucket(bucket, item_id).view(p.shape)

        return bucket

    def free_bucket_storage(self, and_free_params_data: bool = False):
        """
        Release the storage of a temporary communication bucket.

        If the bucket is temporary, this method frees its storage.
        If `and_free_params_data` is True, this method also releases the storage
            of the parameter data stored in the `DataParallelBuffer` instance.

        Args:
            and_free_params_data (bool, optional): Whether to also release the
                storage of the parameter data. Defaults to False.

        Returns:
            None
        """
        if not self.is_data_distributed:
            return

        self.temporary_bucket_allocator.free(self.bucket_index.bucket_id)
        if and_free_params_data:
            if self.placeholder_bucket is None:
                self.placeholder_bucket = Bucket(
                    data=torch.empty(self.bucket_index.size, dtype=self.dtype, device=self.device)
                )
                for p in self.params:
                    item_id = self.param_idx[p]
                    self.placeholder_items[item_id] = self.get_item_from_bucket(
                        self.placeholder_bucket, item_id
                    ).view(p.shape)
                _free_storage(self.placeholder_bucket.data)
            for p in self.params:
                item_id = self.param_idx[p]
                if is_float8tensor(p):
                    p._data = self.placeholder_items[item_id]
                else:
                    p.data = self.placeholder_items[item_id]

    def _get_item_slice_in_shard(self, item_id: int) -> Tuple[int, int]:
        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index

        item_global_start = item_index.global_data_index
        item_global_end = item_index.global_data_index + item_index.size
        shard_bucket_start = shard_bucket_index.global_data_index
        shard_bucket_end = shard_bucket_index.global_data_index + shard_bucket_index.size

        if item_global_start > shard_bucket_end or item_global_end < shard_bucket_start:
            return (0, 0)

        start = max(item_global_start, shard_bucket_start) - item_global_start
        end = min(item_global_end, shard_bucket_end) - item_global_start

        return (start, end)

    # pylint: disable=missing-function-docstring
    def locate_item_in_global_item(self, item_id: int) -> Tuple[int, int]:
        item_index = self.item_index_map[item_id]
        if not self.is_data_distributed:
            return (0, item_index.size)

        slice_start, slice_end = self._get_item_local_shard_index(item_id)
        if slice_start == slice_end:
            return (0, 0)

        local_shard_index_to_global_index_offset = (
            self.shard_bucket_index.global_data_index - self.shard_bucket_index.local_data_index
        )
        slice_start += local_shard_index_to_global_index_offset
        slice_end += local_shard_index_to_global_index_offset
        return (
            slice_start - item_index.global_data_index,
            slice_end - item_index.global_data_index,
        )

    def _get_item_local_shard_index(self, item_id: int) -> Tuple[int, int]:
        slice_start, slice_end = self._get_item_slice_in_shard(item_id)
        if slice_start == slice_end:
            return (0, 0)

        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index
        offset = (
            item_index.global_data_index
            - shard_bucket_index.global_data_index
            + shard_bucket_index.local_data_index
        )

        return (offset + slice_start, offset + slice_end)

    def _get_item_local_index(self, item_id: int) -> Tuple[int, int]:
        if not self.is_data_distributed:
            item_index = self.item_index_map[item_id]
            return (item_index.global_data_index, item_index.global_data_index + item_index.size)

        return self._get_item_local_shard_index(item_id)

    def set_item(self, item_id: int, item: torch.Tensor) -> None:
        """
        Update a tensor item managed by the `DataParallelBuffer` instance.

        The storage of the item is mapped to the communication bucket.
        This method updates the item data and ensures consistency with the bucket.

        Args:
            item_id (int): The ID of the tensor item to update.
            item (torch.Tensor): The original tensor to be put into the buffer.

        Returns:
            None
        """
        if is_float8tensor(item):
            item_data = item._data
        else:
            item_data = item.data

        if self.is_data_distributed:
            slice_start, slice_end = self._get_item_slice_in_shard(item_id)
            item_data = item_data.flatten()[slice_start:slice_end]
        local_index_start, local_index_end = self._get_item_local_index(item_id)
        shard = self.data[local_index_start:local_index_end]
        if shard.numel() > 0:
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, only_shard: bool = False) -> torch.Tensor:
        """
        Retrieve a tensor item managed by the `DataParallelBuffer` instance.

        The storage of the item is mapped to the communication bucket.
        If `only_shard` is True, returns only the shard of the item corresponding
            to the current process.
        Otherwise, returns the entire item.

        Args:
            item_id (int): The ID of the tensor item to retrieve.
            only_shard (bool, optional): Whether to return only the shard of the
                item. Defaults to False.

        Returns:
            torch.Tensor: The retrieved tensor item.
        """
        if only_shard:
            start, end = self._get_item_local_shard_index(item_id)
        else:
            start, end = self._get_item_local_index(item_id)

        return self.data[start:end]

    def get_item_from_bucket(self, bucket: Bucket, item_id: int):
        """get item from bucket."""
        item_index = self.item_index_map[item_id]
        bucket_index = self.bucket_index
        start_index = item_index.global_data_index - bucket_index.global_data_index
        end_index = start_index + item_index.size
        item = bucket.data[start_index:end_index]
        return item

    def get_shard_from_bucket(self, bucket: Bucket):
        """Get the local sharding of the bucket."""
        shard_bucket_index = self.shard_bucket_index
        offset = shard_bucket_index.bucket_data_index
        shard_size = shard_bucket_index.size
        shard = bucket.data[offset : offset + shard_size]
        return shard

    def get_shard_from_local_buffer(self) -> torch.Tensor:
        """Get the local sharding of the bucket."""
        index = self.shard_bucket_index
        return self.data[index.local_data_index : index.local_data_index + index.size]


@dataclasses.dataclass
class ParameterGroup:
    """
    A group of model parameters with associated metadata for data-parallel training.

    This dataclass encapsulates a list of PyTorch parameters and additional information
    necessary for managing data-parallel operations, such as data type, gradient requirements,
    and buffer assignments.
    """

    params: List[torch.nn.Parameter]
    dtype: Optional[torch.dtype] = None
    is_expert_param: bool = False
    requires_grad: Optional[bool] = None
    fsdp_unit_id: Optional[int] = None
    data_parallel_world_size: Optional[int] = None
    model_weight_buffer: Optional[DataParallelBuffer] = None
    main_weight_buffer: Optional[DataParallelBuffer] = None
    main_grad_buffer: Optional[DataParallelBuffer] = None


def _get_parameter_groups(
    module: torch.nn.Module,
    policy: BucketingPolicy,
    meta_device_init_fp8_params: dict,
    bucket_group_by_fsdp_unit: bool = True,
):
    """
    Get the parameter group for the given module and parameters.
    """
    param_to_name = {p: name for name, p in module.named_parameters()}
    fsdp_units = []
    if policy.fsdp_unit_modules:
        param_to_id = {}
        for i, p in enumerate(module.parameters()):
            param_to_id[p] = i
        fsdp_modules = []
        for m in module.modules():
            # Skip nested FSDP module.
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue
            if isinstance(m, tuple(policy.fsdp_unit_modules)):
                fsdp_units.append([param_to_name[p] for p in m.parameters()])
                fsdp_modules.append(m)

    def _does_param_require_new_bucket(param):
        """
        Split shared embedding parameters into separate bucket if using distributed
        optimizer that makes use of reduce-scatters instead of all-reduces.
        This ensures that the first and last pipeline stage partition optimizer state
        for the shared embedding parameters the same way across DP replicas, allowing
        the DP reduce-scatter to be before the embedding all-reduce.
        """
        return (
            getattr(param, "shared_embedding", False)
            and policy.data_parallel_sharding_strategy != "no_shard"
        )

    is_expert_parameter = lambda p: not getattr(p, 'allreduce', True)

    # Step 1: Group the parameters according to their execution order and attributes.
    parameter_groups = []
    for name, param in module.named_parameters():
        param_attrs = dict(
            dtype=(
                "float8"
                if is_float8tensor(param) or meta_device_init_fp8_params.get(name, False)
                else param.dtype
            ),
            is_expert_param=is_expert_parameter(param),
            requires_grad=param.requires_grad,
            fsdp_unit_id=None,
        )
        for fsdp_unit_id, fsdp_unit in enumerate(fsdp_units):
            if name in fsdp_unit:
                param_attrs["fsdp_unit_id"] = fsdp_unit_id
                break

        found_group = False
        for param_group in parameter_groups:
            group_attrs = {
                key: value for key, value in param_group.__dict__.items() if key in param_attrs
            }
            if group_attrs == param_attrs:
                param_group.params.append(param)
                found_group = True
                break

        if not found_group:
            parameter_groups.append(ParameterGroup([param], **param_attrs))

    # Step 2: Bucket the parameters based on the guide bucket size.
    suggested_bucket_size = policy.suggested_bucket_size
    bucket_groups = []
    for group in parameter_groups:
        bucket = []

        basic_attrs = {
            key: value
            for key, value in group.__dict__.items()
            if key in ['dtype', 'is_expert_param', 'requires_grad', 'fsdp_unit_id']
        }
        for param in group.params:
            if _does_param_require_new_bucket(param):
                if len(bucket) > 0:
                    bucket_groups.append(ParameterGroup(bucket, **basic_attrs))
                bucket_groups.append(ParameterGroup([param], **basic_attrs))
                bucket = []
                continue

            bucket.append(param)
            if (
                group.fsdp_unit_id is None
                and suggested_bucket_size
                and sum([p.numel() for p in bucket]) >= suggested_bucket_size
            ):
                bucket_groups.append(ParameterGroup(bucket, **basic_attrs))
                bucket = []
                continue

        if bucket:
            bucket_groups.append(ParameterGroup(bucket, **basic_attrs))

    param_to_param_group = {}
    for group_id, group in enumerate(bucket_groups):
        for param in group.params:
            param_to_param_group[param] = group_id

    # Generate the groups of collective buckets, where each group aggregates
    # the collectives per FSDP unit. This improves performance by reducing
    # the number of collective calls and increasing per-collective efficiency.
    #
    # Set default aggregate buckets of bucket.
    bucket_to_bucket_group = {}
    for bucket_id in range(len(bucket_groups)):
        bucket_to_bucket_group[bucket_id] = [bucket_id]

    # Set aggregate buckets by FSDP units.
    if bucket_group_by_fsdp_unit:
        bucket_group_map = {}
        for bucket_id, param_group in enumerate(bucket_groups):
            if param_group.fsdp_unit_id is None:
                continue
            id = (param_group.fsdp_unit_id, param_group.is_expert_param)
            if id not in bucket_group_map:
                bucket_group_map[id] = []
            bucket_group_map[id].append(bucket_id)
        for bucket_group in bucket_group_map.values():
            for bucket_id in bucket_group:
                bucket_to_bucket_group[bucket_id] = bucket_group

    return (bucket_groups, param_to_param_group, bucket_to_bucket_group)


class ParamAndGradBuffer:
    """A class that manages parameter grouping, buffer allocation, and
    communication operations for data-parallel distributed training.

    This class provides functionality to:
    1. Group parameters based on their data types and communication group sizes
    2. Create contiguous buffers for model weights, gradients, and high-precision
        main weights
    3. Handle parameter unsharding, gradient reduction, and weight
        synchronization operations

    Key Features:
        - Efficient parameter grouping based on data types and communication patterns
        - Memory-efficient contiguous buffer allocation
        - Support for mixed-precision training with main weights
        - Distributed operations including parameters all-gather and gradients
            reduce-scatter/all-reduce
        - Synchronized weight updates between model and main weights

    Note:
        This class is designed for distributed training scenarios where efficient
        parameter management and communication are crucial for performance.

    Args:
        ddp_config (DistributedDataParallelConfig): The distributed data parallel
            configuration.
        module (torch.nn.Module): The module whose parameters are to be grouped
            and flatten.
        bucketing_policy (BucketingPolicy): The bucketing policy.
        data_parallel_group (torch.distributed.ProcessGroup): The data parallel group.
        expert_data_parallel_group (Optional[torch.distributed.ProcessGroup]):
            The expert data parallel group.
        preserve_fp32_weights (bool): Whether to preserve FP32 weights.
        grad_reduce_in_fp32 (bool): Whether to reduce gradients in FP32.
        gradient_scaling_factor (Optional[float]): The gradient scaling factor.
        expert_gradient_scaling_factor (Optional[float]): The expert gradient
            scaling factor.
        device (torch.device): The parameter and gradient buffer device.
        only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad (bool):
            Whether to only create the gradient buffer and main weight buffer
            for parameters that require gradients. Default is True.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        bucketing_policy: BucketingPolicy,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        inter_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        preserve_fp32_weights: bool = True,
        grad_reduce_in_fp32: bool = True,
        gradient_scaling_factor: Optional[float] = None,
        expert_gradient_scaling_factor: Optional[float] = None,
        device: torch.device = torch.device('cuda'),
        only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad: bool = True,
        reset_parameters_for_meta_device_init_module: bool = False,
    ):
        self.ddp_config = ddp_config
        self.module = module
        self.bucketing_policy = bucketing_policy
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.preserve_fp32_weights = preserve_fp32_weights
        self.grad_reduce_in_fp32 = grad_reduce_in_fp32
        self.data_parallel_group = data_parallel_group
        self.expert_data_parallel_group = expert_data_parallel_group
        self.inter_data_parallel_group = inter_data_parallel_group
        self.params = list(module.parameters())
        self.gradient_scaling_factor = gradient_scaling_factor
        self.expert_gradient_scaling_factor = expert_gradient_scaling_factor
        self.device = device
        self.only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad = (
            only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad
        )
        self.reset_parameters_for_meta_device_init_module = (
            reset_parameters_for_meta_device_init_module
        )

        # User buffer registration related settings
        if self.ddp_config.nccl_ub:
            # Since the user buffer registration requires (non-dynamic) persistent memory,
            # it always uses fsdp double buffer.
            self.ddp_config.fsdp_double_buffer = True
            # Initialize the NCCL memory pool.
            global NCCL_MEMORY_POOL
            NCCL_MEMORY_POOL = nccl_allocator.create_nccl_mem_pool()
            if torch.distributed.get_rank() == 0:
                logging.info(
                    f"[Rank {torch.distributed.get_rank()}] Created NCCL memory pool for \
                        UserBuffer Registration"
                )
                logging.info(
                    f"[Rank {torch.distributed.get_rank()}] FSDP double buffer is enabled."
                )
        # If using nccl_ub, it returns a function that registers buffers to the NCCL memory pool
        # Buffer is registered to data_parallel_group and expert_data_parallel_group if it exists
        # In the case of not using nccl_ub, it returns a nullcontext
        self.mem_alloc_context = self.get_mem_alloc_context(
            group=self.data_parallel_group, additional_group=self.expert_data_parallel_group
        )

        # Mark fp8 param.
        meta_device_init_fp8_params = {}
        if reset_parameters_for_meta_device_init_module:
            for m in module.modules():
                if not isinstance(m, TransformerEngineBaseModule):
                    continue
                for name, param in m.named_parameters(recurse=False):
                    # The fp8 param initialized from the meta device may NOT be
                    # an fp8 tensor, according to the internal logic of the TE
                    # to determine whether this parameter is fp8 or not.
                    fp8_meta_index = m.param_init_meta[name].fp8_meta_index
                    if m.primary_weights_in_fp8 and fp8_meta_index is not None:
                        meta_device_init_fp8_params[self.param_to_name[param]] = True

        # Get the parameter groups.
        (self.parameter_groups, self.param_to_param_group, self.bucket_to_bucket_group) = (
            _get_parameter_groups(module, bucketing_policy, meta_device_init_fp8_params)
        )
        self._init_each_parameter_group_buffers(meta_device_init_fp8_params)

        # Initialize the optimizer named parameters.
        self.optimizer_named_parameters = self._init_optimizer_named_parameters()

        self._log_parameter_groups()

    def get_mem_alloc_context(self, group=None, additional_group=None):
        """
        Get the memory allocation context for the parameter and gradient buffers.
        """
        if self.ddp_config.nccl_ub:
            assert nccl_allocator is not None, "NCCL allocator is not available."
            global NCCL_MEMORY_POOL
            if group is None:
                # data parallel group is a default group for user buffer registration
                group = self.data_parallel_group
            if additional_group is None:
                # register buffers to the default group directly using apex memory allocator
                mem_alloc_context = functools.partial(
                    nccl_allocator.nccl_mem, NCCL_MEMORY_POOL, group=group
                )
            else:
                # In case of MoE, we need to register buffer to both DP and EP communicator groups.
                # Custom DualUBRAllocator class is used to register buffers to both groups.
                # Register buffers to the data_parallel_group using apex memory allocator
                # and register buffers to the expert_data_parallel_group.
                assert group != additional_group, "Group and additional group must be different."
                mem_alloc_context = functools.partial(
                    DualUBRAllocator,
                    NCCL_MEMORY_POOL,
                    group=group,
                    additional_group=additional_group,
                )
            return mem_alloc_context
        else:
            return nullcontext

    def _log_parameter_groups(self):
        """
        Log the parameter groups for all pipeline stages.
        """
        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            bucket_groups = self.parameter_groups
            param_to_name = self.param_to_name
            log_strs = []
            log_strs.append(f'Number of parameter groups for FSDP: {len(bucket_groups)}')
            for index, group in enumerate(bucket_groups):
                numel = 0
                for param in group.params:
                    numel += param.numel()
                log_strs.append(
                    f"Params for group {index+1} ({numel} elements, dtype: {group.dtype}, "
                    f"fsdp_unit_id: {group.fsdp_unit_id}, "
                    f"has_weight_buffer: {group.model_weight_buffer is not None}, "
                    f"has_grad_buffer: {group.main_grad_buffer is not None}, "
                    f"has_main_weight_buffer: {group.main_weight_buffer is not None}):"
                )
                for param in group.params:
                    log_strs.append(f'\t{param_to_name[param]}')
            log_on_each_pipeline_stage(logger, logging.INFO, '\n'.join(log_strs))

    def _init_each_parameter_group_buffers(self, meta_device_init_fp8_params):
        """
        Initialize the buffers for each parameter group.
        """
        data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        if data_parallel_sharding_strategy == 'no_shard':
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = False
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == 'optim':
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == 'optim_grads':
            is_model_weight_buffer_distributed = False
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        elif data_parallel_sharding_strategy == 'optim_grads_params':
            is_model_weight_buffer_distributed = True
            is_main_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        else:
            raise ValueError(
                f'Invalid data_parallel_sharding_strategy: {data_parallel_sharding_strategy}'
            )
        if self.ddp_config.nccl_ub:
            assert self.ddp_config.fsdp_double_buffer, (
                "NCCL UB is only supported with FSDP double buffer. "
                "Please set fsdp_double_buffer=True in the ddp config."
            )
        if self.ddp_config.fsdp_double_buffer:
            UB_BUFFER_NUM = 2
            self.weight_alloc = FixedPoolAllocator(
                name="fsdp_params", fsdp_param_groups=self.parameter_groups, size=UB_BUFFER_NUM
            )
            self.main_grad_alloc = FixedPoolAllocator(
                name="fsdp_grads", fsdp_param_groups=self.parameter_groups, size=UB_BUFFER_NUM
            )
            self.double_buf_units = self.weight_alloc.fsdp_double_buffer_units
        else:
            self.weight_alloc = StorageResizeBasedBucketAllocator()
            self.main_grad_alloc = None

        self.buffer_all_in_one = True

        preserve_fp32_weights = self.preserve_fp32_weights
        grad_reduce_in_fp32 = self.grad_reduce_in_fp32
        buffer_size = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0, "float8": 0}
        for group_id, group in enumerate(self.parameter_groups):
            dp_group = (
                self.data_parallel_group
                if not group.is_expert_param
                else self.expert_data_parallel_group
            )
            group.data_parallel_world_size = dp_group.size()
            gradient_scaling_factor = (
                self.gradient_scaling_factor
                if not group.is_expert_param
                else self.expert_gradient_scaling_factor
            )
            one_param = group.params[0]
            is_dtype_float8 = is_float8tensor(one_param) or meta_device_init_fp8_params.get(
                self.param_to_name[one_param], False
            )
            if is_dtype_float8:
                param_dtype = torch.uint8
                grad_dtype = torch.bfloat16
            else:
                param_dtype = group.params[0].dtype
                grad_dtype = param_dtype
            should_create_grad_buffer_or_main_weight_buffer = (
                not self.only_create_grad_buffer_and_main_weight_buffer_for_param_requires_grad
                or group.requires_grad
            )
            # Initialize the model weight buffer.
            if data_parallel_sharding_strategy != 'no_shard':
                group.model_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_model_weight_buffer_distributed
                    and group.data_parallel_world_size > 1,
                    dtype=param_dtype,
                    device=self.device,
                    data_parallel_group=dp_group,
                    inter_data_parallel_group=self.inter_data_parallel_group,
                    init_meta_only=True,
                    is_dtype_float8=is_dtype_float8,
                    temporary_bucket_allocator=self.weight_alloc,
                    bucket_id=group_id,
                    mem_alloc_context=self.mem_alloc_context,
                )

            # Initialize the main weight buffer.
            if should_create_grad_buffer_or_main_weight_buffer and preserve_fp32_weights:
                group.main_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_main_weight_buffer_distributed
                    and group.data_parallel_world_size > 1,
                    dtype=torch.float32,
                    device=self.device,
                    data_parallel_group=dp_group,
                    inter_data_parallel_group=self.inter_data_parallel_group,
                    init_meta_only=True,
                    bucket_id=group_id,
                    mem_alloc_context=self.mem_alloc_context,
                )

            # Initialize the main grad buffer.
            if should_create_grad_buffer_or_main_weight_buffer:
                group.main_grad_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_grad_buffer_distributed
                    and group.data_parallel_world_size > 1,
                    dtype=torch.float32 if grad_reduce_in_fp32 else grad_dtype,
                    device=self.device,
                    data_parallel_group=dp_group,
                    inter_data_parallel_group=self.inter_data_parallel_group,
                    init_meta_only=True,
                    is_dtype_float8=not grad_reduce_in_fp32 and grad_dtype is torch.uint8,
                    temporary_bucket_allocator=self.main_grad_alloc,
                    gradient_scaling_factor=gradient_scaling_factor,
                    bucket_id=group_id,
                    mem_alloc_context=self.mem_alloc_context,
                )
                if grad_reduce_in_fp32:
                    buffer_size[torch.float32] += group.main_grad_buffer.data_size
                elif group.main_grad_buffer.is_dtype_float8:
                    buffer_size["float8"] += group.main_grad_buffer.data_size
                else:
                    buffer_size[group.main_grad_buffer.dtype] += group.main_grad_buffer.data_size

        reset_context_args = {"init_param_with_fp8": self.ddp_config.fp8_param_gather}
        module_reset_flag = {}
        if self.reset_parameters_for_meta_device_init_module:
            self.param_to_direct_module = {}
            for name, m in self.module.named_modules():
                for p in m.parameters(recurse=False):
                    self.param_to_direct_module[p] = (name, m)

            meta_params_numel = 0
            cuda_params_numel = 0
            cpu_params_numel = 0
            for group in self.parameter_groups:
                for p in group.params:
                    if p.is_meta:
                        meta_params_numel += p.numel()
                    elif p.device.type == 'cuda':
                        cuda_params_numel += p.numel()
                    else:
                        cpu_params_numel += p.numel()
            log_str = (
                f"Meta params numel: {meta_params_numel / 1_000_000:.2f} M, "
                f"CUDA params numel: {cuda_params_numel / 1_000_000:.2f} M, "
                f"CPU params numel: {cpu_params_numel / 1_000_000:.2f} M"
            )
            log_on_each_pipeline_stage(logger, logging.INFO, log_str)

        # Initialize the model weight buffer data of each parameter group.
        for group in self.parameter_groups:
            wbuf = group.model_weight_buffer
            if wbuf:
                with self.mem_alloc_context():
                    wbuf.data = torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device)
                bucket = wbuf.fetch_bucket()
            mbuf = group.main_weight_buffer
            if mbuf:
                mbuf.data = torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device)
            for item_id, p in enumerate(group.params):
                if wbuf:
                    if self.reset_parameters_for_meta_device_init_module and p.is_meta:
                        m_name, m = self.param_to_direct_module[p]
                        if not module_reset_flag.get(m_name, False) and hasattr(
                            m, "reset_parameters"
                        ):
                            old_params = list(m.parameters(recurse=False))

                            # If the GPU memory over threshold, empty cache to leave
                            # some memory for initialization of the model on the
                            # CUDA device.
                            if check_gpu_memory(threshold=0.5):
                                gc.collect()
                                torch.cuda.empty_cache()

                            m.to_empty(device=self.device, recurse=False)
                            if is_te_min_version("0.9.0") and not isinstance(
                                m, TransformerEngineBaseModule
                            ):
                                reset_context_args["with_cuda_rng_tracker"] = True
                            with ResetParametersContext(**reset_context_args):
                                m.reset_parameters()
                            module_reset_flag[m_name] = True
                            new_params = list(m.parameters(recurse=False))

                            self._reset_parameters(old_params, new_params)
                            p = group.params[item_id]

                            # After resetting parameters, delete fp8 transpose cache
                            # if we do not need keep cache.
                            if not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                                for _param in m.parameters(recurse=False):
                                    if is_float8tensor(_param):
                                        _param._transpose_invalid = True
                                        _param._transpose = None
                    assert not p.is_meta, (self.param_to_name[p], module_reset_flag)
                    wbuf.set_item(item_id, p)

                    # reset the parameter data to the buffer
                    new_param_data = wbuf.get_item_from_bucket(bucket, item_id).view(p.shape)
                    if is_float8tensor(p):
                        modify_underlying_storage(p, new_param_data)
                    else:
                        old_param_data = p.data
                        p.data = new_param_data
                        assert old_param_data._base is None
                        p.data.detach().copy_(old_param_data)
                        del old_param_data
                if mbuf:
                    if hasattr(p, 'get_high_precision_init_val'):
                        mbuf.set_item(item_id, p.get_high_precision_init_val())
                        p.clear_high_precision_init_val()
                    else:
                        mbuf.set_item(item_id, p)

                if wbuf and wbuf.is_data_distributed:
                    """
                    When MCore Custom FSDP `optim_grads_params` is enabled,
                    it is necessary to save the tensor local shard. This local shard is
                    accessible through the  `fully_shard_param_local_shard`
                    attribute of the tensor.

                    This attribute contains the local shard of the fully
                    sharded parameter, which is essential for correctly
                    saving and loading the model state when using
                    `optim_grads_params` with FSDP.

                    Example:
                        >>> # Assuming `tensor` is a fully sharded parameter
                        >>> local_shard = tensor.fully_shard_param_local_shard
                        >>> # Save the local shard as needed
                    """
                    local_shard = wbuf.get_item(item_id, only_shard=True)
                    local_shard.fsdp_shard_orig_param = p
                    p.fully_shard_param_local_shard = local_shard
                    p.fully_shard_param_local_index = wbuf.locate_item_in_global_item(item_id)
                    if self.ddp_config.num_distributed_optimizer_instances > 1:
                        p.fsdp_instance_id = torch.distributed.get_rank(
                            self.inter_data_parallel_group
                        )
                    else:
                        p.fsdp_instance_id = 0

            if wbuf and wbuf.is_data_distributed:
                wbuf.free_bucket_storage()

        # Allocate the main_weight buffer and main_grad buffer data in one buffer.
        if self.buffer_all_in_one:
            with self.mem_alloc_context():
                self.buffer = {
                    torch.float32: torch.empty(
                        buffer_size[torch.float32], dtype=torch.float32, device=self.device
                    ),
                    torch.float16: torch.empty(
                        buffer_size[torch.float16], dtype=torch.float16, device=self.device
                    ),
                    torch.bfloat16: torch.empty(
                        buffer_size[torch.bfloat16], dtype=torch.bfloat16, device=self.device
                    ),
                    "float8": torch.empty(
                        buffer_size["float8"], dtype=torch.uint8, device=self.device
                    ),
                }
            offset = {torch.float32: 0, torch.float16: 0, torch.bfloat16: 0, "float8": 0}

        def _alloc(dtype, size):
            if self.buffer_all_in_one:
                if dtype == torch.uint8:
                    dtype = "float8"
                data = self.buffer[dtype][offset[dtype] : offset[dtype] + size]
                offset[dtype] += size
                return data
            return torch.empty(size, dtype=dtype, device=self.device)

        # Initialize the main grad buffer data of each parameter group.
        for group in self.parameter_groups:
            gbuf = group.main_grad_buffer
            if not gbuf:
                continue
            with self.mem_alloc_context():
                gbuf.data = _alloc(gbuf.dtype, gbuf.data_size)
            gbuf.data.zero_()
            for item_id, p in enumerate(group.params):
                p.fsdp_managed_main_grad = gbuf.get_item(item_id)
                p._gbuf = gbuf
                p._item_id = item_id

                def main_grad_getter(p):
                    # Make sure main_grad memory storage ready.
                    bucket = p._gbuf.fetch_bucket()
                    gbuf = p._gbuf
                    item_id = p._item_id
                    return gbuf.get_item_from_bucket(bucket, item_id).view(p.shape)

                setattr(p.__class__, 'main_grad', property(main_grad_getter))

            if gbuf.is_data_distributed:
                gbuf.free_bucket_storage()

        gc.collect()
        torch.cuda.empty_cache()

    def _reset_parameters(self, old_params, new_params):
        assert len(old_params) == len(new_params)
        param_map = {}
        for old_param, new_param in zip(old_params, new_params):
            param_map[old_param] = new_param
            self.param_to_name[new_param] = self.param_to_name[old_param]
            del self.param_to_name[old_param]

            self.param_to_param_group[new_param] = self.param_to_param_group[old_param]
            del self.param_to_param_group[old_param]

            self.param_to_direct_module[new_param] = self.param_to_direct_module[old_param]
            del self.param_to_direct_module[old_param]

        for item_id, p in enumerate(self.params):
            if p in param_map:
                new_p = param_map[p]
                self.params[item_id] = new_p

        for group in self.parameter_groups:
            for item_id, p in enumerate(group.params):
                if p not in param_map:
                    continue
                new_p = param_map[p]
                group.params[item_id] = new_p
                for buf in [
                    group.model_weight_buffer,
                    group.main_weight_buffer,
                    group.main_grad_buffer,
                ]:
                    if buf is None:
                        continue
                    buf.param_idx[new_p] = buf.param_idx[p]
                    del buf.param_idx[p]

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        for group in self.parameter_groups:
            if group.main_grad_buffer is None:
                continue
            group.main_grad_buffer.data *= scaling_factor
        self.update_main_grads()

    def zero_grad(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation
        for the next iteration of training.
        """
        for _, param in self.optimizer_named_parameters:
            if param.grad is not None and param.grad._base is None:
                # For tensors that are not referenced, trying to use storage
                # resize to make memory free immediately.
                _free_storage(param.grad)
            param.grad = None

        for group in self.parameter_groups:
            if group.main_grad_buffer is None:
                continue
            group.main_grad_buffer.data.zero_()

    def _init_optimizer_named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        named_parameters = []
        for pg in self.parameter_groups:
            if pg.main_grad_buffer is None:
                continue

            optimizer_state_is_shard = pg.main_grad_buffer.is_data_distributed or (
                pg.main_weight_buffer and pg.main_weight_buffer.is_data_distributed
            )
            for item_id, orig_param in enumerate(pg.params):
                if pg.main_weight_buffer:
                    param = pg.main_weight_buffer.get_item(
                        item_id, only_shard=optimizer_state_is_shard
                    )
                elif pg.model_weight_buffer:
                    param = pg.model_weight_buffer.get_item(
                        item_id, only_shard=optimizer_state_is_shard
                    )
                else:
                    param = orig_param

                def set_param_attribute_closure(param, orig_param):
                    def set_param_attribute():
                        for attr_name in [
                            'requires_grad',
                            'sequence_parallel',
                            'shared',
                            'tensor_model_parallel',
                            'partition_dim',
                            'partition_stride',
                            'is_embedding_or_output_parameter',
                        ]:
                            if hasattr(orig_param, attr_name):
                                setattr(param, attr_name, getattr(orig_param, attr_name))

                    return set_param_attribute

                setattr(param, 'reset_attribute', set_param_attribute_closure(param, orig_param))
                setattr(param, 'orig_param', orig_param)
                param.reset_attribute()
                named_parameters.append((self.param_to_name[orig_param], param))

        return named_parameters

    def update_main_grads(self):
        """Update the main gradients for preparing the optimizer step."""
        update_shard_main_grad = self.ddp_config.data_parallel_sharding_strategy in [
            'optim',
            'optim_grads',
            'optim_grads_params',
        ]
        for _, param in self.optimizer_named_parameters:
            param.reset_attribute()
            orig_param = param.orig_param
            group = self.parameter_groups[self.param_to_param_group[orig_param]]
            item_id = group.main_grad_buffer.param_idx[orig_param]
            optimizer_grad = group.main_grad_buffer.get_item(
                item_id, only_shard=update_shard_main_grad
            )
            # The presence of main_grad_buffer but no main_weight_buffer means
            # that a precision-aware optimizer is used.
            if group.main_weight_buffer is None:
                setattr(
                    param, 'decoupled_grad', optimizer_grad if optimizer_grad.numel() > 0 else None
                )
            else:
                setattr(
                    param,
                    'grad',
                    optimizer_grad.to(param.dtype) if optimizer_grad.numel() > 0 else None,
                )

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return len(self.parameter_groups)

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """Update the model weights from the main weights."""
        for pg in self.parameter_groups:
            mbuf = pg.main_weight_buffer
            wbuf = pg.model_weight_buffer
            if mbuf is None:
                continue

            fp8_params = []
            shard_fp32_from_fp8 = []
            shard_offsets_in_fp8 = []
            shard_model_params = []

            for param in pg.params:
                item_id = mbuf.param_idx[param]
                if wbuf:
                    if wbuf.is_data_distributed or mbuf.is_data_distributed:
                        model_param = wbuf.get_item(item_id, only_shard=True)
                        main_weight = mbuf.get_item(item_id, only_shard=True)
                    else:
                        model_param = wbuf.get_item(item_id)
                        main_weight = mbuf.get_item(item_id)
                else:
                    assert not mbuf.is_data_distributed
                    model_param = param
                    main_weight = pg.main_weight_buffer.get_item(item_id)

                if is_float8tensor(param):
                    fp8_params.append(param)
                    if model_param.numel() == 0:
                        shard_fp32_from_fp8.append(None)
                        shard_offsets_in_fp8.append(None)
                        shard_model_params.append(None)
                    else:
                        shard_fp32_from_fp8.append(main_weight)
                        shard_offsets_in_fp8.append(wbuf.locate_item_in_global_item(item_id)[0])
                        shard_model_params.append(model_param)
                    continue

                if model_param.numel() > 0:
                    model_param.data.copy_(main_weight.view(model_param.shape))

            quantize_param_shard(
                fp8_params,
                shard_fp32_from_fp8,
                shard_offsets_in_fp8,
                wbuf.data_parallel_group,
                shard_model_params,
            )

    @torch.no_grad()
    def copy_model_weights_to_main_weights(self):
        """Copy the model weights to the main weights."""
        for group in self.parameter_groups:
            mbuf = group.main_weight_buffer
            if mbuf is None:
                continue
            wbuf = group.model_weight_buffer
            if mbuf.is_data_distributed:
                copyin_data = wbuf.get_shard_from_local_buffer()
            else:
                copyin_data = wbuf.data
            assert mbuf.data.numel() == copyin_data.numel(), (
                f"Master weight buffer size {mbuf.data.numel()} does not match "
                f"model weight buffer size {copyin_data.numel()}"
            )
            mbuf.data.copy_(copyin_data.data)

    def all_gather_parameters(self, async_op: bool = True):
        """All gather the parameters.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.model_weight_buffer.is_data_distributed for g in self.parameter_groups]
        ), 'all_gather_parameters() should only be called when parameters are not sharded.'

        all_gather_ops = []
        for g in self.parameter_groups:
            shard = g.model_weight_buffer.get_shard_from_local_buffer()
            all_gather_handler = torch.distributed.all_gather_into_tensor(
                output_tensor=g.model_weight_buffer.data,
                input_tensor=shard,
                group=g.model_weight_buffer.data_parallel_group,
                async_op=async_op,
            )
            if async_op:
                all_gather_ops.append(all_gather_handler)

        for op in all_gather_ops:
            op.wait()

    def reduce_scatter_gradients(self, async_op: bool = True):
        """Reduce scatter the gradients.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [not g.main_grad_buffer.is_data_distributed for g in self.parameter_groups]
        ), 'reduce_scatter_gradients() should only be called when gradients are not sharded.'

        reduce_scatter_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if gbuf is None:
                continue
            scaling_factor = gbuf.gradient_scaling_factor
            reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, self.ddp_config)
            reduce_scatter_handler = torch.distributed.reduce_scatter_tensor(
                output=gbuf.get_shard_from_local_buffer(),
                input=gbuf.data,
                op=reduce_op,
                group=g.main_grad_buffer.data_parallel_group,
                async_op=async_op,
            )

            if async_op:
                reduce_scatter_ops.append(reduce_scatter_handler)

        for op in reduce_scatter_ops:
            op.wait()

    def all_reduce_gradients(self, async_op: bool = False):
        """All reduce the gradients.
        Args:
            async_op (bool, optional): Whether to do the all-reduce
                asynchronously. Defaults to False.
        """
        assert all(
            [
                not g.main_grad_buffer.is_data_distributed
                for g in self.parameter_groups
                if g.main_grad_buffer
            ]
        ), 'all_reduce_gradients() should only be called when gradients are not sharded.'

        all_reduce_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if gbuf is None:
                continue
            scaling_factor = gbuf.gradient_scaling_factor
            reduce_op = gradient_reduce_preprocessing(gbuf.data, scaling_factor, self.ddp_config)
            all_reduce_handler = torch.distributed.all_reduce(
                gbuf.data, op=reduce_op, group=gbuf.data_parallel_group, async_op=async_op
            )
            if async_op:
                all_reduce_ops.append(all_reduce_handler)

        for op in all_reduce_ops:
            op.wait()


class BucketStatus(Enum):
    """
    An enumeration of possible statuses for a data-parallel communication bucket.

    Attributes:
        EMPTY (int): The bucket is empty and not in use.
        COMMUNICATING (int): The bucket is currently being used for communication.
        READY_TO_USE (int): The bucket is filled with data and ready for use.
    """

    EMPTY = 1
    COMMUNICATING = 2
    READY_TO_USE = 3


class GradReducePipeline:
    """
    Pipeline for reducing gradients.
    """

    def __init__(
        self,
        param_and_grad_buffer: ParamAndGradBuffer,
        rs_stream: Optional[torch.cuda.Stream] = None,
        check_nans: bool = False,
        inter_fsdp_group_grad_reduce: bool = False,
    ) -> None:
        self.buffer = param_and_grad_buffer
        self.grad_reduce_queue = []
        self.bucket_status = {
            i: BucketStatus.EMPTY
            for i in range(self.buffer.num_buckets)
            if self.buffer.parameter_groups[i].main_grad_buffer
        }
        self.bucket_grad_ready_params = [set() for _ in range(self.buffer.num_buckets)]
        self.rs_stream = rs_stream
        self.check_nans = check_nans
        self.inter_fsdp_group_grad_reduce = inter_fsdp_group_grad_reduce
        if inter_fsdp_group_grad_reduce:
            self.hsdp_all_reduce_stream = torch.cuda.Stream()

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return self.buffer.num_buckets

    def reset(self):
        """Handle the processing tasks and reset the pipeline."""
        self.wait_for_previous_grad_reduce(0)
        for bucket_id, grad_ready_params in enumerate(self.bucket_grad_ready_params):
            param_list = self.buffer.parameter_groups[bucket_id].params
            n_params = len(param_list)
            param_to_name = self.buffer.param_to_name
            assert len(grad_ready_params) == 0, (
                f"Found {len(grad_ready_params)} out of {n_params} parameters that are ready for "
                f"reduce-scatter/all-reduce, but the pipeline is being reset. "
                f"grad_ready_params: {[param_to_name[p] for p in grad_ready_params]} "
                f"param_list: {[param_to_name[p] for p in param_list]}"
            )

        for bucket_id, _ in self.bucket_status.items():
            gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
            gbuf.free_bucket_storage()
            self.bucket_status[bucket_id] = BucketStatus.EMPTY

    def reduce_gradients(
        self,
        params: List[torch.Tensor],
        suggested_queue_capacity: Optional[int] = None,
        inter_fsdp_group_grad_reduce: bool = False,
        async_grad_reduce: bool = True,
    ):
        """Reduce the gradients for the given parameters.
        Args:
            params (List[torch.Tensor]): The parameters.
            suggested_queue_capacity (int, optional): The suggested queue capacity.
                Defaults to None.
            inter_fsdp_group_grad_reduce (bool, optional): Whether to use inter-group
                gradient reduction. Defaults to False.
            async_grad_reduce (bool, optional): Whether to do the gradient-reduce
                asynchronously. Defaults to True.
        """
        for param in params:
            bucket_id = self.buffer.param_to_param_group[param]
            param_group = self.buffer.parameter_groups[bucket_id]
            if not param.requires_grad:
                assert param_group.requires_grad is False, (
                    f"Param {self.buffer.param_to_name[param]} has requires_grad=False, "
                    f"but it is in a parameter group with requires_grad=True."
                )
                continue
            assert param_group.requires_grad, (
                f"Param {self.buffer.param_to_name[param]} has requires_grad=True, "
                f"but it is in a parameter group with requires_grad=False."
            )

            # Mark grad as ready for reduce-scatter/all-reduce.
            self.bucket_grad_ready_params[bucket_id].add(param)
            if len(self.bucket_grad_ready_params[bucket_id]) == len(param_group.params):
                self.wait_for_previous_grad_reduce(
                    suggested_queue_capacity=suggested_queue_capacity
                )
                self.mark_bucket_ready(
                    bucket_id, inter_fsdp_group_grad_reduce, async_op=async_grad_reduce
                )

    def wait_for_previous_grad_reduce(
        self, suggested_queue_size: int = 1, suggested_queue_capacity: Optional[int] = None
    ):
        """
        Wait for the previous reduce-scatter/all-reduce to finish.
        Args:
            suggested_queue_size (int, optional): The recommended queue size. Defaults to 1.
            suggested_queue_capacity (Optional[int], optional): The recommended queue capacity.
                Defaults to None.
        """
        if suggested_queue_capacity is not None:
            queue_space = sum(
                [
                    self.buffer.parameter_groups[bucket_id].main_grad_buffer.bucket_index.size
                    for _, _, bucket_id in self.grad_reduce_queue
                ]
            )
            while queue_space > suggested_queue_capacity:
                grad_reduce_event, free_up_grad_bucket, bucket_id = self.grad_reduce_queue.pop(0)
                grad_reduce_event.wait()
                free_up_grad_bucket()
                queue_space -= self.buffer.parameter_groups[
                    bucket_id
                ].main_grad_buffer.bucket_index.size
        else:
            suggested_queue_size = max(0, min(suggested_queue_size, self.buffer.num_buckets - 1))
            while len(self.grad_reduce_queue) > suggested_queue_size:
                grad_reduce_event, free_up_grad_bucket, _ = self.grad_reduce_queue.pop(0)
                grad_reduce_event.wait()
                free_up_grad_bucket()

        if suggested_queue_size == 0 and self.inter_fsdp_group_grad_reduce:
            torch.cuda.current_stream().wait_stream(self.hsdp_all_reduce_stream)

    def _enforce_double_buffer_limit(self, add_buckets):
        if not self.buffer.ddp_config.fsdp_double_buffer:
            return

        param_groups = self.buffer.parameter_groups
        double_buf_units = set()
        for bucket_id in add_buckets:
            fsdp_unit_id = param_groups[bucket_id].fsdp_unit_id
            if fsdp_unit_id in self.buffer.double_buf_units:
                double_buf_units.add(fsdp_unit_id)
        assert len(double_buf_units) <= 2, (
            f"Double buffer limit exceeded. " f"Current double_buf_units: {double_buf_units}."
        )

        keep_n = len(self.grad_reduce_queue)
        for _, _, bucket_id in reversed(self.grad_reduce_queue):
            fsdp_unit_id = param_groups[bucket_id].fsdp_unit_id
            double_buf_units.add(fsdp_unit_id)
            if len(double_buf_units) > 2:
                keep_n -= 1
        self.wait_for_previous_grad_reduce(keep_n)

    def _bucket_group_gradient_reduce(
        self,
        bucket_group: List[int],
        async_op: bool = False,
        inter_fsdp_group_grad_reduce: bool = False,
    ):
        """Mark the bucket ready for reduce-scatter/all-reduce, if all bucket in
        the bucket group are ready, then do the reduce-scatter/all-reduce.
        Args:
            bucket_id (int): The bucket to be marked.
            async_rs (bool, optional): Whether to do the reduce-scatter/all-reduce
                asynchronously. Defaults to False.
        Returns:
            bool: True if the bucket is go for reduce-scatter/all-reduce.
        """
        # When using FSDP double buffer, waiting for the necessary bucket to be
        # released ensures that our double buffer will not explode due to too
        # many empty bucket requests.
        if self.buffer.ddp_config.fsdp_double_buffer:
            self._enforce_double_buffer_limit(bucket_group)

        current_stream = torch.cuda.current_stream()
        reduce_scatter_stream = (
            self.rs_stream if self.rs_stream is not None else torch.cuda.current_stream()
        )
        reduce_scatter_stream.wait_stream(current_stream)

        dp_group = self.buffer.parameter_groups[
            bucket_group[0]
        ].main_grad_buffer.data_parallel_group
        with torch.cuda.stream(reduce_scatter_stream):
            with _coalescing_manager(dp_group, async_ops=async_op) as coalescing_event:
                grad_shards = {}
                for bucket_id in bucket_group:
                    gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                    bucket = gbuf.fetch_bucket()
                    scaling_factor = gbuf.gradient_scaling_factor
                    reduce_op = gradient_reduce_preprocessing(
                        gbuf.data, scaling_factor, gbuf.ddp_config
                    )
                    if gbuf.ddp_config.data_parallel_sharding_strategy == 'no_shard':
                        torch.distributed.all_reduce(
                            bucket.data, op=reduce_op, group=gbuf.data_parallel_group
                        )
                    else:
                        grad_shard = gbuf.get_shard_from_bucket(bucket)
                        # pylint: disable=C0301
                        # The `grad_shard`` is part of `bucket.data`` and the following
                        # new empty is important for memory safety, when using
                        # TORCH_NCCL_AVOID_RECORD_STREAMS=1.
                        # For reference: https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486
                        if not self.buffer.ddp_config.nccl_ub:
                            grad_shard = torch.empty_like(grad_shard)
                        torch.distributed.reduce_scatter_tensor(
                            output=grad_shard,
                            input=bucket.data,
                            op=reduce_op,
                            group=gbuf.data_parallel_group,
                        )
                        grad_shards[bucket_id] = grad_shard
                    self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING
            coalescing_event.wait()
            for bucket_id in bucket_group:
                # Local gradient accumulate
                gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                if gbuf.ddp_config.data_parallel_sharding_strategy != 'no_shard':
                    # Gradient accumulate on local buffer
                    local_buffer = gbuf.get_shard_from_local_buffer()
                    local_buffer += grad_shards[bucket_id]
            reduce_scatter_view_out_event = reduce_scatter_stream.record_event()

        # Gradient reduction within the model replication domain
        if inter_fsdp_group_grad_reduce:
            ddp_config = self.buffer.ddp_config
            assert ddp_config.data_parallel_sharding_strategy != 'no_shard'
            self.hsdp_all_reduce_stream.wait_stream(reduce_scatter_stream)
            inter_data_parallel_group = self.buffer.parameter_groups[
                bucket_group[0]
            ].main_grad_buffer.inter_data_parallel_group
            with torch.cuda.stream(self.hsdp_all_reduce_stream):
                with _coalescing_manager(inter_data_parallel_group):
                    for bucket_id in bucket_group:
                        gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                        grad_local_buffer = gbuf.get_shard_from_local_buffer()
                        if ddp_config.average_in_collective:
                            reduce_op = torch.distributed.ReduceOp.AVG
                        else:
                            reduce_op = torch.distributed.ReduceOp.SUM
                        torch.distributed.all_reduce(
                            grad_local_buffer, group=gbuf.inter_data_parallel_group, op=reduce_op
                        )

        free_up_grad_bucket_func = {}
        for bucket_id in bucket_group:

            def get_closure(bucket_id):
                def free_up_grad_bucket():
                    self.bucket_grad_ready_params[bucket_id] = set()
                    gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                    if gbuf.is_data_distributed:
                        gbuf.free_bucket_storage()
                    self.bucket_status[bucket_id] = BucketStatus.EMPTY

                return free_up_grad_bucket

            free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

        if async_op:
            for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
                self.grad_reduce_queue.append(
                    (reduce_scatter_view_out_event, free_up_grad_bucket, bucket_id)
                )
            return

        reduce_scatter_view_out_event.wait()
        for free_up_grad_bucket in free_up_grad_bucket_func.values():
            free_up_grad_bucket()

    def mark_bucket_ready(
        self, bucket_id: int, inter_fsdp_group_grad_reduce: bool = False, async_op: bool = True
    ) -> bool:
        """Mark the bucket ready for gradient reduce, if all bucket in the bucket group
        are ready, reduce-scatter or all-reduce gradient bucket, in the case of HSDP,
        there is an additional all-reduce in the model replication domain.
        Args:
            bucket_id (int): The bucket to be marked ready to reduce-scatter or
                all-reduce.
            inter_fsdp_group_grad_reduce (bool, optional): Whether to use inter-group
                gradient reduction. Defaults to False.
            async_op (bool, optional): Whether to do the gradient-reduce
                asynchronously. Defaults to True.
        Returns:
            bool: True if the bucket is go for reduce-scatter/all-reduce.
        """
        # Prepare bucket group for gradient reduce. Note that the
        # some bucket parameters do not require grad, so we need to
        # remove them from the bucket group.
        bucket_group = self.buffer.bucket_to_bucket_group[bucket_id]
        bucket_group = [i for i in bucket_group if self.buffer.parameter_groups[i].main_grad_buffer]
        # If any bucket in the bucket group is not ready, skip the gradient reduce
        # waiting for the bucket group to be all ready before executing.
        for bucket_id in bucket_group:
            param_group = self.buffer.parameter_groups[bucket_id]
            if len(self.bucket_grad_ready_params[bucket_id]) != len(param_group.params):
                return False

        self._bucket_group_gradient_reduce(
            bucket_group,
            async_op=async_op,
            inter_fsdp_group_grad_reduce=inter_fsdp_group_grad_reduce,
        )
        return True


class PrefetchOrder(Enum):
    """
    An enumeration of possible prefetch orders for data-parallel operations.

    Attributes:
        FORWARD_PASS_ORDER (int): Prefetch in the order of forward pass computation.
        BACKWARD_PASS_ORDER (int): Prefetch in the order of backward pass computation.
    """

    FORWARD_PASS_ORDER = 0
    BACKWARD_PASS_ORDER = 1


class AllGatherPipeline:
    """
    Pipeline for all-gathering parameters.
    """

    def __init__(self, param_and_grad_buffer: ParamAndGradBuffer) -> None:
        self.buffer = param_and_grad_buffer
        self.param_gather_event_map = {}
        self.bucket_status = {i: BucketStatus.EMPTY for i in range(self.buffer.num_buckets)}
        self.bucket_can_be_released = {i: False for i in range(self.buffer.num_buckets)}

        self.bucket_to_bucket_group = {}
        group_id = 0
        for bucket_group in self.buffer.bucket_to_bucket_group.values():
            new_group = False
            for bucket_id in bucket_group:
                if bucket_id not in self.bucket_to_bucket_group:
                    new_group = True
                    break
            if new_group:
                group_id += 1
                for bucket_id in bucket_group:
                    self.bucket_to_bucket_group[bucket_id] = group_id

    @property
    def num_buckets(self):
        """Return the number of buckets."""
        return self.buffer.num_buckets

    def reset(self):
        """Reset the pipeline state."""
        if len(self.param_gather_event_map) > 0:
            warnings.warn(
                "There are still pending all-gather tasks, process them. "
                f"Bucket status: {self.bucket_status}.",
                UserWarning,
            )
            while len(self.param_gather_event_map) > 0:
                bucket_id = next(iter(self.param_gather_event_map))
                self.wait_bucket_ready(bucket_id)
        for bucket_id in self.bucket_can_be_released:
            self.bucket_can_be_released[bucket_id] = True
        self.recycle_unused_buckets()

        assert all([status is BucketStatus.EMPTY for status in self.bucket_status.values()]), (
            f"There are still working buckets, it is not safe to reset. "
            f"bucket_status: {self.bucket_status}."
        )
        assert all(
            [not can_be_released for can_be_released in self.bucket_can_be_released.values()]
        ), (
            f"The bucket can be released table is in an abnormal state, not safe to reset. "
            f"bucket_can_be_released: {self.bucket_can_be_released}."
        )

    def all_gather_params(
        self,
        params: List[torch.Tensor],
        prefetch: bool = False,
        prefetch_order: PrefetchOrder = PrefetchOrder.FORWARD_PASS_ORDER,
        suggested_AG_prefetch_size: Optional[int] = None,
        async_param_gather: bool = True,
    ):
        """All-gather the params. If prefetch is enabled, prefetch next buckets
        in the order of `prefetch_order`.

        Args:
            params (List[torch.Tensor]): The list of params to be all-gathered.
            prefetch (bool, optional): Whether to prefetch the next bucket. Defaults to False.
            prefetch_order (PrefetchOrder, optional): The order of prefetching.
                Defaults to PrefetchOrder.FORWARD_PASS_ORDER.
            suggested_AG_prefetch_size (Optional[int], optional):
                The suggested prefetch size for all-gathering. Defaults to None.
        """
        if len(params) == 0:
            return

        ag_buckets = [self.buffer.param_to_param_group[item] for item in params]
        ag_buckets = list(sorted(set(ag_buckets)))
        parameter_groups = self.buffer.parameter_groups
        if self.buffer.ddp_config.fsdp_double_buffer:
            double_buf_units = set()
            for bucket_id in ag_buckets:
                fsdp_unit_id = parameter_groups[bucket_id].fsdp_unit_id
                if fsdp_unit_id in self.buffer.double_buf_units:
                    double_buf_units.add(fsdp_unit_id)
            if len(double_buf_units) > 2:
                raise ValueError(
                    f"{double_buf_units} FSDP units were requested, "
                    "but double buffers can support no more than 2 FSDP units."
                )

        # If prefetch is enabled, we will add prefetch buckets to ag_buckets.
        if prefetch:

            def next_bucket_id(ag_buckets):
                if prefetch_order == PrefetchOrder.FORWARD_PASS_ORDER:
                    bucket_id = ag_buckets[0] + 1
                    for i in ag_buckets[1:]:
                        if i != bucket_id:
                            break
                        bucket_id += 1
                else:
                    bucket_id = ag_buckets[-1] - 1
                    for i in reversed(ag_buckets[:-1]):
                        if i != bucket_id:
                            break
                        bucket_id -= 1
                if bucket_id < 0 or bucket_id >= self.buffer.num_buckets:
                    return None
                return bucket_id

            def need_skip_prefetch(bucket_id):
                # If use double buffer, we need to check if the next bucket
                # is exceeding the coverage of the double buffer.
                if self.buffer.ddp_config.fsdp_double_buffer:
                    fsdp_unit_id = parameter_groups[bucket_id].fsdp_unit_id
                    double_buf_units.add(fsdp_unit_id)
                    if len(double_buf_units) > 2:
                        # Prefetching the next bucket will exceed the coverage of
                        # the double buffer, so we need to stop prefetching.
                        return True
                return False

            if suggested_AG_prefetch_size is not None:
                bucket_id = next_bucket_id(ag_buckets)
                while bucket_id is not None:
                    all_gather_size = sum(
                        [
                            parameter_groups[i].model_weight_buffer.bucket_index.size
                            for i in ag_buckets
                        ]
                    )
                    if all_gather_size >= suggested_AG_prefetch_size:
                        break

                    if need_skip_prefetch(bucket_id):
                        break

                    ag_buckets.extend(self.buffer.bucket_to_bucket_group[bucket_id])
                    ag_buckets = list(sorted(set(ag_buckets)))
                    bucket_id = next_bucket_id(ag_buckets)
            else:
                bucket_id = next_bucket_id(ag_buckets)

                if need_skip_prefetch(bucket_id):
                    bucket_id = None

                if bucket_id is not None:
                    ag_buckets.extend(self.buffer.bucket_to_bucket_group[bucket_id])
                    ag_buckets = list(sorted(set(ag_buckets)))

        ag_buckets = [i for i in ag_buckets if self.bucket_status[i] == BucketStatus.EMPTY]
        if len(ag_buckets) == 0:
            return

        # Divide buckets into aggregate groups
        bucket_group_to_buckets = {}
        for bucket_id in ag_buckets:
            group_id = self.bucket_to_bucket_group[bucket_id]
            if group_id not in bucket_group_to_buckets:
                bucket_group_to_buckets[group_id] = []
            bucket_group_to_buckets[group_id].append(bucket_id)

        # Coalesce all-gather operations for all buckets in the same data-parallel-group
        for _, buckets in bucket_group_to_buckets.items():
            param_group = parameter_groups[buckets[0]]
            dp_group = param_group.model_weight_buffer.data_parallel_group
            with _coalescing_manager(dp_group, async_ops=async_param_gather) as coalescing_event:
                for bucket_id in buckets:
                    self.async_bucket_gather(bucket_id)

                # reset param gather event with coalescing event
                for bucket_id in buckets:
                    _, mark_bucket_ready_to_use = self.param_gather_event_map[bucket_id]
                    self.param_gather_event_map[bucket_id] = (
                        coalescing_event,
                        mark_bucket_ready_to_use,
                    )

            # Wait for all-gather to finish
            if not async_param_gather:
                for bucket_id in buckets:
                    self.wait_bucket_ready(bucket_id)

    def wait_bucket_ready(self, bucket_id, empty_ok=False):
        """Wait for the bucket to be ready."""
        if self.bucket_status[bucket_id] == BucketStatus.READY_TO_USE:
            return
        if self.bucket_status[bucket_id] == BucketStatus.EMPTY:
            if empty_ok:
                return
            raise ValueError(f"Bucket {bucket_id} is empty.")

        param_gather_event, mark_bucket_ready_to_use = self.param_gather_event_map.pop(bucket_id)
        param_gather_event.wait()
        mark_bucket_ready_to_use()

    @torch.no_grad()
    def release_bucket(self, bucket_id: int):
        """Release the bucket."""
        if self.bucket_status[bucket_id] == BucketStatus.EMPTY:
            return

        if self.bucket_status[bucket_id] == BucketStatus.COMMUNICATING:
            raise ValueError(f"Bucket {bucket_id} is communicating.")

        wbuf = self.buffer.parameter_groups[bucket_id].model_weight_buffer
        wbuf.free_bucket_storage()
        self.bucket_status[bucket_id] = BucketStatus.EMPTY

    def recycle_unused_buckets(self):
        """Recycle the unused buckets."""
        for bucket_id, can_be_released in self.bucket_can_be_released.items():
            if can_be_released:
                self.release_bucket(bucket_id)
                self.bucket_can_be_released[bucket_id] = False

    @torch.no_grad()
    def async_bucket_gather(self, bucket_id: int) -> None:
        """All-gather the bucket and set the items."""
        self.bucket_can_be_released[bucket_id] = False
        if self.bucket_status[bucket_id] != BucketStatus.EMPTY:
            return

        self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING
        wbuf = self.buffer.parameter_groups[bucket_id].model_weight_buffer

        # Lazy release the unused buckets.
        self.recycle_unused_buckets()
        bucket = wbuf.fetch_bucket(and_allocate_params_data=True)
        param_gather_event = torch.distributed.all_gather_into_tensor(
            output_tensor=bucket.data,
            input_tensor=wbuf.get_shard_from_local_buffer(),
            group=wbuf.data_parallel_group,
            async_op=True,
        )

        def get_closure(bucket_id):
            @torch.no_grad()
            def mark_bucket_ready_to_use():
                self.bucket_status[bucket_id] = BucketStatus.READY_TO_USE

            return mark_bucket_ready_to_use

        mark_bucket_ready_to_use = get_closure(bucket_id)
        self.param_gather_event_map[bucket_id] = (param_gather_event, mark_bucket_ready_to_use)


@torch.no_grad()
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
    """
    Gradient reduce preprocessing for gradient averaging and gradient scaling.
    """

    if scaling_factor is None:
        reduce_op = torch.distributed.ReduceOp.SUM
    elif ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG
    elif ddp_config.gradient_reduce_div_fusion and grad_data.dtype != torch.bfloat16:
        reduce_op = torch.distributed._make_nccl_premul_sum(scaling_factor)
    else:
        grad_data.mul_(scaling_factor)
        reduce_op = torch.distributed.ReduceOp.SUM

    return reduce_op


def check_gpu_memory(threshold=0.9):
    """
    Check if the GPU memory is over the threshold.
    Args:
        threshold (float, optional): The threshold to check if the GPU memory is over.
            Defaults to 0.9.
    Returns:
        bool: True if the GPU memory is over the threshold.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    allocated_ratio = allocated / total
    reserved_ratio = reserved / total

    near_full = allocated_ratio >= threshold or reserved_ratio >= threshold

    if near_full:
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"GPU Memory: Allocated: {allocated_ratio:.2%}, Reserved: {reserved_ratio:.2%}",
        )
    return near_full


class ResetParametersContext:
    """
    Context manager for resetting parameters for meta device initialization module.
    """

    def __init__(self, init_param_with_fp8=False, with_cuda_rng_tracker=False):
        self.init_param_with_fp8 = init_param_with_fp8
        self.with_cuda_rng_tracker = with_cuda_rng_tracker

    def __enter__(self):
        self.stack = ExitStack()
        if self.init_param_with_fp8:
            args = {"enabled": True}
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                args["preserve_high_precision_init_val"] = True
            self.stack.enter_context(fp8_model_init(**args))

        if self.with_cuda_rng_tracker:
            self.stack.enter_context(get_cuda_rng_tracker().fork())

        return self

    def __exit__(self, *exc_details):
        self.stack.__exit__(*exc_details)


def override_sharded_param_methods_with_safety_checks(params, all_gather_pipeline):
    """
    Override the methods of the parameters to prevent undefined behavior.
    Args:
        params (List[torch.Tensor]): The parameters to add hint on shard to functions.
        all_gather_pipeline (AllGatherPipeline): The all-gather pipeline.
    """
    for p in params:
        to_function = p.to
        cpu_function = p.cpu

        def override_sharded_param_to_function_closure(p, to_function):
            def override_sharded_param_to_function(*args, **kwargs):
                bucket_id = all_gather_pipeline.buffer.param_to_param_group[p]
                status = all_gather_pipeline.bucket_status[bucket_id]
                if status == BucketStatus.READY_TO_USE:
                    return to_function(*args, **kwargs)
                raise RuntimeError(
                    "This parameter is already shard by MCore FSDP and the "
                    "shared-state parameter does not support 'to' function."
                    "please define the dtype and device of the parameter before FSDP wrap."
                )

            return override_sharded_param_to_function

        setattr(p, 'to', override_sharded_param_to_function_closure(p, to_function))

        def override_sharded_param_cpu_function_closure(p, cpu_function):
            def override_sharded_param_cpu_function(*args, **kwargs):
                bucket_id = all_gather_pipeline.buffer.param_to_param_group[p]
                status = all_gather_pipeline.bucket_status[bucket_id]
                if status == BucketStatus.READY_TO_USE:
                    return cpu_function(*args, **kwargs)
                warnings.warn(
                    "The parameters are sharded by MCore FSDP, and no actual "
                    "cpu operation is performed."
                )
                return torch.empty([], device='cpu')

            return override_sharded_param_cpu_function

        setattr(p, 'cpu', override_sharded_param_cpu_function_closure(p, cpu_function))
