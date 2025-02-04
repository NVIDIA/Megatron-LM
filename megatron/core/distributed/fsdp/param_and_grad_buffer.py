# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import logging
import math
import traceback
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch

from megatron.core import parallel_state
from megatron.core.utils import is_float8tensor
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

try:
    # This will be used when "--use-fp8-params" is enabled.
    # When BF16/FP16 parameters don't exist, we need to cast the FP32 main parameters to
    # FP8 directly in the optimizer.
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
except:
    pass

logger = logging.getLogger(__name__)


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
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


@dataclasses.dataclass
class BucketingPolicy:
    guide_bucket_size: Optional[int] = 40_000_000
    fsdp_modules: List[torch.nn.Module] = dataclasses.field(default_factory=list)
    data_parallel_sharding_strategy: str = 'NO_OP'


def _pad(number_to_be_padded: int, divisor: int) -> int:
    return int(math.ceil(number_to_be_padded / divisor) * divisor)


def build_data_parallel_buffer_index(
    elements: List[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    is_data_distributed: bool,
    ddp_config: DistributedDataParallelConfig,
    bucket_id: int = 0,
) -> Tuple[int, List[tuple], List[tuple], List[tuple]]:
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
        Tuple[int, List[tuple], List[tuple], List[tuple]]: The index range of every tensor,
            every bucket and every in bucket local buffer.
    """

    def _pad_if_needed(data_index: int) -> int:
        """
        Pads data indices if using distributed optimizer (to ensure uniform sharding).
        """
        if ddp_config.data_parallel_sharding_strategy != 'NO_OP':
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
    bucket_id = 0
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
    data: torch.Tensor


class TemporaryBucketAllocator:
    """Temporary bucket allocator. This class is used to allocate temporary buckets,
    the default implementation is like PyTorch's temporary buffer allocator."""

    def __init__(self):
        self.buckets = {}

    def get_bucket(
        self, bucket_id: int, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        bucket = self.buckets[bucket_id]
        bucket.data = bucket.data.to(dtype=dtype)
        _alloc_storage(bucket.data, torch.Size([size]))

        return bucket

    def free_the_bucket_storage(self, bucket_id: int):
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)


class RotaryBucketAllocator(TemporaryBucketAllocator):
    def __init__(self, name: str, capacity: int = 2):
        self.name = name
        self.capacity = capacity
        self.gbuf_reference = {self._get_gbuf_name(i): set() for i in range(self.capacity)}

    def get_bucket(self, bucket_id: int, size: int, dtype: torch.dtype, device: torch.device):
        gbuf_name = self._get_gbuf_name(bucket_id)
        assert self.gbuf_reference[gbuf_name] in [
            set(),
            {bucket_id},
        ], f"gbuf {gbuf_name} is already in use, {self.gbuf_reference[gbuf_name]}"
        self.gbuf_reference[gbuf_name].add(bucket_id)
        return Bucket(
            data=parallel_state.get_global_memory_buffer().get_tensor(
                [size], dtype=dtype, name=gbuf_name
            )
        )

    def _get_gbuf_name(self, bucket_id: int):
        return f"{self.name}_bucket_{bucket_id % self.capacity}"

    def free_the_bucket_storage(self, bucket_id: int):
        gbuf_name = self._get_gbuf_name(bucket_id)
        self.gbuf_reference[gbuf_name].discard(bucket_id)


class DataParallelBuffer:
    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        is_data_distributed: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = torch.get_default_device(),
        data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        temporary_bucket_allocator: Optional[TemporaryBucketAllocator] = None,
        init_meta_only: bool = False,
        is_dtype_float8: bool = False,
        gradient_scaling_factor: Optional[float] = None,
    ) -> None:
        self.ddp_config = ddp_config
        self.params = params
        _param_dtype = {p.dtype for p in self.params}
        assert len(_param_dtype) == 1, f'params have different dtypes: {_param_dtype}'
        self.is_data_distributed = is_data_distributed
        self.dtype = dtype if dtype else next(iter(_param_dtype))
        self.device = device
        self.data_parallel_group = data_parallel_group
        self.dp_rank = torch.distributed.get_rank(group=self.data_parallel_group)
        self.dp_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        self.temporary_bucket_allocator = (
            temporary_bucket_allocator if temporary_bucket_allocator else TemporaryBucketAllocator()
        )
        self.is_dtype_float8 = is_dtype_float8
        self.gradient_scaling_factor = gradient_scaling_factor

        (
            self.item_index_map,
            self.bucket_index,
            self.shard_bucket_index,
        ) = build_data_parallel_buffer_index(
            [p.data.shape for p in self.params],
            self.dp_rank,
            self.dp_world_size,
            is_data_distributed,
            ddp_config,
        )

        self.data_size = (
            self.bucket_index.size if not is_data_distributed else self.shard_bucket_index.size
        )
        if init_meta_only:
            self.data = None
        else:
            self.data = torch.empty(self.data_size, dtype=self.dtype, device=device,)

        self.param_idx = {p: i for i, p in enumerate(self.params)}

    def fetch_the_bucket(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        bucket_index = self.bucket_index

        if not self.is_data_distributed and dtype == self.dtype:
            return Bucket(
                data=self.data[
                    bucket_index.global_data_index : bucket_index.global_data_index
                    + bucket_index.size
                ],
            )

        bucket = self.temporary_bucket_allocator.get_bucket(
            bucket_id=bucket_index.bucket_id,
            size=bucket_index.size,
            dtype=dtype,
            device=self.device,
        )
        return bucket

    def free_the_bucket_storage(self):
        if not self.is_data_distributed:
            return

        self.temporary_bucket_allocator.free_the_bucket_storage(self.bucket_index.bucket_id)

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
        return (slice_start - item_index.global_data_index, slice_end - item_index.global_data_index)

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

    def set_item(self, item_id: int, item_data: torch.Tensor) -> None:
        if self.is_data_distributed:
            slice_start, slice_end = self._get_item_slice_in_shard(item_id)
            item_data = item_data.flatten()[slice_start:slice_end]
        local_index_start, local_index_end = self._get_item_local_index(item_id)
        shard = self.data[local_index_start:local_index_end]
        if shard.numel() > 0:
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, only_shard: bool = False) -> torch.Tensor:
        if only_shard:
            start, end = self._get_item_local_shard_index(item_id)
        else:
            start, end = self._get_item_local_index(item_id)

        return self.data[start:end]

    def get_item_from_bucket(self, bucket: Bucket, item_id: int):
        item_index = self.item_index_map[item_id]
        bucket_index = self.bucket_index
        start_index = item_index.global_data_index - bucket_index.global_data_index
        end_index = start_index + item_index.size
        item = bucket.data[start_index:end_index]
        return item

    def get_shard_from_bucket(self, bucket: Bucket):
        shard_bucket_index = self.shard_bucket_index
        offset = shard_bucket_index.bucket_data_index
        shard_size = shard_bucket_index.size
        shard = bucket.data[offset : offset + shard_size]
        return shard

    def get_shard_from_local_buffer(self) -> torch.Tensor:
        """Get the local sharding of the bucket."""
        index = self.shard_bucket_index
        return self.data[index.local_data_index : index.local_data_index + index.size]

    def free_data_storage(self):
        # NOTE: Be careful when calling this function, it will free the storage of the buffer.
        # Make sure that the buffer is not used anymore.
        _free_storage(self.data)

    def realloc_data_storage(self):
        _alloc_storage(self.data, torch.Size([self.data_size]))


@dataclasses.dataclass
class ParameterGroup:
    params: List[torch.nn.Parameter]
    dtype: Optional[torch.dtype] = None
    is_expert_parameter: bool = False
    data_parallel_world_size: Optional[int] = None
    model_weight_buffer: Optional[DataParallelBuffer] = None
    # master_weights: List[torch.Tensor] = dataclasses.field(default_factory=list)
    master_weight_buffer: Optional[DataParallelBuffer] = None
    main_grad_buffer: Optional[DataParallelBuffer] = None


def _get_parameter_groups(
    module: torch.nn.Module,
    parameters: List[torch.nn.Parameter],
    policy: BucketingPolicy,
    ddp_config: DistributedDataParallelConfig,
):
    """
    Get the parameter group for the given module and parameters.
    """
    align_with_distopt_bucketing_method = False
    if (
        align_with_distopt_bucketing_method
        and ddp_config.data_parallel_sharding_strategy != "MODEL_AND_OPTIMIZER_STATES"
    ):
        parameters = list(reversed(parameters))

    parameter_groups = []
    fsdp_units = []
    if policy.fsdp_modules:
        param_to_id = {}
        for i, p in enumerate(parameters):
            param_to_id[p] = i
        for m in module.modules():
            if isinstance(m, tuple(policy.fsdp_modules)):
                fsdp_units.append(list(m.parameters()))
        fsdp_param_id_list = [param_to_id[p] for fsdp_unit in fsdp_units for p in fsdp_unit]
        fsdp_unit_0 = [p for p in parameters if param_to_id[p] not in fsdp_param_id_list]
        fsdp_units = [fsdp_unit_0] + fsdp_units
        assert len(parameters) == sum(len(fsdp_unit) for fsdp_unit in fsdp_units), (
            f'Number of parameters in the module ({len(parameters)}) does not match '
            f'the number of parameters in the FSDP units ({sum(len(fsdp_unit) for fsdp_unit in fsdp_units)}), '
            f'which is unexpected, please check the FSDP module list {policy.fsdp_modules}.'
        )

        parameter_groups = [ParameterGroup(fsdp_unit) for fsdp_unit in fsdp_units]
    else:
        parameter_groups = [ParameterGroup(parameters)]

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
            and policy.data_parallel_sharding_strategy != "NO_OP"
        )

    is_expert_parameter = lambda p: not getattr(p, 'allreduce', True)
    bucket_parameter_groups = []
    for i, group in enumerate(parameter_groups):
        seperate_group = {}
        for p in group.params:
            if is_float8tensor(p):
                dtype = "float8"
            else:
                dtype = p.dtype
            is_expert = is_expert_parameter(p)
            if (dtype, is_expert) not in seperate_group:
                seperate_group[(dtype, is_expert)] = ParameterGroup(
                    [], dtype=dtype, is_expert_parameter=is_expert
                )
            seperate_group[(dtype, is_expert)].params.append(p)

        # Bucket the parameters based on guide bucket size.
        guide_bucket_size = policy.guide_bucket_size
        bucket_groups = []
        for g in seperate_group.values():
            bucket_size = 0
            bucket = []
            for p in g.params:
                if _does_param_require_new_bucket(p):
                    if len(bucket) > 0:
                        bucket_groups.append(
                            ParameterGroup(
                                bucket, dtype=g.dtype, is_expert_parameter=g.is_expert_parameter
                            )
                        )
                    bucket_groups.append(
                        ParameterGroup(
                            [p], dtype=g.dtype, is_expert_parameter=g.is_expert_parameter
                        )
                    )
                    bucket = []
                    bucket_size = 0
                    continue

                bucket.append(p)
                bucket_size += p.data.nelement()
                if guide_bucket_size and bucket_size >= guide_bucket_size:
                    bucket_groups.append(
                        ParameterGroup(
                            bucket, dtype=g.dtype, is_expert_parameter=g.is_expert_parameter
                        )
                    )
                    bucket = []
                    bucket_size = 0
                    continue

            if bucket:
                bucket_groups.append(
                    ParameterGroup(bucket, dtype=g.dtype, is_expert_parameter=g.is_expert_parameter)
                )
        bucket_parameter_groups.extend(bucket_groups)

    if (
        align_with_distopt_bucketing_method
        and ddp_config.data_parallel_sharding_strategy != "MODEL_AND_OPTIMIZER_STATES"
    ):
        bucket_parameter_groups = list(reversed(bucket_parameter_groups))

    param_to_param_group = {}
    for i, group in enumerate(bucket_parameter_groups):
        for p in group.params:
            param_to_param_group[p] = i

    bucket_to_fsdp_unit = {}
    if fsdp_units:
        for i, fsdp_unit in enumerate(fsdp_units):
            for p in fsdp_unit:
                bucket_to_fsdp_unit[param_to_param_group[p]] = i

    return bucket_parameter_groups, fsdp_units, param_to_param_group, bucket_to_fsdp_unit


class ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        params: List[torch.nn.Parameter],
        bucketing_policy: BucketingPolicy,
        data_parallel_group: torch.distributed.ProcessGroup,
        param_to_name: Dict[torch.nn.Parameter, str],
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        preserve_fp32_weights: bool = True,
        grad_reduce_in_fp32: bool = True,
        gradient_scaling_factor: Optional[float] = None,
        expert_gradient_scaling_factor: Optional[float] = None,
        device: torch.device = torch.device('cuda'),
    ):
        self.ddp_config = ddp_config

        # Check that params are unique.
        params = list(params)
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        self.bucketing_policy = bucketing_policy
        self.param_to_name = param_to_name
        self.preserve_fp32_weights = preserve_fp32_weights
        self.grad_reduce_in_fp32 = grad_reduce_in_fp32
        self.data_parallel_group = data_parallel_group
        self.expert_data_parallel_group = expert_data_parallel_group
        self.params = params
        self.gradient_scaling_factor = gradient_scaling_factor
        self.expert_gradient_scaling_factor = expert_gradient_scaling_factor
        self.device = device

        # Get the parameter groups.
        (
            self.parameter_groups,
            self.fsdp_units,
            self.param_to_param_group,
            self.bucket_to_fsdp_unit,
        ) = _get_parameter_groups(module, params, bucketing_policy, ddp_config)
        self._init_each_parameter_group_buffers()

        # Initialize the optimizer named parameters.
        self.optimizer_named_parameters = self._init_optimizer_named_parameters()

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            logger.setLevel(logging.INFO)
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.parameter_groups)}'
            )
            for index, bucket in enumerate(self.parameter_groups):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index+1} ({numel} elements, dtype {bucket.dtype}):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}')

    def _init_each_parameter_group_buffers(self):
        """
        Initialize the buffers for each parameter group.
        """
        data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        if data_parallel_sharding_strategy == 'NO_OP':
            is_model_weight_buffer_distributed = False
            is_master_weight_buffer_distributed = False
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == 'OPTIMIZER_STATES':
            is_model_weight_buffer_distributed = False
            is_master_weight_buffer_distributed = True
            is_grad_buffer_distributed = False
        elif data_parallel_sharding_strategy == 'OPTIMIZER_STATES_AND_GRADS':
            is_model_weight_buffer_distributed = False
            is_master_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        elif data_parallel_sharding_strategy == 'MODEL_AND_OPTIMIZER_STATES':
            is_model_weight_buffer_distributed = True
            is_master_weight_buffer_distributed = True
            is_grad_buffer_distributed = True
        else:
            raise ValueError(
                f'Invalid data_parallel_sharding_strategy: {data_parallel_sharding_strategy}'
            )

        self.temp_mem_alloc_for_main_grad_buffer = RotaryBucketAllocator(
            name='temp_mem_alloc_for_main_grad_buffer'
        )
        self.temp_mem_alloc_for_main_grad_buffer = None
        self.buffer_all_in_one = False

        preserve_fp32_weights = self.preserve_fp32_weights
        grad_reduce_in_fp32 = self.grad_reduce_in_fp32
        buffer_size = {
            torch.float32: 0,
            torch.float16: 0,
            torch.bfloat16: 0,
            "float8": 0,
        }
        for group_id, group in enumerate(self.parameter_groups):
            dp_group = (
                self.data_parallel_group
                if not group.is_expert_parameter
                else self.expert_data_parallel_group
            )
            group.data_parallel_world_size = torch.distributed.get_world_size(group=dp_group)
            gradient_scaling_factor = (
                self.gradient_scaling_factor
                if not group.is_expert_parameter
                else self.expert_gradient_scaling_factor
            )
            is_dtype_float8 = is_float8tensor(group.params[0])
            if is_dtype_float8:
                param_dtype = torch.uint8
            else:
                param_dtype = group.params[0].dtype

            # Initialize the model weight buffer.
            if data_parallel_sharding_strategy != 'NO_OP':
                group.model_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_model_weight_buffer_distributed
                    and group.data_parallel_world_size > 1,
                    dtype=param_dtype,
                    device=self.device,
                    data_parallel_group=dp_group,
                    init_meta_only=True,
                    is_dtype_float8=is_dtype_float8,
                )
                if is_dtype_float8:
                    buffer_size["float8"] += group.model_weight_buffer.data_size
                else:
                    buffer_size[param_dtype] += group.model_weight_buffer.data_size

            # Initialize the master weight buffer.
            if preserve_fp32_weights:
                group.master_weight_buffer = DataParallelBuffer(
                    self.ddp_config,
                    group.params,
                    is_data_distributed=is_master_weight_buffer_distributed
                    and group.data_parallel_world_size > 1,
                    dtype=torch.float32,
                    device=self.device,
                    data_parallel_group=dp_group,
                    init_meta_only=True,
                )
                buffer_size[torch.float32] += group.master_weight_buffer.data_size

            # Initialize the main grad buffer.
            group.main_grad_buffer = DataParallelBuffer(
                self.ddp_config,
                group.params,
                is_data_distributed=is_grad_buffer_distributed
                and group.data_parallel_world_size > 1,
                dtype=torch.float32 if grad_reduce_in_fp32 else param_dtype,
                device=self.device,
                data_parallel_group=dp_group,
                temporary_bucket_allocator=self.temp_mem_alloc_for_main_grad_buffer,
                init_meta_only=True,
                is_dtype_float8=not grad_reduce_in_fp32 and is_dtype_float8,
                gradient_scaling_factor=gradient_scaling_factor,
            )
            if grad_reduce_in_fp32:
                buffer_size[torch.float32] += group.main_grad_buffer.data_size
            elif is_dtype_float8:
                buffer_size["float8"] += group.main_grad_buffer.data_size
            else:
                buffer_size[group.main_grad_buffer.dtype] += group.main_grad_buffer.data_size

        # Allocate the buffer.
        if self.buffer_all_in_one:
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
            offset = {
                torch.float32: 0,
                torch.float16: 0,
                torch.bfloat16: 0,
                "float8": 0,
            }

        def _alloc(dtype, size):
            if self.buffer_all_in_one:
                if dtype == torch.uint8:
                    dtype = "float8"
                data = self.buffer[dtype][offset[dtype] : offset[dtype] + size]
                offset[dtype] += size
                return data
            return torch.empty(size, dtype=dtype, device=self.device)

        # Initialize the model weight buffer data of each parameter group.
        if data_parallel_sharding_strategy != 'NO_OP':
            for group in self.parameter_groups:
                wbuf = group.model_weight_buffer
                wbuf.data = _alloc(wbuf.dtype, wbuf.data_size)
                bucket = wbuf.fetch_the_bucket()
                for item_id, p in enumerate(group.params):
                    wbuf.set_item(item_id, p.data)

                    # reset the parameter data to the buffer
                    old_param_data = p.data
                    new_param_data = wbuf.get_item_from_bucket(bucket, item_id).view(p.shape)
                    if is_float8tensor(p):
                        p._data = new_param_data
                    else:
                        p.data = new_param_data
                    assert old_param_data._base is None
                    p.data.detach().copy_(old_param_data)
                    if wbuf.is_data_distributed:
                        p.fully_shard_param_local_index = wbuf.locate_item_in_global_item(item_id)
                        local_shard = wbuf.get_item(item_id, only_shard=True)
                        local_shard.fsdp_shard_raw_p = p.data
                        p.fully_shard_param_local_shard = local_shard
                    del old_param_data

                # Initialize the master weight buffer data of each parameter group.
                if preserve_fp32_weights:
                    mbuf = group.master_weight_buffer
                    mbuf.data = _alloc(mbuf.dtype, mbuf.data_size)
                    for item_id, p in enumerate(group.params):
                        mbuf.set_item(item_id, p.data)

                if wbuf.is_data_distributed:
                    wbuf.free_the_bucket_storage()

        # Initialize the main grad buffer data of each parameter group.
        for group in self.parameter_groups:
            gbuf = group.main_grad_buffer
            gbuf.data = _alloc(gbuf.dtype, gbuf.data_size)
            gbuf.data.zero_()
            bucket = gbuf.fetch_the_bucket()
            for item_id, p in enumerate(group.params):
                p.fsdp_managed_main_grad = gbuf.get_item(item_id)
                fsdp_pre_allocate_main_grad = True
                if fsdp_pre_allocate_main_grad:
                    p._gbuf = gbuf
                    p._main_grad = gbuf.get_item_from_bucket(bucket, item_id).view(p.shape)

                    def main_grad_getter(self):
                        # Make sure main_grad memory storage ready.
                        self._gbuf.fetch_the_bucket()
                        return self._main_grad

                    setattr(p.__class__, 'main_grad', property(main_grad_getter))
                else:
                    p.main_grad = gbuf.get_item_from_bucket(bucket, item_id).view(p.shape)

            if gbuf.is_data_distributed:
                gbuf.free_the_bucket_storage()

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        for parameter_group in self.parameter_groups:
            parameter_group.main_grad_buffer.data *= scaling_factor

    def zero_grad(self):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        for pg in self.parameter_groups:
            pg.main_grad_buffer.data.zero_()

    def _init_optimizer_named_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        named_parameters = []
        for pg in self.parameter_groups:
            optimizer_state_shard = self.ddp_config.data_parallel_sharding_strategy in [
                'OPTIMIZER_STATES',
                'OPTIMIZER_STATES_AND_GRADS',
                'MODEL_AND_OPTIMIZER_STATES',
            ]
            for item_id, orig_param in enumerate(pg.params):
                if self.preserve_fp32_weights:
                    # param = pg.master_weights[item_id]
                    param = pg.master_weight_buffer.get_item(
                        item_id, only_shard=optimizer_state_shard
                    )
                elif self.ddp_config.data_parallel_sharding_strategy == 'NO_OP':
                    param = orig_param
                else:
                    param = pg.model_weight_buffer.get_item(
                        item_id, only_shard=optimizer_state_shard
                    )

                # wbuf = pg.master_weight_buffer if self.preserve_fp32_weights else pg.model_weight_buffer
                # param = wbuf.get_item(item_id, only_shard=optimizer_state_shard)
                main_grad = pg.main_grad_buffer.get_item(item_id, only_shard=optimizer_state_shard)

                def set_param_attribute_closure(param, main_grad, orig_param):
                    def set_param_attribute():
                        setattr(param, 'grad', main_grad if main_grad.numel() > 0 else None)
                        for attr_name in [
                            'requires_grad',
                            'sequence_parallel',
                            'shared',
                            'tensor_model_parallel',
                            'partition_dim',
                            'partition_stride',
                        ]:
                            if hasattr(orig_param, attr_name):
                                setattr(param, attr_name, getattr(orig_param, attr_name))

                    return set_param_attribute

                setattr(
                    param,
                    'reset_attribute',
                    set_param_attribute_closure(param, main_grad, orig_param),
                )
                param.reset_attribute()
                named_parameters.append((self.param_to_name[orig_param], param))

        return named_parameters

    @property
    def num_buckets(self):
        return len(self.parameter_groups)

    @torch.no_grad()
    def update_model_weights(self):
        for pg in self.parameter_groups:
            master_weight_buf = pg.master_weight_buffer
            model_weight_buf = pg.model_weight_buffer

            if master_weight_buf is None:
                continue

            for param in pg.params:
                item_id = model_weight_buf.param_idx[param]

                if model_weight_buf:
                    if model_weight_buf.is_data_distributed or master_weight_buf.is_data_distributed:
                        operate_model_param = model_weight_buf.get_item(item_id, only_shard=True)
                        operate_master_weight = master_weight_buf.get_item(item_id, only_shard=True)
                    else:
                        operate_model_param = model_weight_buf.get_item(item_id)
                        operate_master_weight = master_weight_buf.get_item(item_id)
                else:
                    assert not master_weight_buf.is_data_distributed
                    operate_model_param = param
                    operate_master_weight = pg.master_weight_buffer.get_item(item_id)

                if operate_model_param.numel() == 0:
                    continue

                if is_float8tensor(param):
                    # 1. When "--fp8-param-gather" is disabled, the main param is first casted to
                    # BF16/FP16, and then casted to FP8, so the amax_history is calculated
                    # using BF16/FP16 param.
                    # 2. When "--fp8-param-gather" is enabled, we can cast the FP32 main param to
                    # FP8 directly, which results in slightly different results with higher
                    # performance. In theory, this does not affect convergence.
                    # TODO: The following code maintains the logic of the point-1 above. It can
                    # be deleted if it is not necessary.
                    operate_master_weight = operate_master_weight.to(param.dtype)
                    cast_to_fp8(
                        operate_master_weight.view(1, -1),
                        param._fp8_meta['scaling_fwd'],
                        param._fp8_meta_index,
                        param._fp8_dtype,
                        out=operate_model_param.view(1, -1),
                    )
                else:
                    operate_model_param.data.copy_(operate_master_weight)

    @torch.no_grad()
    def copy_model_weight_to_master_weight(self):
        if self.preserve_fp32_weights:
            for group in self.parameter_groups:
                mbuf = group.master_weight_buffer
                for item_id, p in enumerate(group.params):
                    mbuf.set_item(item_id, p.data)

    def all_gather_parameters(self, async_op: bool = True):
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
        assert all(
            [not g.main_grad_buffer.is_data_distributed for g in self.parameter_groups]
        ), 'reduce_scatter_gradients() should only be called when gradients are not sharded.'

        reduce_scatter_ops = []
        for g in self.parameter_groups:
            gbuf = g.main_grad_buffer
            if self.ddp_config.average_in_collective:
                reduce_op = torch.distributed.ReduceOp.AVG
            else:
                reduce_op = torch.distributed.ReduceOp.SUM
                if g.main_grad_buffer.gradient_scaling_factor is not None:
                    g.main_grad_buffer.data *= g.main_grad_buffer.gradient_scaling_factor

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
        assert all(
            [not g.main_grad_buffer.is_data_distributed for g in self.parameter_groups]
        ), 'all_reduce_gradients() should only be called when gradients are not sharded.'

        all_reduce_ops = []
        for g in self.parameter_groups:
            if self.buffer.ddp_config.average_in_collective:
                reduce_op = torch.distributed.ReduceOp.AVG
            else:
                reduce_op = torch.distributed.ReduceOp.SUM
                if g.main_grad_buffer.gradient_scaling_factor is not None:
                    g.main_grad_buffer.data *= g.main_grad_buffer.gradient_scaling_factor
            all_reduce_handler = torch.distributed.all_reduce(
                g.main_grad_buffer.data,
                op=reduce_op,
                group=g.main_grad_buffer.data_parallel_group,
                async_op=async_op,
            )
            if async_op:
                all_reduce_ops.append(all_reduce_handler)

        for op in all_reduce_ops:
            op.wait()


class BucketStatus(Enum):
    EMPTY = 1
    COMMUNICATING = 2
    READY_TO_USE = 3


class GradReducePipeline:
    def __init__(
        self,
        param_and_grad_buffer: ParamAndGradBuffer,
        cuda_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.buffer = param_and_grad_buffer
        self.grad_reduce_queue = []
        self.bucket_status = [BucketStatus.EMPTY for _ in range(self.buffer.num_buckets)]
        self.buckets = {}
        self.cuda_stream = cuda_stream

    @property
    def num_buckets(self):
        return self.buffer.num_buckets

    def reset(self):
        assert (
            len(self.grad_reduce_queue) == 0
        ), f"There are still pending reduce-scatter tasks, it is not safe to reset. items: {self.grad_reduce_queue.keys()}, bucket_status: {self.bucket_status}."
        for bucket_id, status in enumerate(self.bucket_status):
            if status == BucketStatus.READY_TO_USE:
                gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                gbuf.free_the_bucket_storage()
            self.bucket_status[bucket_id] = BucketStatus.EMPTY
        assert all(
            [status is BucketStatus.EMPTY for status in self.bucket_status]
        ), f"There are still pending buckets, it is not safe to reset. bucket_status: {self.bucket_status}."

        self.buckets = {}

    def place_the_bucket(self, bucket_id: int) -> bool:
        """Place a full size bucket by bucket id.
        Args:
            bucket_id (int): The bucket id.
        Returns:
            bool: True if the bucket is placed successfully.
        """
        bucket_status = self.bucket_status[bucket_id]
        if bucket_status == BucketStatus.READY_TO_USE:
            return False
        if bucket_status == BucketStatus.COMMUNICATING:
            self.wait_for_previous_grad_reduce(0)

        assert bucket_id not in self.buckets, f"Bucket {bucket_id} is already allocated."

        gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
        bucket = gbuf.fetch_the_bucket()
        requires_grad_items = sum([p.requires_grad for p in gbuf.params])
        setattr(bucket, 'requires_grad_items', requires_grad_items)
        setattr(bucket, 'items', [])

        self.buckets[bucket_id] = bucket
        self.bucket_status[bucket_id] = BucketStatus.READY_TO_USE
        return True

    def wait_for_previous_grad_reduce(self, recommeded_queue_size: int = 1, recommeded_queue_capacity: Optional[int] = None):
        if recommeded_queue_capacity is not None:
            queue_space = sum([self.buffer.parameter_groups[bucket_id].main_grad_buffer.bucket_index.size for _, _, bucket_id in self.grad_reduce_queue])
            while queue_space > recommeded_queue_capacity:
                async_work_handler, callback, bucket_id = self.grad_reduce_queue.pop(0)
                async_work_handler.wait()
                callback()
                queue_space -= self.buffer.parameter_groups[bucket_id].main_grad_buffer.bucket_index.size
        else:
            recommeded_queue_size = max(0, min(recommeded_queue_size, self.buffer.num_buckets - 1))
            while len(self.grad_reduce_queue) > recommeded_queue_size:
                async_work_handler, callback, _ = self.grad_reduce_queue.pop(0)
                async_work_handler.wait()
                callback()

    def mark_item_ready(self, item: torch.Tensor, async_rs: bool = False) -> bool:
        """Mark the item ready for reduce-scatter/all-reduce.
        Args:
            item (torch.Tensor): The item to be marked.
            async_rs (bool, optional): Whether to do the reduce-scatter/all-reduce asynchronously. Defaults to False.
        Returns:
            bool: True if the item is go for reduce-scatter/all-reduce.
        """
        bucket_id = self.buffer.param_to_param_group[item]
        assert bucket_id in self.buckets, f"Bucket {bucket_id} is not allocated."

        scaling_factor = self.buffer.gradient_scaling_factor
        bucket = self.buckets[bucket_id]
        bucket.items.append(item)
        assert len(bucket.items) <= bucket.requires_grad_items, "Too many items in the bucket."
        if len(bucket.items) != bucket.requires_grad_items:
            return False

        self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING

        current_stream = torch.cuda.current_stream()
        reduce_scatter_stream = (
            self.cuda_stream if self.cuda_stream is not None else torch.cuda.current_stream()
        )
        reduce_scatter_stream.wait_stream(current_stream)
        with torch.cuda.stream(reduce_scatter_stream):
            if self.buffer.ddp_config.average_in_collective:
                reduce_op = torch.distributed.ReduceOp.AVG
            else:
                if scaling_factor is not None:
                    bucket.data *= scaling_factor
                reduce_op = torch.distributed.ReduceOp.SUM
            gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
            if gbuf.ddp_config.data_parallel_sharding_strategy == 'NO_OP':
                torch.distributed.all_reduce(
                    bucket.data,
                    op=reduce_op,
                    group=gbuf.data_parallel_group,
                )
            else:
                grad_shard = gbuf.get_shard_from_bucket(bucket)
                torch.distributed.reduce_scatter_tensor(
                    output=grad_shard,
                    input=bucket.data,
                    op=reduce_op,
                    group=gbuf.data_parallel_group,
                )
                if gbuf.is_data_distributed:
                    # gradient accumulation on local buffer
                    local_buffer = gbuf.get_shard_from_local_buffer()
                    local_buffer += grad_shard
            reduce_scatter_view_out_event = reduce_scatter_stream.record_event()
            del self.buckets[bucket_id]

        def get_closure():
            def gradient_accumulate():
                nonlocal gbuf, grad_shard, bucket_id
                if gbuf.is_data_distributed:
                    gbuf.free_the_bucket_storage()
                self.bucket_status[bucket_id] = BucketStatus.EMPTY

            return gradient_accumulate

        gradient_accumulate = get_closure()

        if async_rs:
            self.grad_reduce_queue.append((reduce_scatter_view_out_event, gradient_accumulate, bucket_id))
            return True

        gradient_accumulate()

        return True


class PrefetchOrder(Enum):
    AFTER = 0
    BEFORE = 1


class AllGatherPipeline:
    def __init__(
        self,
        param_and_grad_buffer: ParamAndGradBuffer,
        cuda_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.buffer = param_and_grad_buffer
        self.all_gather_handler_map = {}
        self.bucket_status = [BucketStatus.EMPTY for _ in range(self.buffer.num_buckets)]
        self.bucket_items = {}
        self.cuda_stream = cuda_stream

    def reset(self):
        if len(self.all_gather_handler_map) > 0:
            warnings.warn(
                "There are still pending all-gather tasks, it is not safe to reset."
                f"Bucket status: {self.bucket_status}.",
                UserWarning,
            )
            while len(self.all_gather_handler_map) > 0:
                bucket_id = next(iter(self.all_gather_handler_map))
                self.wait_bucket_ready(bucket_id)
        self.bucket_status = [BucketStatus.EMPTY] * self.buffer.num_buckets
        self.bucket_items = {}

    def queue_bucket(
        self,
        bucket_id: int,
        prefetch: bool = False,
        prefetch_order: PrefetchOrder = PrefetchOrder.AFTER,
        prefetch_capacity: Optional[int] = None,
    ):
        parameter_groups = self.buffer.parameter_groups
        self.all_gather_bucket_and_set_items(bucket_id, async_op=True)
        if prefetch:
            if prefetch_capacity is not None:
                prefetched_space = 0
                while prefetched_space < prefetch_capacity:
                    if prefetch_order == PrefetchOrder.AFTER:
                        next_bucket_id = bucket_id + 1
                    else:
                        next_bucket_id = bucket_id - 1
                    if next_bucket_id < 0 or next_bucket_id >= self.buffer.num_buckets:
                        break

                    self.all_gather_bucket_and_set_items(
                        next_bucket_id,
                        async_op=True,
                    )
                    prefetched_space += parameter_groups[next_bucket_id].model_weight_buffer.bucket_index.size
                    bucket_id = next_bucket_id
            else:
                if prefetch_order == PrefetchOrder.AFTER:
                    prefetch_bucket_id = bucket_id + 1
                else:
                    prefetch_bucket_id = bucket_id - 1
                if prefetch_bucket_id >= 0 and prefetch_bucket_id < self.buffer.num_buckets:
                    self.all_gather_bucket_and_set_items(
                        prefetch_bucket_id,
                        async_op=True,
                    )

    def wait_bucket_ready(self, bucket_id, empty_ok=False):
        if self.bucket_status[bucket_id] == BucketStatus.READY_TO_USE:
            return
        if self.bucket_status[bucket_id] == BucketStatus.EMPTY:
            if empty_ok:
                return
            raise ValueError(f"Bucket {bucket_id} is empty.")

        async_work_handler, callback = self.all_gather_handler_map.pop(bucket_id)
        async_work_handler.wait()

        callback()

    @torch.no_grad()
    def release_item(self, item: torch.Tensor):
        bucket_id = self.buffer.param_to_param_group[item]
        assert (
            self.bucket_status[bucket_id] == BucketStatus.READY_TO_USE
        ), f"Bucket {bucket_id} is not ready, {self.bucket_status[bucket_id]}."
        if bucket_id not in self.bucket_items:
            return False

        bucket_items = self.bucket_items[bucket_id]
        bucket_items.discard(item)
        if len(bucket_items) == 0:
            del bucket_items, self.bucket_items[bucket_id]
            self.buffer.parameter_groups[bucket_id].model_weight_buffer.free_the_bucket_storage()
            self.bucket_status[bucket_id] = BucketStatus.EMPTY
            return True
        return False

    @torch.no_grad()
    def all_gather_bucket_and_set_items(
        self, bucket_id: int, async_op: bool = False,
    ) -> None:
        if self.bucket_status[bucket_id] != BucketStatus.EMPTY:
            return

        self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING

        model_weight_buf = self.buffer.parameter_groups[bucket_id].model_weight_buffer

        current_stream = torch.cuda.current_stream()
        all_gather_stream = (
            self.cuda_stream if self.cuda_stream is not None else torch.cuda.current_stream()
        )
        all_gather_stream.wait_stream(current_stream)
        with torch.cuda.stream(all_gather_stream):
            shard = model_weight_buf.get_shard_from_local_buffer()

            bucket = model_weight_buf.fetch_the_bucket()
            all_gather_handler = torch.distributed.all_gather_into_tensor(
                output_tensor=bucket.data,
                input_tensor=shard,
                group=model_weight_buf.data_parallel_group,
                async_op=async_op,
            )

        def get_closure():
            @torch.no_grad()
            def mark_bucket_ready():
                nonlocal model_weight_buf, bucket_id
                self.bucket_items[bucket_id] = set(model_weight_buf.params.copy())
                self.bucket_status[bucket_id] = BucketStatus.READY_TO_USE

            return mark_bucket_ready

        mark_bucket_ready = get_closure()

        if async_op:
            self.all_gather_handler_map[bucket_id] = (all_gather_handler, mark_bucket_ready)
            return
        mark_bucket_ready()
