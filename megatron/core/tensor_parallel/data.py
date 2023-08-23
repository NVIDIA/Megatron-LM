# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
)
from deepspeed.accelerator import get_accelerator

_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert data[key].dtype == target_dtype, '{} has data type {} which '\
            'is different than {}'.format(key, data[key].dtype, target_dtype)


def _build_key_size_numel_dictionaries(keys, data, group=None, rank=-1, src_rank=-1):
    if group is None:
        group = get_tensor_model_parallel_group()
    if src_rank < 0:
        src_rank = get_tensor_model_parallel_src_rank()
    if rank < 0:
        rank = get_tensor_model_parallel_rank()
                    
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if rank == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = get_accelerator().LongTensor(sizes)
    torch.distributed.broadcast(sizes_cuda, src_rank, group=group)

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    if get_sequence_parallel_world_size() > 1:
        rank = get_sequence_parallel_rank()
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
    else:
        rank = get_tensor_model_parallel_rank()
        src_rank = get_tensor_model_parallel_src_rank()
        group = get_tensor_model_parallel_group()

    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(
        keys, data, group=group, rank=rank, src_rank=src_rank)

    # Pack on rank zero.
    if rank == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0).to(get_accelerator().device_name())
    else:
        flatten_data = torch.empty(total_numel,
                                   device=get_accelerator().current_device_name(),
                                   dtype=datatype)

    # Broadcast
    torch.distributed.broadcast(flatten_data, src_rank, group=group)

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output
