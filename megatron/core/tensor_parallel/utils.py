# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import List, Sequence

import torch

from megatron.core.utils import (
    divide,
    get_tensor_model_parallel_group_if_none,
    is_torch_min_version,
)

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Args:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False, tp_group=None):
    """Break a tensor into equal 1D chunks across tensor parallel ranks.

    Returns a Tensor or View with this rank's portion of the data.

    Args:
        tensor: The tensor to split

    Keyword Args:
        new_buffer (bool): If True, returns a new Tensor.
                           If False, returns a view into the existing Tensor.
                           Default is False

    """
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    partition_size = torch.numel(tensor) // tp_group.size()
    start_index = partition_size * tp_group.rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor, tp_group=None):
    """Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
    model parallel ranks.

    Returns a new Tensor with the gathered data.

    Args:
        tensor: A Tensor or view of this rank's portion of the data.
    """
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    numel_gathered = torch.numel(tensor) * tp_group.size()
    gathered = torch.empty(
        numel_gathered, dtype=tensor.dtype, device=torch.cuda.current_device(), requires_grad=False
    )
    dist_all_gather_func(gathered, tensor, group=tp_group)
    return gathered


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        """Vocab range from per partition vocab size."""
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """Vocab range from global vocab size."""
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
