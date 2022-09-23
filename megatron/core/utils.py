"""Utility functions used through Megatron core"""
import torch

from megatron.core import parallel_state


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_into_1d_equal_chunks(tensor):
    """Break a tensor into equal 1D chunks."""
    data = tensor.view(-1)
    partition_size = (
        torch.numel(data) // parallel_state.get_tensor_model_parallel_world_size()
    )
    start_index = partition_size * parallel_state.get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    return data[start_index:end_index]


def gather_split_1d_tensor(tensor):
    """Opposite of above function, gather values from model parallel ranks."""
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed._all_gather_base(
        gathered,
        tensor,
        group=parallel_state.get_tensor_model_parallel_group()
        )
    return gathered
