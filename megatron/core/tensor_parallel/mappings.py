# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_global_memory_buffer,
    get_tensor_and_expert_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=get_tensor_model_parallel_group())

    return input_


def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-1).contiguous()

    return output


def _reduce_scatter_along_last_dim(input_):
    """Reduce-scatter tensors on the last dimension."""
    world_size = get_tensor_model_parallel_world_size()
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor).reshape(target_shape)
    return output


def _gather_along_first_dim(input_, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor): A tensor to be gathered.
        output_split_sizes (List[int], optional): A list specifying the sizes of the output splits along the first dimension. If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._all_gather_base(
            output, input_.contiguous(), group=get_tensor_model_parallel_group()
        )
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(
            output_tensor_list, input_, group=get_tensor_model_parallel_group()
        )

    return output


def _reduce_scatter_along_first_dim(input_, input_split_sizes=None):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._reduce_scatter_base(
            output, input_.contiguous(), group=get_tensor_model_parallel_group()
        )
    else:
        rank = torch.distributed.get_rank(get_tensor_model_parallel_group())
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))
        output = torch.empty_like(input_tensor_list[rank])
        torch.distributed.reduce_scatter(
            output, input_tensor_list, group=get_tensor_model_parallel_group()
        )
    return output


def _gather_along_first_dim_moe(input_, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension."""
    group = get_tensor_and_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    if use_global_buffer:
        output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim_moe(input_, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_tensor_and_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size

    if use_global_buffer:
        output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


def _gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = get_expert_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True, output_split_sizes=None):
        """Symbolic function for tracing."""
        return _gather_along_first_dim(input_, output_split_sizes)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True, output_split_sizes=None):
        """Forward function."""
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.output_split_sizes = output_split_sizes
        return _gather_along_first_dim(input_, ctx.output_split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return (
                _reduce_scatter_along_first_dim(grad_output, ctx.output_split_sizes),
                None,
                None,
            )
        else:
            assert ctx.output_split_sizes is None
            return _split_along_first_dim(grad_output), None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, input_split_sizes=None):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim(input_, input_split_sizes)

    @staticmethod
    def forward(ctx, input_, input_split_sizes=None):
        """Forward function."""
        ctx.input_split_sizes = input_split_sizes
        return _reduce_scatter_along_first_dim(input_, input_split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        input_split_sizes = ctx.input_split_sizes
        return _gather_along_first_dim(grad_output, input_split_sizes), None


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_, use_global_buffer=False):
        """Symbolic function for tracing."""
        return _gather_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, use_global_buffer=False):
        """Forward function."""
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        use_global_buffer = ctx.use_global_buffer
        return _reduce_scatter_along_first_dim_moe(grad_output, use_global_buffer), None


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, use_global_buffer=False):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, use_global_buffer=False):
        """Forward function."""
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        use_global_buffer = ctx.use_global_buffer
        return _gather_along_first_dim_moe(grad_output, use_global_buffer), None


class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce_scatter_along_last_dim(grad_output)


class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _reduce_scatter_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_):
    """Wrapper for autograd function"""
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    """Wrapper for autograd function"""
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    """Wrapper for autograd function"""
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    """Wrapper for autograd function"""
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    """Wrapper for autograd function"""
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(
    input_, tensor_parallel_output_grad=True, output_split_sizes=None
):
    """Wrapper for autograd function"""
    return _GatherFromSequenceParallelRegion.apply(
        input_, tensor_parallel_output_grad, output_split_sizes
    )


def reduce_scatter_to_sequence_parallel_region(input_, input_split_sizes=None):
    """Wrapper for autograd function"""
    return _ReduceScatterToSequenceParallelRegion.apply(input_, input_split_sizes)


def gather_from_sequence_parallel_region_to_moe(input_, use_global_buffer=False):
    """Wrapper for autograd function"""
    return _GatherFromSequenceParallelRegionToMOE.apply(input_, use_global_buffer)


def reduce_scatter_to_sequence_parallel_region_from_moe(input_, use_global_buffer=False):
    """Wrapper for autograd function"""
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_, use_global_buffer)


def all_gather_last_dim_from_tensor_parallel_region(input_):
    """Wrapper for autograd function"""
    return _AllGatherFromTensorParallelRegion.apply(input_)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_):
    """Wrapper for autograd function"""
    return _ReduceScatterToTensorParallelRegion.apply(input_)


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for autograd function"""
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)


def all_to_all_sp2hp(input_):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (torch.Tensor): The input tensor which has been distributed along the sequence dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    world_size = get_tensor_model_parallel_world_size()
    tp_group = get_tensor_model_parallel_group()
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(tp_group, concat_tensor)
    return output


def all_to_all_hp2sp(input_):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (torch.Tensor): The input tensor which has been distributed along the hidden dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    world_size = get_tensor_model_parallel_world_size()
    input_ = input_.reshape(-1, input_.shape[-1])
    tp_group = get_tensor_model_parallel_group()
    input_exchanged = all_to_all(tp_group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
    )
    output = torch.cat(split_tensors, dim=-1)
    return output
