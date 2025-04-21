# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import List, Union
import torch.distributed
from megatron.core.device_utils import get_current_device, get_xla_model
import torch

from megatron.core.parallel_state import get_global_memory_buffer
from megatron.core.utils import get_tensor_model_parallel_group_if_none, is_torch_min_version
from megatron.core.wrapped_process_group import WrappedProcessGroup

from .utils import split_tensor_along_last_dim

xm = get_xla_model()
if xm:
    xm_reduce_op = {
        torch.distributed.ReduceOp.SUM: xm.REDUCE_SUM,
        torch.distributed.ReduceOp.MAX: xm.REDUCE_MAX,
        torch.distributed.ReduceOp.MIN: xm.REDUCE_MIN,
        torch.distributed.ReduceOp.PRODUCT: xm.REDUCE_MUL,
        torch.distributed.ReduceOp.BAND: xm.REDUCE_AND,
        torch.distributed.ReduceOp.BOR: xm.REDUCE_OR,
    }

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base


def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""
    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"

    # Bypass the function if we are using only 1 GPU.
    if group.size() == 1:
        return input_

    # All-reduce.
    all_reduce(input_.contiguous(), group=group)

    return input_


def _split_along_last_dim(input_, group):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"

    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = group.rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"

    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = group.rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_: torch.Tensor, group):
    """Gather tensors and concatinate along the last dimension."""

    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"

    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    if xm:
        output = xm.all_gather(input_.contiguous(), dim=-1, groups=group.rank_groups, pin_layout=False)
    else:
        output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
        dist_all_gather_func(output, input_.contiguous(), group=group.process_group)
        tensor_list = output.chunk(world_size, dim=0)
        output = torch.cat(tensor_list, dim=-1).contiguous()
    
    return output


def _reduce_scatter_along_last_dim(input_, group):
    """Reduce-scatter tensors on the last dimension."""

    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"
    
    world_size = group.size()
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor, group=group).reshape(target_shape)
    return output


def _gather_along_first_dim(input_: torch.Tensor, 
                            group, 
                            output_split_sizes: List[int]=None, 
                            use_global_buffer: bool=False):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """

    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"
    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
    
        if xm:
            output = xm.all_gather(input_.contiguous(), dim=0, 
                                   groups=group.rank_groups, pin_layout=False)
        else:
            if use_global_buffer:
                output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
            else:
                output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
            dist_all_gather_func(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        if xm:
            output = xm.all_gather(input_.contiguous(), 
                                   groups=group.rank_groups, pin_layout=False)
            output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        else:
            if use_global_buffer:
                output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
            else:
                output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
            output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
            torch.distributed.all_gather(output_tensor_list, input_, group=group.process_group)
        output = torch.cat(output_tensor_list, dim=0).contiguous()

    return output


def _reduce_scatter_along_first_dim(input_: torch.Tensor, 
                                    group, 
                                    input_split_sizes: List[int]=None, 
                                    use_global_buffer:bool=False
):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    assert isinstance(group, WrappedProcessGroup), \
        "group {group} should be a WrappedProcessGroup"
    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        if xm:
            output = xm.reduce_scatter(xm.REDUCE_SUM, input_.contiguous(), 1.0, 0, world_size,
                                       groups=group.rank_groups, pin_layout=False)      
        else:
            if use_global_buffer:
                output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
            else:
                output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
            dist_reduce_scatter_func(output, input_.contiguous(), group=group.process_group)
    else:
        rank = group.rank()
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))

        if xm:
            xm.reduce_scatter(xm.REDUCE_SUM, input_tensor_list, 1.0, 0, world_size,
                                       groups=group.rank_groups, pin_layout=False)
            output = input_tensor_list[rank]
        else:
            if use_global_buffer:
                output = get_global_memory_buffer().get_tensor(
                    input_tensor_list[rank].shape, input_.dtype, "mpu"
                )
            else:
                output = torch.empty_like(input_tensor_list[rank])
            torch.distributed.reduce_scatter(output, input_tensor_list, group=group.process_group)
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce(grad_output, ctx.group), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _reduce(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _split_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output, ctx.group), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _split_along_last_dim(grad_output, ctx.group), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _split_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_first_dim(grad_output, ctx.group), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(
        graph,
        input_,
        group=None,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Symbolic function for tracing."""
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def forward(
        ctx,
        input_,
        group=None,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Forward function."""
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

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
                _reduce_scatter_along_first_dim(
                    grad_output, ctx.group, ctx.output_split_sizes, ctx.use_global_buffer
                ),
                None,
                None,
                None,
                None,
            )
        else:
            assert ctx.output_split_sizes is None
            return _split_along_first_dim(grad_output, ctx.group), None, None, None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group=None, input_split_sizes=None, use_global_buffer=False):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, group=None, input_split_sizes=None, use_global_buffer=False):
        """Forward function."""
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        input_split_sizes = ctx.input_split_sizes
        use_global_buffer = ctx.use_global_buffer
        return (
            _gather_along_first_dim(grad_output, ctx.group, input_split_sizes, use_global_buffer),
            None,
            None,
            None,
        )


class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce_scatter_along_last_dim(grad_output, ctx.group), None


class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output, ctx.group), None


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        rank = group.rank()
        input = input.contiguous()
        if xm:
            groups = group.rank_groups
            if output_split_sizes is None:
                orig_dtype = input.dtype
                input = input.to(dtype=torch.float32)
                output = xm.all_to_all(value=input,
                    split_dimension=0,
                    concat_dimension=0,
                    split_count=world_size,
                    groups=groups,
                    pin_layout=False)
                output = output.to(dtype=orig_dtype)
            else:
                input_splits = torch.tensor(input_split_sizes, device=input.device, dtype=torch.int)
                all_input_splits = xm.all_gather(input_splits, dim=0, groups=groups, pin_layout=False).split(world_size)
                all_input_splits = [ [ y.item() for y in x] for x in all_input_splits ]
                all_input_sizes = [ sum(x) for x in all_input_splits]

                output_splits = torch.tensor(output_split_sizes, device=input.device, dtype=torch.int)
                all_output_splits = xm.all_gather(output_splits, dim=0, groups=groups, pin_layout=False).split(world_size)
                all_output_splits = [ [ y.item() for y in x] for x in all_output_splits ]

                max_dim = max(all_input_sizes)
                paddings = [ 0 for _ in range(2*input.dim())]
                paddings[-1] = (max_dim - input.size()[0])
                paddings = tuple(paddings)
                input = torch.nn.functional.pad(input, paddings, value=0.0)
                all_inputs = xm.all_gather(input, dim=0, groups=groups, pin_layout=False)
                all_inputs = all_inputs.split(max_dim)
                all_inputs = [ torch.split(x[:all_input_sizes[i]], all_input_splits[i]) for i, x in enumerate(all_inputs) ]

                for i, x in enumerate(all_inputs[rank]):
                    assert x.size()[0] == all_input_splits[rank][i], f"{x.size()[0]} != {all_input_splits[rank][i]}"

                all_outputs = [ [ x[r] for x in all_inputs] for r in range(world_size) ]
                for i, x in enumerate(all_outputs[rank]):
                    assert x.size()[0] == all_output_splits[rank][i], f"{x.size()[0]} != {all_output_splits[rank][i]}"
                output = torch.cat( all_outputs[rank], dim=0).to(device=input.device)
        else:
            if output_split_sizes is None:
                # Equal split (all2all)
                output = torch.empty_like(input)
            else:
                # Unequal split (all2all-v)
                output = input.new_empty(
                    size=[sum(output_split_sizes)] + list(input.size()[1:]),
                    dtype=input.dtype,
                    device=get_current_device(),
                )
            torch.distributed.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group.process_group,
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


def copy_to_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: copy, backward allreduce"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: all reduce, backward copy"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: RS, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: AG, backward: split <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _GatherFromModelParallelRegion.apply(input_, group)


def scatter_to_sequence_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: split, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _ScatterToSequenceParallelRegion.apply(input_, group)


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_global_buffer=False,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallel_output_grad, output_split_sizes, use_global_buffer
    )


def reduce_scatter_to_sequence_parallel_region(
    input_, group=None, input_split_sizes=None, use_global_buffer=False
):
    """Wrapper for autograd function: forward: RS, backward AG <fisrt dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, group, input_split_sizes, use_global_buffer
    )


def all_gather_last_dim_from_tensor_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: AG, backward RS <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _AllGatherFromTensorParallelRegion.apply(input_, group)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: RS, backward AG: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _ReduceScatterToTensorParallelRegion.apply(input_, group)


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for autograd function"""
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)


def all_to_all_sp2hp(input_, group=None):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (torch.Tensor):
            The input tensor which has been distributed along the sequence
            dimension.
        group (torch.distributed.ProcessGroup, optional):
            The process group to work on. If None, the tensor model parallel group
            will be used.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)

    world_size = group.size()
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(group, concat_tensor)
    return output


def all_to_all_hp2sp(input_, group=None):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (torch.Tensor):
            The input tensor which has been distributed along the hidden
            dimension.
        group (torch.distributed.ProcessGroup, optional):
            The process group to work on. If None, the tensor model parallel group
            will be used.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    group = get_tensor_model_parallel_group_if_none(group, wrapped=True)

    world_size = group.size()
    input_ = input_.reshape(-1, input_.shape[-1])
    input_exchanged = all_to_all(group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
    )
    output = torch.cat(split_tensors, dim=-1)
    return output


def all_reduce(tensor: torch.Tensor, 
               group, 
               op=torch.distributed.ReduceOp.SUM,
               async_op:bool = False):
    if xm:
        assert op in xm_reduce_op, f"Reduce Op {op} not supported in XLA"
        groups = group.rank_groups if group else None
        xm.all_reduce(reduce_type=xm_reduce_op[op], inputs=[tensor], groups=groups, pin_layout=False)
    else:
        group = group.process_group if group else None
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)
