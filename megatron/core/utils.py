# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""
import math
import operator
from functools import reduce

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    """Get an attribute from a wrapped model.
    If return_model_obj is true, return the object that has the 'attr' attribute;
    otherwise, return the attribute directly."""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else:

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module

    if return_model_obj:
        return model
    return getattr(model, attr)


def get_model_type(model):
    return get_attr_wrapped_model(model, 'model_type')


def get_model_config(model):
    return get_attr_wrapped_model(model, 'config', allow_none=False)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad,)
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def assert_viewless_tensor(tensor, extra_msg=None):
    '''Assert that a tensor is not a view (i.e., its '._base' field is
    not set).'''
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    '''
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def make_tp_sharded_tensor_for_checkpoint(
    tensor, key, tp_axis=0, replica_id=None, prepend_offsets=(), **kwargs
):
    """ Helper for instantiating a ShardedTensor where the `tp_axis` dimension is sharded across TP group.

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    if replica_id is None:
        replica_id = (0, 0, parallel_state.get_data_parallel_rank(with_context_parallel=True))

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        (
            tp_axis + prepend_axis_num,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_tensor_model_parallel_world_size(),
        ),
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )


def make_sharded_tensor_for_checkpoint(tensor, key, prepend_offsets=(), replica_id=None, **kwargs):
    """ Helper for instantiating a non-sharded ShardedTensor (replicated across TP and DP group).

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    if replica_id is None:
        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )
