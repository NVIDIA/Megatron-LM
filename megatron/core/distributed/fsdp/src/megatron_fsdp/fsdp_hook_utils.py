# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import functools
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def register_backward_hook(module, custom_backward_handler):
    """
    Creates a custom backward hook via attaching a gradient-triggered hook
    to the output tensor(s) of a module during a post-forward hook.
    """

    def forward_hook(_module, inputs, output):
        # Replace the output to avoid the output tensor being the same as
        # the input tensor, which makes it impossible to identify which
        # layer's output it is. Using view_as to make it does not cause
        # additional memory consumption.
        output = tree_map(lambda t: t.view_as(t) if torch.is_tensor(t) else t, output)

        output_list = []

        # Post-process forward output.
        if isinstance(output, torch.Tensor):
            output_list = [output]
        elif isinstance(output, (tuple, list)):
            output_list = [t for t in output if isinstance(t, torch.Tensor)]

        # Register pre-backward hook on the output tensor(s). This hook
        # will trigger immediately after the gradients of the output
        # tensor(s) have been computed.
        torch.autograd.graph.register_multi_grad_hook(
            output_list, lambda grads: custom_backward_handler(_module, grads), mode="any"
        )
        return output

    # Register the post-forward hook that attaches the custom backward hook
    # on the output tensor(s).
    return module.register_forward_hook(forward_hook)


def register_post_backward_hook(module, post_backward_hook: callable):
    """
    Register a pre-forward hook that attaches a post-backward hook to the module.
    The post-backward hook will be called after the backward pass of the module.
    """

    def _post_backward_hook(
        post_backward_hook: callable,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Pre-forward hook utilized to attach a gradient reduction post-backward
        hook to the module.
        """
        # Register the backward function to reduce gradients after the backward pass.
        # And for optim_grads_params, we need to release the parameters after the backward pass.
        if not torch.is_grad_enabled():
            return args, kwargs

        # Preprocess the input arguments.
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)

        if len(inp_tensors) == 0:
            return args, kwargs

        """
        Bootstrapped identity autograd function that attaches a post-backward
        "hook" to the module to trigger model resharding / deallocation and
        gradient reduce-scatter immediately after the module backward pass has
        completed to deallocate this layer's model and gradient memory before
        the subsequent backward pass.
        """
        inp_tensors = RegisterFSDPBackwardFunction.apply(
            functools.partial(post_backward_hook, module), *inp_tensors
        )

        # Post-process the input arguments for input into the module.
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)

        # Return original input to the module forward pass.
        return args, kwargs

    return module.register_forward_pre_hook(
        functools.partial(_post_backward_hook, post_backward_hook), with_kwargs=True
    )


class RegisterFSDPBackwardFunction(torch.autograd.Function):
    """
    Register a backward function that will be called after the backward pass
    of the model. This function is used to release the parameters after the
    backward pass.
    """

    @staticmethod
    def forward(ctx, post_backward, *inputs: torch.Tensor):
        """
        Forward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward = post_backward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        """
        Backward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward()
        return (None,) + grads
