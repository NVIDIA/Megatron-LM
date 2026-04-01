import functools
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .param_group import ParameterGroup
from .mixed_precision import MixedPrecisionPolicy


def fully_shard(
    module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: "OffloadPolicy" = None,
    ignored_params: set[nn.Parameter] | None = None,
):
    if isinstance(module, FSDPModule):
        raise ValueError(
            "The input module has already been fully sharded. "
            "Please do not call fully_shard on the same module more than once."
        )

    fsdp_sub_modules = []
    for name, child in module.named_modules():
        if isinstance(child, FSDPModule):
            fsdp_sub_modules.append(child)

    if len(fsdp_sub_modules) > 0:
        ignored_params = set() if ignored_params is None else ignored_params
        for fsdp_sub_module in fsdp_sub_modules:
            ignored_params.update(fsdp_sub_module.parameters())

    fsdp_param_groups = _get_module_fsdp_param_groups(
        module, mesh=mesh, ignored_params=ignored_params
    )

    # Replace the module class with FSDPModule and register the FSDP forward/backward hooks

    module.__class__ = FSDPModule
    setattr(module, "_fsdp_param_groups", fsdp_param_groups)
    _register_forward_pre_hook(module)
    _register_forward_hook(module)
    _register_backward_pre_hook(module)
    _register_backward_hook(module)
    # _register_post_accumulate_grad_hooks(module)

    return module


class FSDPModule(nn.Module):
    pass


def _get_module_fsdp_param_groups(
    module, mesh: DeviceMesh | None = None, ignored_params: set[nn.Parameter] | None = None
):
    dtype_to_param_groups = {
        torch.bfloat16: [],
        torch.float16: [],
        torch.float32: [],
        torch.float64: [],
    }

    for param in module.parameters():
        if ignored_params is not None and param in ignored_params:
            continue
        assert (
            param.dtype in dtype_to_param_groups
        ), f"Unsupported dtype {param.dtype} for FSDP parameters."
        dtype_to_param_groups[param.dtype].append(param)

    param_groups = []
    for dtype, params in dtype_to_param_groups.items():
        param_groups.append(ParameterGroup(params, dtype, mesh=mesh))

    return param_groups


def _register_forward_pre_hook(module: FSDPModule):
    def unshard_param_groups(module, *unused):
        for fsdp_param_group in module._fsdp_param_groups:
            fsdp_param_group.unshard()

    module._mfsdp_forward_pre_hook = module.register_forward_pre_hook(
        unshard_param_groups, prepend=True
    )


def _register_forward_hook(module: FSDPModule):
    def reshard_param_groups(module, *unused):
        for fsdp_param_group in module._fsdp_param_groups:
            fsdp_param_group.reshard()

    module._mfsdp_forward_hook = module.register_forward_hook(reshard_param_groups)


def _register_backward_pre_hook(module: FSDPModule):
    def create_custom_backward_hook(module, custom_backward_handler):
        """
        Creates a custom backward hook via attaching a gradient-triggered hook
        to the output tensor(s) of a module during a post-forward hook.
        """

        @torch.compiler.disable
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

    def param_groups_unshard(module, grads):
        for fsdp_param_group in module._fsdp_param_groups:
            fsdp_param_group.unshard()

    module._mfsdp_backward_pre_hook = create_custom_backward_hook(
        module, custom_backward_handler=param_groups_unshard
    )


def _register_backward_hook(module: FSDPModule):
    def post_backward(module, grads):
        for fsdp_param_group in module._fsdp_param_groups:
            fsdp_param_group.reshard()
            fsdp_param_group.grad_reduce()

    @torch.compiler.disable
    def _register_post_backward_hook(
        post_backward_hook: callable,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Register a post-backward hook for the given module by inserting an autograd
        Function in front of it. Note that a post-backward hook implemented in this
        way is not compatible with in-place modifications of the module's inputs,
        since such operations can trigger an autograd error that
        "the output is a view and is being modified in-place".
        """
        if not torch.is_grad_enabled():
            # No gradients / backward pass, don't attach the post-backward hook.
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
        Identity autograd Function that attaches a post-backward "hook" to the
        module, triggering parameter deallocation immediately after the module's
        backward pass has completed in order to shard this layer's model memory
        once the current backward stage is done.
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

    module._mfsdp_backward_hook = module.register_forward_pre_hook(
        functools.partial(_register_post_backward_hook, post_backward), with_kwargs=True
    )


def _register_post_accumulate_grad_hooks(module: FSDPModule):

    def process_post_accumulate_gradients(param_list):
        pass

    for param_group in module._fsdp_param_groups:
        for param in param_group.params:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    lambda p: process_post_accumulate_gradients([p])
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
