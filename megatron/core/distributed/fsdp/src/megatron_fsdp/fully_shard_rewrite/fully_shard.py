import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .param_group import ParameterGroup


def fully_shard(
    module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = None,
    mp_policy: Optional["MixedPrecisionPolicy"] = None,
    offload_policy: Optional["OffloadPolicy"] = None,
    ignored_params: set[nn.Parameter] | None = None,
):
    if isinstance(module, FSDPModule):
        raise ValueError(
            "The input module has already been fully sharded. "
            "Please do not call fully_shard on the same module more than once."
        )

    # Make the module class as FSDPModule and register the FSDP forward/backward hooks
    cls = module.__class__
    new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), {})
    module.__class__ = new_cls

    module._init_named_param_groups(mesh, ignored_params)
    module._init_fsdp_state()
    module._init_param_main_grad_func()
    _register_forward_pre_hook(module)
    _register_forward_hook(module)
    _register_backward_pre_hook(module)
    _register_backward_hook(module)

    module.reshard()

    return module


class FSDPModule(nn.Module):

    def _init_named_param_groups(self, mesh, ignored_params):
        ignored_params = ignored_params or set()
        ignored_modules = set()
        for _, child in self.named_modules():
            if child is not self and isinstance(child, FSDPModule):
                ignored_params.update(child.parameters())
                for child_submodule in child.modules():
                    ignored_modules.add(child_submodule)

        self._materialize_meta_module(ignored_modules)
        fsdp_param_groups = _get_module_fsdp_param_groups(self, mesh, ignored_params=ignored_params)

        setattr(self, "_fsdp_param_groups", fsdp_param_groups)
        param_to_name = {p: n for n, p in self.named_parameters()}
        self._named_param_groups = []
        for fsdp_param_group in self._fsdp_param_groups:
            param_names = []
            for param in fsdp_param_group.params:
                param_name = param_to_name[param]
                param_names.append(param_name)
            self._named_param_groups.append((param_names, fsdp_param_group))

    def _init_param_main_grad_func(self):
        def main_grad_getter(p):
            # Make sure main_grad memory is allocated when initially accessed.
            # When gradients are sharded, we can pre-allocate a communication
            # bucket to avoid casting to a communication data-type. Otherwise,
            # return the item backed by the main gradient buffer required to
            # support un-sharded gradient accumulation at high precision.
            gbuf = p._gbuf
            item_id = p._item_id

            gbuf_data = gbuf.fetch_unsharded_buffer()
            assert gbuf_data is not None
            assert gbuf_data.numel() > 0
            offset, size = gbuf.buffer_index._get_item_offset(item_id)
            grad_data = gbuf_data[offset : offset + size].view(p.shape)

            return grad_data

        for param_group in self._fsdp_param_groups:
            for param in param_group.params:
                setattr(param, "_gbuf", param_group.main_grad_buffer)
                setattr(param, "_item_id", param_group.param_idx[param])
                param.get_main_grad = main_grad_getter.__get__(param)

    def _materialize_meta_module(self, ignored_modules):
        materialization_device = torch.cuda.current_device()
        for m in self.modules():
            if m in ignored_modules:
                continue
            if all(not p.is_meta for p in m.parameters(recurse=False)):
                continue

            m.to_empty(device=materialization_device, recurse=False)
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if hasattr(m, "_reset_parameters"):
                m._reset_parameters()

    def _init_fsdp_state(self):
        setattr(self, "_fsdp_state", _FSDPState())
        for child in self.modules():
            if child is not self and isinstance(child, FSDPModule):
                child._init_fsdp_state()
                child._fsdp_state._is_root = False

    def unshard(self):
        unsharded_param_names = set()
        for param_names, param_group in self._named_param_groups:
            unsharded_param_names.update(param_names)
            if getattr(self, "_enable_nan_checks", False):
                for name, dist_param in zip(param_names, param_group.dist_params):
                    assert not torch.isnan(
                        dist_param._local_tensor
                    ).any(), f"NaN detected in dist param for parameter {name} in unshard for module {self}"

            param_group.unshard()

            for name, param in zip(param_names, param_group.params):
                _replace_module_parameter(self, name, param)

                if getattr(self, "_enable_nan_checks", False):
                    assert not torch.isnan(
                        param
                    ).any(), f"NaN detected in parameter {name} in unshard for module {self}"

        for name, param in self.named_parameters():
            if name not in unsharded_param_names:
                continue
            assert not isinstance(
                param, DTensor
            ), f"Parameter {name} is expected to be unsharded in unshard for module {self}, but got a sharded tensor. "

    def reshard(self):
        for param_names, param_group in self._named_param_groups:
            param_group.reshard()

            for name, dist_param in zip(param_names, param_group.dist_params):
                _replace_module_parameter(self, name, dist_param)

    def reduce_grad(self):
        for param_names, param_group in self._named_param_groups:
            if not param_group.requires_grad:
                continue

            if getattr(self, "_enable_nan_checks", False):
                for param in param_group.params:
                    if param.grad is None:
                        continue
                    assert not torch.isnan(
                        param.grad
                    ).any(), f"NaN detected in parameter {param} in post_backward for module {self}"

            for name, param in zip(param_names, param_group.params):
                main_grad = param.get_main_grad()
                if param.grad is None:
                    main_grad.zero_()
                else:
                    main_grad.copy_(param.grad.detach())
                    del param.grad

            param_group.reduce_grad(async_op=False)

            for name, param, dist_param, dist_grad in zip(
                param_names, param_group.params, param_group.dist_params, param_group.dist_grads
            ):
                if param.requires_grad and dist_grad is not None:
                    setattr(dist_param, "grad", dist_grad.to(param.dtype))

            if getattr(self, "_enable_nan_checks", False):
                for name, dist_grad in zip(param_names, param_group.dist_grads):
                    if dist_grad is None:
                        continue
                    assert not torch.isnan(
                        dist_grad._local_tensor
                    ).any(), f"NaN detected in dist grad for parameter {name} in post_backward for module {self}"

            param_group.main_grad_buffer.reshard()

        setattr(self, "needs_grad_reduce", False)

    def _scale_gradients(self, scaling_factor: float):
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_grad_buffer is not None:
                    param_group.main_grad_buffer.data.mul_(scaling_factor)

    def _zero_grad_buffer(self):
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_grad_buffer is not None:
                    param_group.main_grad_buffer.data.zero_()

    def _copy_main_weights_to_model_weights(self):
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_weight_buffer is None:
                    continue
                param_group.model_weight_buffer.data.copy_(param_group.main_weight_buffer.data)

    def _set_nan_check(self, enable_nan_checks: bool):
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            setattr(child, "_enable_nan_checks", enable_nan_checks)


class _FSDPState:

    def __init__(self):
        self._is_root = True
        self._post_backward_callback_queued = False


def _get_module_fsdp_param_groups(
    module, mesh: DeviceMesh | None = None, ignored_params: set[nn.Parameter] | None = None
):
    param_groups = {}

    for param in module.parameters():
        if ignored_params is not None and param in ignored_params:
            continue

        param_attrs = (param.device, param.dtype, param.requires_grad)
        if param_attrs not in param_groups:
            param_groups[param_attrs] = []
        param_groups[param_attrs].append(param)

    fsdp_param_groups = []
    for params in param_groups.values():
        fsdp_param_groups.append(ParameterGroup(params, params[0].dtype, mesh=mesh))

    return fsdp_param_groups


def _register_forward_pre_hook(module: FSDPModule):
    def unshard_param_groups(module, *unused):
        module.unshard()

    module._mfsdp_forward_pre_hook = module.register_forward_pre_hook(
        unshard_param_groups, prepend=True
    )


def _register_forward_hook(module: FSDPModule):
    def reshard_param_groups(module, *unused):
        module.reshard()

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

    def pre_backward_hook(module, grads):
        setattr(module, "needs_grad_reduce", True)
        if module._fsdp_state._is_root and not module._fsdp_state._post_backward_callback_queued:
            _register_post_backward_final_callback(module._fsdp_state, module)
        module.unshard()

    module._mfsdp_backward_pre_hook = create_custom_backward_hook(
        module, custom_backward_handler=pre_backward_hook
    )


def _register_backward_hook(module: FSDPModule):
    def post_backward(module):
        module.reshard()
        module.reduce_grad()

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


def _register_post_backward_final_callback(state: _FSDPState, module: nn.Module) -> None:
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
    assert state._is_root, "Only the root FSDP instance should register the post-backward callback"
    if state._post_backward_callback_queued:
        return

    def _post_backward_final_callback(root_state: _FSDPState, root_module: nn.Module) -> None:
        # Reshard all FSDP modules after the backward pass is done.
        # for module in root_module.modules():
        for name, module in root_module.named_modules():
            if not isinstance(module, FSDPModule):
                continue
            module.reshard()
            if getattr(module, "needs_grad_reduce", False):
                module.reduce_grad()

        # Reset the flag
        root_state._post_backward_callback_queued = False

    # Trace does not need this callback
    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
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


def _replace_module_parameter(module, name, new_param):
    """
    Replace a module's parameter with a new parameter, preserving the hierarchy.
    """
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:  # Navigate to parent module
        parent = getattr(parent, part)

    # Replace the parameter
    setattr(parent, parts[-1], new_param)
