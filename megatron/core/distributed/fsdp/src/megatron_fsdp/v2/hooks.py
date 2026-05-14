# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward and backward hook registration for Megatron-FSDP2."""

import functools
import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .allocator import TracePoolAllocator
from .fsdp_module import FSDPModule, _FSDPState
from .utils import RegisterFSDPBackwardFunction

logger = logging.getLogger(__name__)


def _register_forward_pre_hook(module: FSDPModule):
    """Register pre-forward hook to unshard parameters."""

    def unshard_param_groups(module, *unused):
        ctx = module._fsdp_root_context
        module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

    module._mfsdp_forward_pre_hook = module.register_forward_pre_hook(
        unshard_param_groups, prepend=True
    )


def _register_forward_hook(module: FSDPModule):
    """Register post-forward hook to reshard parameters."""

    def reshard_param_groups(module, *unused):
        ctx = module._fsdp_root_context
        if ctx.backward_phase and id(module) == ctx.backward_module:
            return
        module.reshard()

    module._mfsdp_forward_hook = module.register_forward_hook(reshard_param_groups)


def _register_backward_pre_hook(module: FSDPModule):
    """
    Register backward pre-hook to handle gradient computation.

    This uses a custom backward hook that attaches to output tensors
    to trigger unshard at the right time during backward pass.
    """

    def create_custom_backward_hook(module: FSDPModule, custom_backward_handler: Callable):
        """Create a custom backward hook attached to output tensors."""

        @torch.compiler.disable
        def forward_hook(_module, inputs, output):
            # View-as to avoid output being the same tensor object
            output = tree_map(lambda t: t.view_as(t) if torch.is_tensor(t) else t, output)

            # Collect tensor outputs
            output_list = []
            if isinstance(output, torch.Tensor):
                output_list = [output]
            elif isinstance(output, (tuple, list)):
                output_list = [t for t in output if isinstance(t, torch.Tensor)]

            # Register pre-backward hook on output tensors.
            # This triggers when gradients are computed.
            torch.autograd.graph.register_multi_grad_hook(
                output_list, lambda grads: custom_backward_handler(_module, grads), mode="any"
            )
            return output

        return module.register_forward_hook(forward_hook)

    def pre_backward_hook(module: FSDPModule, grads):
        """Hook called before backward pass for this module."""
        ctx = module._fsdp_root_context
        if module._fsdp_state._is_root:
            ctx.backward_done_modules.clear()
            ctx.backward_phase = True
            ctx._advance_backward_module()
        setattr(module, "post_backward_issued", False)
        for param_group in module._fsdp_param_groups:
            for param in param_group.params:
                setattr(param, "grad_added_to_main_grad", False)
        if module._fsdp_state._is_root and not module._fsdp_state._post_backward_callback_queued:
            _register_post_backward_final_callback(module._fsdp_state, module)
        module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)

    module._mfsdp_backward_pre_hook = create_custom_backward_hook(
        module, custom_backward_handler=pre_backward_hook
    )


def _register_backward_hook(module: FSDPModule):
    """
    Register backward hook using autograd Function.

    This inserts a RegisterFSDPBackwardFunction in the backward pass
    that triggers reshard and reduce_grad after gradients are computed.
    """

    def post_backward(module: FSDPModule):
        """Hook called after backward pass for this module."""
        ctx = module._fsdp_root_context
        ctx.backward_done_modules.add(id(module))
        ctx._advance_backward_module()
        module.reshard()
        module.reduce_grad(async_op=ctx.enable_async_reduce_grad)
        module.post_backward_issued = True

    @torch.compiler.disable
    def _register_post_backward_hook(
        post_backward_hook: Callable,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Register a post-backward hook by inserting an autograd Function.

        This approach works by registering a pre-forward hook that wraps
        input tensors in an autograd Function. The Function's backward
        calls the post_backward_hook after gradients are computed.
        """
        if not torch.is_grad_enabled():
            return args, kwargs

        # Flatten args and kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)

        # Filter to tensors with gradients
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)

        if len(inp_tensors) == 0:
            return args, kwargs

        # Wrap inputs in autograd Function.
        # The Function's backward will call post_backward_hook.
        inp_tensors = RegisterFSDPBackwardFunction.apply(
            functools.partial(post_backward_hook, module), *inp_tensors
        )

        # Restore args and kwargs
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)

        return args, kwargs

    module._mfsdp_backward_hook = module.register_forward_pre_hook(
        functools.partial(_register_post_backward_hook, post_backward), with_kwargs=True
    )


def _register_post_backward_final_callback(state: _FSDPState, module: nn.Module) -> None:
    """
    Register the final callback that runs after all backward passes complete.

    This is only registered by the root FSDP module to avoid duplicate
    callbacks. It reshards all modules and reduces gradients at the end
    of the backward pass.
    """
    assert state._is_root, "Only root FSDP should register post-backward callback"
    if state._post_backward_callback_queued:
        return

    def _post_backward_final_callback(root_state: _FSDPState, root_module: nn.Module):
        """Final callback: reshard all modules and reduce gradients."""
        ctx = root_module._fsdp_root_context
        stream = ctx.rs_stream
        for module in reversed(ctx.forward_order):
            if getattr(module, "post_backward_issued", False):
                continue
            module.reshard()
            module.reduce_grad(async_op=ctx.enable_async_reduce_grad)
        for buckets in ctx.reduce_grad_buckets.values():
            while len(buckets) > 0:
                event, param_group = buckets.pop()
                event.wait()
                param_group.release_grad_buffer()
        torch.cuda.current_stream().wait_stream(stream)
        root_state._post_backward_callback_queued = False
        ctx.backward_phase = False
        ctx.backward_module = None
        ctx.backward_done_modules.clear()

        # After the first backward pass, we have the full trace of bucket allocations
        # and releases. We can now plan the memory pool based on this trace.
        if isinstance(ctx.weight_bucket_allocator, TracePoolAllocator):
            wbuf_alloc = ctx.weight_bucket_allocator
            if wbuf_alloc.phase == "trace":
                if torch.distributed.get_rank() == 0:
                    logger.debug(wbuf_alloc.dump_trace())
                wbuf_alloc.plan()
            elif wbuf_alloc.phase == "optimized":
                wbuf_alloc.reset_cursor()
            else:
                raise ValueError(f"Unexpected weight bucket allocator phase: {wbuf_alloc.phase}")
        if isinstance(ctx.grad_bucket_allocator, TracePoolAllocator):
            gbuf_alloc = ctx.grad_bucket_allocator
            if gbuf_alloc.phase == "trace":
                if torch.distributed.get_rank() == 0:
                    logger.debug(gbuf_alloc.dump_trace())
                gbuf_alloc.plan()
            elif gbuf_alloc.phase == "optimized":
                gbuf_alloc.reset_cursor()
            else:
                raise ValueError(f"Unexpected grad bucket allocator phase: {gbuf_alloc.phase}")

    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
    )
