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
from .cuda_graph_runner import FSDPCudaGraphRunner

logger = logging.getLogger(__name__)


def _register_forward_pre_hook(fsdp_module: FSDPModule, fine_grained: bool = False):
    """Register a pre-forward hook on *hook_module*.

    Called before every ``forward()`` of the FSDP module — both in the
    forward and backward (activation recomputation) passes.  Handles
    root-level phase bookkeeping, parameter unshard, and CUDA graph
    capture (once per compatible module).
    """

    def forward_pre_hook(hook_module, args, kwargs):
        ctx = fsdp_module._fsdp_root_context
        assert not ctx.cuda_graph_active, (
            "hooks must not fire during CUDA graph capture"
        )

        # ---- root: forward-phase setup (once per micro-batch) --------------
        if hook_module is fsdp_module and fsdp_module._fsdp_state._is_root:
            if ctx.enable_cuda_graph and ctx.cuda_graph_stream is None:
                ctx.cuda_graph_stream = torch.cuda.Stream()
                torch.cuda.set_stream(ctx.cuda_graph_stream)
                ctx.cuda_graph_pool = torch.cuda.graph_pool_handle()
            ctx.forward_phase = True
            ctx.backward_phase = False

        # ---- unshard parameters for this module ---------------------------
        if ctx.backward_phase:
            fsdp_module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)
        fsdp_module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

        for param_group in fsdp_module._fsdp_param_groups:
            param_group._maybe_free_grad_data()

        # ---- CUDA graph capture (once per compatible module) --------------
        if (
            hook_module is fsdp_module
            and fsdp_module._fsdp_state.enable_cuda_graph
            and (not hasattr(fsdp_module, "_fsdp_cg_runner"))
            and not ctx.backward_phase
            and fsdp_module.cuda_graph_compatible
        ):
            if torch.distributed.get_rank() == 0:
                logger.debug(
                    "Capturing CUDA graph for module %s (id=%s)",
                    fsdp_module._fsdp_module_name,
                    id(fsdp_module),
                )
            cg_runner = FSDPCudaGraphRunner(
                fsdp_module, graph_pool=ctx.cuda_graph_pool
            )
            cg_runner.capture_forward(*args, **kwargs)
            cg_runner.install()
            fsdp_module._fsdp_cg_runner = cg_runner
            if torch.distributed.get_rank() == 0:
                logger.debug(
                    "Captured CUDA graph for module %s (id=%s)",
                    fsdp_module._fsdp_module_name,
                    id(fsdp_module),
                )

    if fine_grained:
        for submodule in fsdp_module.modules():
            submodule.register_forward_pre_hook(
                forward_pre_hook, prepend=True, with_kwargs=True,
            )
    else:
        fsdp_module.register_forward_pre_hook(
            forward_pre_hook, prepend=True, with_kwargs=True
        )


def _register_forward_hook(module: FSDPModule):
    """Register post-forward hook to reshard parameters."""

    def reshard_param_groups(module, *unused):
        ctx = module._fsdp_root_context
        assert not ctx.cuda_graph_active, (
            "hooks must not fire during CUDA graph capture"
        )
        if ctx.backward_phase and id(module) == ctx.backward_module:
            return
        module.reshard()

    module._mfsdp_forward_hook = module.register_forward_hook(reshard_param_groups)


def _register_backward_pre_hook(module: FSDPModule):
    """
    Register backward pre-hook using multi-grad hooks on output tensors.

    Attaches a ``register_multi_grad_hook`` to every tensor output of
    ``module.forward()``.  When autograd reaches this module during the
    backward pass, the hook fires *before* the module's own backward,
    giving FSDP a chance to unshard parameters for gradient computation.
    """

    def create_custom_backward_hook(
        module: FSDPModule, custom_backward_handler: Callable
    ):
        """Wrap *module* so that ``custom_backward_handler`` fires as a
        pre-backward hook via ``register_multi_grad_hook``."""

        @torch.compiler.disable
        def forward_hook(_module, inputs, output):
            assert not _module._fsdp_root_context.cuda_graph_active, (
                "hooks must not fire during CUDA graph capture"
            )
            # ``view_as`` ensures the autograd graph sees a distinct tensor
            # object even when the module returns a view of an input.
            output = tree_map(
                lambda t: t.view_as(t) if torch.is_tensor(t) else t, output
            )

            output_list = []
            if isinstance(output, torch.Tensor):
                output_list = [output]
            elif isinstance(output, (tuple, list)):
                output_list = [t for t in output if isinstance(t, torch.Tensor)]

            torch.autograd.graph.register_multi_grad_hook(
                output_list,
                lambda grads: custom_backward_handler(_module, grads),
                mode="any",
            )
            return output

        return module.register_forward_hook(forward_hook)

    def pre_backward_hook(module: FSDPModule, grads):
        """Pre-backward callback for a single FSDP module.

        Invoked by ``register_multi_grad_hook`` when autograd reaches
        this module during the backward pass — *before* the module's
        own ``backward()`` runs.  Executed in reverse forward order.

        Execution flow
        --------------
        1. **Assert** not inside CUDA graph capture.
        2. **Root setup** (first module in backward):
           clear backward tracking, switch phase flags, advance the
           backward module cursor, and enqueue the post-backward final
           callback (once).
        3. **Unshard** parameters for this module (bwd pass).
        4. **Reset per-module state**: mark ``post_backward_issued``
           as not yet handled, and reset Transformer Engine gradient
           accumulation flags.
        5. **TE wgrad fusion** (``optim_grads`` / ``optim_grads_params``):
           set ``overwrite_main_grad`` so TE writes weight gradients
           directly into ``param.main_grad`` (overwrite, not accumulate)
           to avoid double-counting across micro-batches.
        6. **Unshard main_grad_buffer** if the parameter group owns one.
        """
        ctx = module._fsdp_root_context
        assert not ctx.cuda_graph_active, (
            "hooks must not fire during CUDA graph capture"
        )

        # ---- root: backward-phase setup -----------------------------------
        if module._fsdp_state._is_root:
            ctx.backward_done_modules.clear()
            ctx.forward_phase = False
            ctx.backward_phase = True
            ctx._advance_backward_module()
            if not module._fsdp_state._post_backward_callback_queued:
                _register_post_backward_final_callback(module._fsdp_state, module)

        # ---- unshard params for backward compute --------------------------
        module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)

        # ---- reset per-module bookkeeping ---------------------------------
        # ``post_backward_issued`` guards against the post-backward path
        # (reshard + reduce_grad) being skipped.
        setattr(module, "post_backward_issued", False)

        # ---- Transformer Engine gradient-accumulation fusion ---------------
        #
        # In the eager (trace) backward, TE sets
        # ``grad_added_to_main_grad = True`` on each param whose weight
        # gradient it writes directly to ``main_grad``.  Under CUDA graph
        # replay the GPU kernel still runs, but the Python-side
        # ``setattr`` is not part of the graph.  We recorded the per-param
        # flag after the trace backward (see ``_post_backward_final_callback``)
        # and restore it here so that ``reduce_grad`` knows TE already
        # populated ``main_grad``.
        for param_group in module._fsdp_param_groups:
            for param in param_group.params:
                setattr(param, "grad_added_to_main_grad", False)
                if param_group.sharding_strategy in (
                    "optim_grads_params",
                    "optim_grads",
                ):
                    # TE's backward kernel writes weight gradients directly
                    # into param.main_grad.  By default TE *accumulates*
                    # (adds) into main_grad, which silently doubles
                    # gradients when the buffer isn't zeroed between
                    # micro-batches.  ``overwrite_main_grad`` tells TE to
                    # overwrite instead.
                    setattr(param, "overwrite_main_grad", True)
            if param_group.main_grad_buffer is not None:
                param_group._init_dist_grads()
                param_group.main_grad_buffer.fetch_buffer()

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
        assert not ctx.cuda_graph_active, (
            "hooks must not fire during CUDA graph capture"
        )
        ctx.backward_done_modules.add(id(module))
        ctx._advance_backward_module()
        module.reshard()
        if any(
            param_group.sharding_strategy in ("optim_grads", "optim_grads_params")
            for param_group in module._fsdp_param_groups
        ):
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
        assert not module._fsdp_root_context.cuda_graph_active, (
            "hooks must not fire during CUDA graph capture"
        )
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


def _register_post_backward_final_callback(
    state: _FSDPState, module: nn.Module
) -> None:
    """
    Enqueue a *single* engine callback that fires after every module's
    backward pass has completed.

    Registered once by the root FSDP module (avoids duplicates).
    The callback:
    - Reshards and reduces gradients for any module whose per-module
      post-backward hook was silently skipped (e.g. activation
      recomputation).
    - Waits for all async reduce-grad operations to finish.
    - Resets root / context state for the next micro-batch.
    - On the first call (trace → optimized transition): builds the
      pool plan and records TE wgrad-fusion flags for CUDA graph restore.
    """
    assert state._is_root, "Only root FSDP should register post-backward callback"
    if state._post_backward_callback_queued:
        return

    def _post_backward_final_callback(root_state: _FSDPState, root_module: nn.Module):
        """Engine callback — the last thing autograd runs after the backward
        pass of every micro-batch."""
        ctx = root_module._fsdp_root_context
        assert not ctx.cuda_graph_active, (
            "hooks must not fire during CUDA graph capture"
        )

        # ---- handle modules whose per-module post-backward was skipped ----
        for module in reversed(ctx.forward_order):
            if getattr(module, "post_backward_issued", False):
                continue
            module.reshard()
            if any(
                param_group.sharding_strategy in ("optim_grads", "optim_grads_params")
                for param_group in module._fsdp_param_groups
            ):
                module.reduce_grad(async_op=ctx.enable_async_reduce_grad)

        # ---- drain pending async reduce-grad events -----------------------
        stream = ctx.rs_stream
        for buckets in ctx.reduce_grad_buckets.values():
            while len(buckets) > 0:
                event, param_group = buckets.pop()
                event.wait()
                param_group.release_grad_buffer()
        torch.cuda.current_stream().wait_stream(stream)

        # ---- reset root / context state for the next micro-batch ----------
        root_state._post_backward_callback_queued = False
        ctx.backward_phase = False
        ctx.backward_module = None
        ctx.backward_done_modules.clear()

        # ---- trace → optimized transition (first micro-batch only) --------
        if isinstance(ctx.bucket_allocator, TracePoolAllocator):
            bucket_alloc = ctx.bucket_allocator
            if bucket_alloc.phase == "trace":
                if torch.distributed.get_rank() == 0:
                    logger.debug(bucket_alloc.dump_trace())
                    for m in ctx.forward_order:
                        logger.debug(
                            f"module_id={id(m)}, module_name={m._fsdp_module_name}"
                        )
                bucket_alloc.plan()
            elif bucket_alloc.phase != "optimized":
                raise ValueError(
                    f"Unexpected bucket allocator phase: {bucket_alloc.phase}"
                )

    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
    )
