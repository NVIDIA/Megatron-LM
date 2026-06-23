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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .allocator import TracePoolAllocator
from .cuda_graph_runner import FSDPCudaGraphRunner
from .fsdp_module import FSDPModule, _FSDPState
from .utils import RegisterFSDPBackwardFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _find_fsdp_target(hook_module: nn.Module) -> Optional[FSDPModule]:
    """Return the nearest parent FSDPModule for *hook_module*.

    Used by fine-grained hooks registered on sub-modules to resolve the
    FSDPModule that owns the sub-module.  The reference is stored as
    ``_fsdp_parent_module`` during FSDP init (a ``weakref.ref`` to avoid
    reference cycles).

    Returns:
        The owning FSDPModule, or ``None`` if the module has no FSDP parent.
    """
    if isinstance(hook_module, FSDPModule):
        return hook_module
    parent_ref = getattr(hook_module, '_fsdp_parent_module', None)
    if parent_ref is not None:
        return parent_ref()
    return None


def mfsdp_forward_pre_hook(hook_module: nn.Module, args: Any, kwargs: Any):
    """Pre-forward hook for FSDP modules and fine-grained sub-modules.

    Resolves the target FSDPModule via :func:`_find_fsdp_target`, performs
    parameter unshard, root-phase bookkeeping, and (for direct FSDPModule
    calls only) CUDA graph capture.

    **Repeatability**: This function MUST be safe to call multiple times per
    module without observable overhead.  Fine-grained hook registration
    (``_register_forward_pre_hook(fine_grained=True)``) installs the hook on
    every sub-module of an FSDPModule.  When a sub-module's ``forward()`` is
    called, PyTorch triggers the pre-forward hook, which calls this function.
    If the enclosing FSDPModule is also directly invoked (and its own pre-forward
    hook fires), this function will be invoked again for the same target.
    The implementation must handle this gracefully — duplicating a no-op
    ``unshard()`` call or re-applying idempotent bookkeeping must not introduce
    measurable latency.
    """
    target = _find_fsdp_target(hook_module)
    if target is None:
        return

    ctx = target._fsdp_root_context
    assert not ctx.cuda_graph_active, (
        "hooks must not fire during CUDA graph capture"
    )

    # ---- root: forward-phase setup (once per micro-batch) ------------------
    if target._fsdp_state._is_root:
        if ctx.enable_cuda_graph and ctx.cuda_graph_stream is None:
            ctx.cuda_graph_stream = torch.cuda.Stream()
            torch.cuda.set_stream(ctx.cuda_graph_stream)
            ctx.cuda_graph_pool = torch.cuda.graph_pool_handle()
        ctx.forward_phase = True
        ctx.backward_phase = False

    # ---- unshard parameters for this module -------------------------------
    if ctx.backward_phase:
        target.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)
    target.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

    # ---- free stale grad data (safe to repeat, idempotent) ----------------
    for param_group in target._fsdp_param_groups:
        param_group._maybe_free_grad_data()

    # ---- CUDA graph capture (FSDPModule targets only) ---------------------
    # Fine-grained hooks fire on sub-modules whose forward args differ from
    # the FSDPModule's; CG capture is not meaningful there.
    if (
        isinstance(hook_module, FSDPModule)
        and target._fsdp_state.enable_cuda_graph
        and (not hasattr(target, "_fsdp_cg_runner"))
        and not ctx.backward_phase
        and target.cuda_graph_compatible
    ):
        if torch.distributed.get_rank() == 0:
            logger.debug(
                "Capturing CUDA graph for module %s (id=%s)",
                target._fsdp_module_name,
                id(target),
            )
        cg_runner = FSDPCudaGraphRunner(
            target, graph_pool=ctx.cuda_graph_pool
        )
        cg_runner.capture_forward(*args, **kwargs)
        cg_runner.install()
        target._fsdp_cg_runner = cg_runner
        if torch.distributed.get_rank() == 0:
            logger.debug(
                "Captured CUDA graph for module %s (id=%s)",
                target._fsdp_module_name,
                id(target),
            )


def mfsdp_post_forward_hook(module: nn.Module, *unused):
    """Post-forward hook: reshard parameters.

    Only supports direct FSDPModule calls.  Raises ``TypeError`` when
    called with a non-FSDPModule (fine-grained path is not yet handled).
    """
    if not isinstance(module, FSDPModule):
        raise TypeError(
            "mfsdp_post_forward_hook only supports FSDPModule, "
            f"got {type(module).__name__}"
        )
    ctx = module._fsdp_root_context
    assert not ctx.cuda_graph_active, (
        "hooks must not fire during CUDA graph capture"
    )
    if ctx.backward_phase and id(module) == ctx.backward_module:
        return
    module.reshard()


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def _register_forward_pre_hook(
    module: FSDPModule, fine_grained: bool = False
) -> None:
    """Register a pre-forward hook on the FSDP module or its sub-modules.

    Args:
        fsdp_module: The FSDP module to instrument.
        fine_grained: If ``True``, register on every sub-module of
            *fsdp_module* (for EP-overlap / 1F1B schedules).
            ``_fsdp_parent_module`` must already be set on sub-modules
            (done by :meth:`FSDPModule._init_fsdp_state`).
    """
    if fine_grained:
        for submodule in module.modules():
            fsdp_module = _find_fsdp_target(submodule)
            if fsdp_module is None or fsdp_module is not module:
                continue
            submodule.register_forward_pre_hook(
                mfsdp_forward_pre_hook, prepend=True, with_kwargs=True,
            )
    else:
        module.register_forward_pre_hook(
            mfsdp_forward_pre_hook, prepend=True, with_kwargs=True
        )


def _register_forward_hook(module: FSDPModule):
    """Register post-forward hook to reshard parameters."""
    module._mfsdp_forward_hook = module.register_forward_hook(mfsdp_post_forward_hook)


# ---------------------------------------------------------------------------
# Internal: backward hook helpers
# ---------------------------------------------------------------------------


def mfsdp_pre_backward_setup(
    hook_module: nn.Module, grads: Any = None, skip_final_callback: bool = False
):
    """Pre-backward hook for FSDP modules and fine-grained sub-modules.

    Resolves the target FSDPModule via :func:`_find_fsdp_target`, performs
    backward-phase root setup, parameter unshard, and TE gradient-fusion
    bookkeeping.  The ``_fsdp_pre_backward_done`` flag prevents redundant
    calls when multiple sub-modules share the same parent.

    Compatible with ``register_multi_grad_hook`` callback signature
    (module, grads).

    Args:
        hook_module: Module whose backward pass is about to start.
        grads: Gradients from ``register_multi_grad_hook`` (unused).
        skip_final_callback: If ``True``, do **not** auto-enqueue
            ``mfsdp_post_backward_final_callback``.  The caller is
            responsible for calling it manually (used by the 1F1B EP
            overlap schedule).
    """
    target = _find_fsdp_target(hook_module)
    if target is None:
        return
    if target._fsdp_pre_backward_done:
        return

    _pre_backward_setup(target, skip_final_callback=skip_final_callback)
    target._fsdp_pre_backward_done = True


def mfsdp_post_backward_hook(module: nn.Module):
    """Post-backward hook: reshard parameters and reduce gradients.

    Only supports direct FSDPModule calls.  Raises ``TypeError`` when
    called with a non-FSDPModule (fine-grained path is not yet handled).
    """
    if not isinstance(module, FSDPModule):
        raise TypeError(
            "mfsdp_post_backward_hook only supports FSDPModule, "
            f"got {type(module).__name__}"
        )
    ctx = module._fsdp_root_context
    assert not ctx.cuda_graph_active, (
        "hooks must not fire during CUDA graph capture"
    )

    for submodule in module._get_fsdp_modules(recursive=True):
        if submodule.post_backward_issued:
            continue
        ctx.backward_done_modules.add(id(submodule))
        submodule.reshard()
        if any(
            param_group.sharding_strategy in ("optim_grads", "optim_grads_params")
            for param_group in submodule._fsdp_param_groups
        ):
            submodule.reduce_grad(async_op=ctx.enable_async_reduce_grad)
        submodule.post_backward_issued = True
    ctx._advance_backward_module()


def mfsdp_post_backward_final_callback(root_module: nn.Module):
    """Finalise the backward pass: drain skipped modules, reset state,
    clear fine-grained flags, and (on the first micro-batch) transition
    the bucket allocator from trace to optimized plan.

    Only supports the root FSDP module.  Raises ``TypeError`` if
    *root_module* is not an FSDPModule, or ``RuntimeError`` if it is
    not marked as root.
    """
    if not isinstance(root_module, FSDPModule):
        raise TypeError(
            "mfsdp_post_backward_final_callback only supports FSDPModule, "
            f"got {type(root_module).__name__}"
        )
    if not root_module._fsdp_state._is_root:
        raise RuntimeError(
            "mfsdp_post_backward_final_callback requires root FSDP module"
        )

    ctx = root_module._fsdp_root_context
    assert not ctx.cuda_graph_active, (
        "hooks must not fire during CUDA graph capture"
    )

    # ---- handle modules whose per-module post-backward was skipped ----
    for module in reversed(ctx.forward_order):
        if module.post_backward_issued:
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
    root_module._fsdp_state._post_backward_callback_queued = False
    ctx.backward_phase = False
    ctx.backward_module = None
    ctx.backward_done_modules.clear()

    # ---- clear fine-grained pre-backward flags -------------------------
    for module in ctx.forward_order:
        module._fsdp_pre_backward_done = False

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


# ---------------------------------------------------------------------------
# Internal: backward hook helpers
# ---------------------------------------------------------------------------


def _create_custom_backward_hook(
    module: nn.Module,
    custom_backward_handler: Callable,
    ctx_module: Optional[nn.Module] = None,
):
    """Wrap *module* so that ``custom_backward_handler`` fires as a
    pre-backward hook via ``register_multi_grad_hook``.

    Args:
        module: Module whose output tensors are instrumented.
        custom_backward_handler: Callback invoked when backward reaches
            this module.
        ctx_module: Module whose ``_fsdp_root_context`` is checked for
            CUDA-graph safety.  Defaults to *module*.
    """
    _ctx_source = ctx_module if ctx_module is not None else module

    @torch.compiler.disable
    def forward_hook(_module, inputs, output):
        if hasattr(_ctx_source, '_fsdp_root_context'):
            assert not _ctx_source._fsdp_root_context.cuda_graph_active, (
                "hooks must not fire during CUDA graph capture"
            )
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


def _pre_backward_setup(
    module: FSDPModule, skip_final_callback: bool = False
):
    """Shared pre-backward logic: root setup, unshard, TE flags.

    Used by both the normal and fine-grained backward pre-hook paths.

    Args:
        module: The FSDPModule whose backward is starting.
        skip_final_callback: If ``True``, do not enqueue the post-backward
            final callback.  The caller must call
            ``mfsdp_post_backward_final_callback`` manually later.

    .. note::

       When CUDA graph is enabled, TE wgrad fusion writes directly into
       ``param.main_grad`` during the trace (eager) backward.  Under graph
       replay only the GPU kernel runs — the Python-side flags that mark
       ``grad_added_to_main_grad`` are not part of the graph.  We eagerly
       allocate the main gradient buffer and its full unsharded fetch-buffer
       here so that memory addresses are fixed across graph replay iterations.
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
        if not skip_final_callback and not module._fsdp_state._post_backward_callback_queued:
            _register_post_backward_final_callback(module._fsdp_state, module)

    # ---- unshard params for backward compute --------------------------
    module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)

    # ---- reset per-module bookkeeping ---------------------------------
    module.post_backward_issued = False

    # ---- Transformer Engine gradient-accumulation fusion ---------------
    for param_group in module._fsdp_param_groups:
        for param in param_group.params:
            param.grad_added_to_main_grad = False
            if param_group.sharding_strategy in (
                "optim_grads_params",
                "optim_grads",
            ):
                param.overwrite_main_grad = True
        # CUDA graph + TE wgrad fusion: during graph capture the eager backward
        # runs once and TE sets grad_added_to_main_grad=True on each param it
        # writes to.  Under replay only the GPU kernel runs — the Python-side
        # setattr is not part of the graph.  We must allocate the main gradient
        # buffer and its full unsharded fetch-buffer BEFORE capture so that
        # memory addresses are fixed across replay iterations.  Without this,
        # TE would write to stale or uninitialised buffer addresses on replay.
        if module._fsdp_state.enable_cuda_graph and param_group.main_grad_buffer is not None:
            param_group._init_dist_grads()
            param_group.main_grad_buffer.fetch_buffer()

    return ctx


# ---------------------------------------------------------------------------
# Backward hook registration
# ---------------------------------------------------------------------------


def _register_backward_pre_hook(
    module: FSDPModule,
    fine_grained: bool = False,
    skip_final_callback: bool = False,
):
    """Register backward pre-hook using multi-grad hooks on output tensors.

    Attaches a ``register_multi_grad_hook`` to every tensor output of
    ``module.forward()``.  When autograd reaches this module during the
    backward pass, the hook fires *before* the module's own backward,
    giving FSDP a chance to unshard parameters for gradient computation.
    """
    if fine_grained:
        for submodule in module.modules():
            fsdp_module = _find_fsdp_target(submodule)
            if fsdp_module is None or fsdp_module is not module:
                continue
            submodule._mfsdp_backward_pre_hook = _create_custom_backward_hook(
                submodule,
                custom_backward_handler=lambda m, g: mfsdp_pre_backward_setup(
                    m, g, skip_final_callback=skip_final_callback
                ),
                ctx_module=module,
            )
        return

    module._mfsdp_backward_pre_hook = _create_custom_backward_hook(
        module, custom_backward_handler=lambda m, g: mfsdp_pre_backward_setup(
            m, g, skip_final_callback=skip_final_callback
        ),
    )


def _register_backward_hook(module: FSDPModule):
    """
    Register backward hook using autograd Function.

    This inserts a RegisterFSDPBackwardFunction in the backward pass
    that triggers ``mfsdp_post_backward_hook`` after gradients are
    computed — resharding parameters and reducing gradients.
    """
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
        functools.partial(_register_post_backward_hook, mfsdp_post_backward_hook),
        with_kwargs=True,
    )


# ---------------------------------------------------------------------------
# Post-backward final callback
# ---------------------------------------------------------------------------


def _register_post_backward_final_callback(
    state: _FSDPState, module: nn.Module
) -> None:
    """
    Enqueue a *single* engine callback that fires after every module's
    backward pass has completed.

    Registered once by the root FSDP module (avoids duplicates).
    Delegates to :func:`mfsdp_post_backward_final_callback`.
    """
    assert state._is_root, "Only root FSDP should register post-backward callback"
    if state._post_backward_callback_queued:
        return

    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(mfsdp_post_backward_final_callback, module)
    )
