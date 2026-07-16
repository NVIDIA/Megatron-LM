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

"""CUDA graph capture / replay for individual FSDP v2 modules.

Built on ``te_graph_runtime.make_graphed_callables`` which supports
``capture_time_hooks`` — hooks that run outside CUDA graph capture (for
FSDP unshard / reshard) and are not replayed.  ``sample_kwargs`` is used
so modules receive keyword arguments natively.

A single ``CudaGraphRunner`` instance is stored on the root context and
orchestrates:

  1. Recording sample args for each eligible FSDP module during the
     first optimized forward pass.
  2. Calling ``make_graphed_callables`` with all modules and
     ``capture_time_hooks`` that perform unshard / reshard.
"""  # noqa: E501

import inspect
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten

logger = logging.getLogger(__name__)


def _tensor_storage_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identify a tensor storage view.

    :param tensor: Tensor to identify.
    :type tensor: torch.Tensor
    :return: Storage address and view metadata.
    :rtype: Tuple[Any, ...]
    """
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


# ---------------------------------------------------------------------------
# NVML memory helper (real GPU memory, not just torch allocator view)
# ---------------------------------------------------------------------------


def _nvml_device_memory(device: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return (used_MiB, total_MiB) from NVML, or None if unavailable."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        if device is None:
            device = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (info.used // (1024 * 1024), info.total // (1024 * 1024))
    except Exception:
        return None


def _mem_snapshot() -> Dict[str, int]:
    """Capture a snapshot of memory counters across torch and NVML."""
    snap = {
        "torch_alloc": torch.cuda.memory_allocated() // 1_000_000,
        "torch_reserved": torch.cuda.memory_reserved() // 1_000_000,
    }
    nvml = _nvml_device_memory()
    if nvml is not None:
        snap["nvml_used"] = nvml[0]
        snap["nvml_total"] = nvml[1]
    return snap


def _fmt_mem_snapshot(before: Dict[str, int], after: Dict[str, int], peak_alloc: int) -> str:
    """Format memory diff as a human-readable string."""
    parts = [
        f"torch_alloc {before['torch_alloc']}→{after['torch_alloc']} MB "
        f"(Δ{after['torch_alloc'] - before['torch_alloc']:+d})",
        f"torch_reserved {before['torch_reserved']}→{after['torch_reserved']} MB "
        f"(Δ{after['torch_reserved'] - before['torch_reserved']:+d})",
        f"peak_alloc {peak_alloc // 1_000_000} MB",
    ]
    if "nvml_used" in before:
        parts.append(
            f"nvml_used {before['nvml_used']}→{after['nvml_used']} MB "
            f"(Δ{after['nvml_used'] - before['nvml_used']:+d})"
        )
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Hook save / restore
# ---------------------------------------------------------------------------

_HOOK_ATTRS = [
    "_forward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]


def _pop_all_hooks(module):
    saved = []
    for sub in module.modules():
        snap = {}
        for attr in _HOOK_ATTRS:
            if hasattr(sub, attr):
                snap[attr] = getattr(sub, attr)
                setattr(sub, attr, OrderedDict())
        saved.append((sub, snap))
    return saved


def _restore_all_hooks(saved):
    for sub, snap in saved:
        for name, value in snap.items():
            if value is not None:
                setattr(sub, name, value)


def _prepare_compiled_modules_for_capture(modules):
    """Convert ``Module.compile()`` modules to compiled forward bodies.

    ``nn.Module.compile()`` compiles ``Module._call_impl``, which includes
    module-hook dispatch.  FSDP removes those hooks and replaces them with
    ``capture_time_hooks`` while building its explicit CUDA graphs.  Keeping
    the compiled ``_call_impl`` can therefore trigger a guard failure and a
    lazy recompile inside CUDA stream capture.

    Compile the forward body instead, with Inductor CUDA graphs disabled so
    that the FSDP runner remains the sole CUDA-graph owner.  The returned state
    is only for rollback if explicit graph capture fails; after successful
    installation, the stale compiled ``_call_impl`` must remain disabled.
    """
    saved = []
    try:
        for module in modules:
            compiled_call_impl = getattr(module, "_compiled_call_impl", None)
            if compiled_call_impl is None:
                continue

            original_forward = module.forward
            saved.append((module, original_forward, compiled_call_impl))
            module._compiled_call_impl = None

            # Avoid wrapping a forward body that the user already compiled
            # directly.  This branch mainly handles ``module.compile()``.
            if not hasattr(original_forward, "_torchdynamo_orig_callable"):
                module.forward = torch.compile(
                    original_forward, dynamic=False, options={"triton.cudagraphs": False}
                )
    except Exception:
        _restore_compiled_modules_after_capture_failure(saved)
        raise
    return saved


def _restore_compiled_modules_after_capture_failure(saved):
    """Restore module-level compilation when explicit capture fails."""
    for module, original_forward, compiled_call_impl in saved:
        module.forward = original_forward
        module._compiled_call_impl = compiled_call_impl


class CudaGraphRunner:
    """Orchestrates per-module sample-arg recording and batch graph capture.

    Created once by the root forward pre-hook and stored on
    ``ctx.cuda_graph_runner``.
    """

    def __init__(self, graph_pool: Any, num_warmup_iters: int = 3):
        self._graph_pool = graph_pool
        self._num_warmup = num_warmup_iters
        self._captured = False

        # Per-module state recorded during the first optimized forward.
        self._sample_args: Dict[int, Tuple] = {}
        self._sample_kwargs: Dict[int, Dict[str, Any]] = {}
        self._sample_outputs: Dict[int, Any] = {}
        self._modules_ordered: List[torch.nn.Module] = []
        self._compiled_module_state = []

    # ---- called from hooks ------------------------------------------------

    def record_module(self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]) -> None:
        """Record sample args for *module* during the first optimized forward."""
        if self._captured:
            return
        mid = id(module)
        if mid in self._sample_args:
            return

        # Normalize Module.compile() before capture setup. te-graph-runtime
        # detects this compiled forward body and warms the capture-equivalent
        # hook specialization before entering torch.cuda.graph.
        self._compiled_module_state.extend(_prepare_compiled_modules_for_capture([module]))

        sig = inspect.signature(module.forward)
        has_self = "self" in sig.parameters
        bound = sig.bind(module, *args, **kwargs) if has_self else sig.bind(*args, **kwargs)
        all_kwargs = {
            n: bound.arguments[n] for n in bound.arguments if not (has_self and n == "self")
        }
        self._sample_args[mid] = tuple()  # all via kwargs
        self._sample_kwargs[mid] = all_kwargs
        self._modules_ordered.append(module)

        n_tensor = sum(1 for v in all_kwargs.values() if isinstance(v, torch.Tensor))
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: recorded module %s (id=%s), " "%d kwargs (%d tensor)",
                getattr(module, "_fsdp_module_name", module.__class__.__name__),
                id(module),
                len(all_kwargs),
                n_tensor,
            )

    def record_module_output(self, module: torch.nn.Module, output: Any) -> None:
        """Record an eager output for static graph linking.

        :param module: Recorded FSDP module.
        :type module: torch.nn.Module
        :param output: Output from the eager sample forward.
        :type output: Any
        """
        mid = id(module)
        if self._captured or mid not in self._sample_args or mid in self._sample_outputs:
            return
        self._sample_outputs[mid] = output

    def capture_and_install(
        self, root_module: torch.nn.Module, capture_stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """Capture all graphs + install wrappers on recorded modules."""
        if self._captured or not self._modules_ordered:
            return
        self._captured = True

        modules = self._modules_ordered
        n = len(modules)
        saved_parameter_grads = tuple(
            (param, param.grad)
            for module in modules
            for param_group in module._fsdp_param_groups
            for param in param_group.params
        )

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: capturing %d modules", n)

        # Use the installed runtime only when it supports M-FSDP capture.
        try:
            from te_graph_runtime import make_graphed_callables
            from te_graph_runtime.graph import (
                _get_compatible_main_grad_buffer as _installed_static_grad_support,
            )
            from te_graph_runtime.graph import (
                _refresh_module_parameter_surface as _installed_parameter_refresh,
            )

            if not all(
                callable(helper)
                for helper in (_installed_static_grad_support, _installed_parameter_refresh)
            ):
                raise ImportError("Installed te-graph-runtime lacks M-FSDP CUDA graph support")
        except ImportError:
            from .te_graph_runtime import make_graphed_callables

        sample_args_list: List[Tuple] = []
        sample_kwargs_list: List[Dict[str, Any]] = []
        capture_hooks: List[Dict] = []

        producer_outputs: Dict[Tuple[Any, ...], Tuple[int, int]] = {}
        input_output_aliases: List[Dict[int, Tuple[int, int]]] = []
        for producer_idx, module in enumerate(modules):
            flat_outputs, _ = tree_flatten(self._sample_outputs.get(id(module), ()))
            for output_idx, output in enumerate(flat_outputs):
                if isinstance(output, torch.Tensor):
                    producer_outputs[_tensor_storage_key(output)] = (producer_idx, output_idx)

        for consumer_idx, module in enumerate(modules):
            mid = id(module)
            flat_args, _ = tree_flatten(self._sample_args[mid])
            flat_kwargs, _ = tree_flatten(list(self._sample_kwargs[mid].values()))
            aliases = {}
            for input_idx, input_tensor in enumerate(flat_args + flat_kwargs):
                if not isinstance(input_tensor, torch.Tensor):
                    continue
                producer = producer_outputs.get(_tensor_storage_key(input_tensor))
                if producer is not None and producer[0] < consumer_idx:
                    aliases[input_idx] = producer
            input_output_aliases.append(aliases)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: linked %d static input/output tensors",
                sum(len(aliases) for aliases in input_output_aliases),
            )

        for m in modules:
            mid = id(m)
            # Clone tensor values so warmup gets fresh leaves without
            # residual autograd state from the first forward+backward.
            args = tuple(
                (
                    v.detach().clone().requires_grad_(v.requires_grad)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for v in self._sample_args[mid]
            )
            kw = {
                k: (
                    v.detach().clone().requires_grad_(v.requires_grad)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in self._sample_kwargs[mid].items()
            }
            sample_args_list.append(args)
            sample_kwargs_list.append(kw)

            capture_hooks.append(
                {
                    "forward_pre_hooks": {0: _make_fwd_pre_hook(m)},
                    "forward_pre_hooks_with_kwargs": {0: True},
                    "forward_hooks": {0: _make_fwd_post_hook(m)},
                    "forward_hooks_with_kwargs": {0: True},
                    "backward_pre_hooks": {0: _make_bwd_pre_hook(m)},
                    "backward_hooks": {0: _make_bwd_post_hook(m)},
                }
            )

        self._sample_args.clear()
        self._sample_kwargs.clear()
        self._sample_outputs.clear()

        compiled_module_state = self._compiled_module_state
        if compiled_module_state and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            logger.info(
                "CudaGraphRunner: converted %d Module.compile() wrappers to "
                "compiled forward bodies",
                len(compiled_module_state),
            )

        # Pop real FSDP hooks so make_graphed_callables passes its assertion.
        # capture_time_hooks handle unshard/reshard during warmup + capture.
        saved_hooks = _pop_all_hooks(root_module)

        runtime_options = {}
        supports_input_output_aliases = (
            "_input_output_aliases" in inspect.signature(make_graphed_callables).parameters
        )
        if any(input_output_aliases):
            if not supports_input_output_aliases:
                from .te_graph_runtime import make_graphed_callables

                supports_input_output_aliases = True
            runtime_options["_input_output_aliases"] = tuple(input_output_aliases)

        try:
            torch.cuda.reset_peak_memory_stats()
            _mem_before = _mem_snapshot()

            graphed = make_graphed_callables(
                tuple(modules),
                sample_args_list,
                num_warmup_iters=self._num_warmup,
                sample_kwargs=sample_kwargs_list,
                pool=self._graph_pool,
                capture_time_hooks=capture_hooks,
                capture_stream=capture_stream,
                **runtime_options,
            )
        except Exception:
            _restore_compiled_modules_after_capture_failure(compiled_module_state)
            raise
        finally:
            _restore_all_hooks(saved_hooks)
            for param, grad in saved_parameter_grads:
                param.grad = grad

        _mem_after = _mem_snapshot()
        _peak_alloc = torch.cuda.max_memory_allocated()

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: %d modules captured  %s",
                n,
                _fmt_mem_snapshot(_mem_before, _mem_after, _peak_alloc),
            )

        if not isinstance(graphed, tuple):
            graphed = (graphed,)

        # make_graphed_callables already replaced module.forward with
        # the graphed version that handles kwargs natively.
        for module in modules:
            module._fsdp_cg_installed = True
        self._compiled_module_state = []

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: installed CUDA graphs on %d modules", n)


# ---------------------------------------------------------------------------
# capture_time_hooks (unshard / reshard outside graph, not replayed)
# ---------------------------------------------------------------------------


def _make_fwd_pre_hook(module):
    def hook(mod, args, kwargs):
        module.unshard()

    return hook


def _make_fwd_post_hook(module):
    def hook(mod, args, kwargs, output):
        module.reshard()

    return hook


def _make_bwd_pre_hook(module):
    def hook(mod, grad_output):
        module.unshard(bwd_pass=True)
        for param_group in module._fsdp_param_groups:
            has_fused_wgrad = any(
                getattr(param, "_mfsdp_recorded_te_wgrad", False) for param in param_group.params
            )
            if has_fused_wgrad and param_group.main_grad_buffer is not None:
                param_group._init_dist_grads()
                param_group.main_grad_buffer.fetch_buffer()

    return hook


def _make_bwd_post_hook(module):
    def hook(mod, grad_input, grad_output):
        module.reshard()
        # Clear grad to avoid memory leak in CUDA graph capture.
        for param_group in module._fsdp_param_groups:
            for param in param_group.params:
                param.grad = None

    return hook
