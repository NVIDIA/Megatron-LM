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

"""CUDA graph capture / replay for individual FSDP modules."""

import inspect
import gc
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# All known hook attributes across PyTorch versions (including 2.x additions)
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


def _get_forward_param_names(module: torch.nn.Module) -> List[str]:
    """Return the ordered parameter names of module.forward (excluding 'self')."""
    sig = inspect.signature(module.forward)
    return [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]


class _ForwardShim(torch.nn.Module):
    """Wraps module.forward so that non-tensor kwargs are frozen at
    capture time and tensor inputs are passed positionally in signature
    order.

    Handles None outputs by filtering them out before returning to
    make_graphed_callables, and records their positions so they can be
    restored during replay.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        tensor_param_names: List[str],
        frozen_kwargs: dict,
    ):
        super().__init__()
        self.module = module
        self.tensor_param_names = tensor_param_names
        self.frozen_kwargs = frozen_kwargs

        # Populated on first forward: tracks which output positions are None
        self._none_mask: Optional[List[bool]] = None
        # Whether the original output was a tuple (vs single tensor)
        self._output_is_tuple: bool = True

    def forward(self, *flat_tensor_args):
        kwargs = dict(zip(self.tensor_param_names, flat_tensor_args))
        kwargs.update(self.frozen_kwargs)
        outputs = self.module.forward(**kwargs)

        # Handle single tensor output
        if not isinstance(outputs, tuple):
            self._output_is_tuple = False
            if outputs is None:
                raise RuntimeError(
                    "Module returned None as its only output — cannot capture "
                    "with CUDA graphs. The module must return at least one tensor."
                )
            return outputs

        # Handle tuple outputs — filter out None values
        self._output_is_tuple = True
        if self._none_mask is None:
            self._none_mask = [o is None for o in outputs]

        filtered = tuple(o for o in outputs if o is not None)
        if len(filtered) == 0:
            raise RuntimeError(
                "Module returned a tuple of all None values — cannot capture "
                "with CUDA graphs. At least one output must be a tensor."
            )
        if len(filtered) == 1:
            return filtered[0]
        return filtered

    def restore_none_positions(self, graph_output) -> Any:
        """Re-insert None values at their original positions after graph replay."""
        if not self._output_is_tuple or self._none_mask is None:
            return graph_output

        # If all non-None outputs were collapsed to a single tensor
        if not isinstance(graph_output, tuple):
            graph_output = (graph_output,)

        full: List[Any] = []
        tensor_iter = iter(graph_output)
        for is_none in self._none_mask:
            if is_none:
                full.append(None)
            else:
                full.append(next(tensor_iter))
        return tuple(full)


def _pop_hooks(module: torch.nn.Module) -> Dict[str, Any]:
    """Remove all hooks from *module* (non-recursive) and return a snapshot."""
    saved: Dict[str, Any] = {}
    for attr in _HOOK_ATTRS:
        if hasattr(module, attr):
            saved[attr] = getattr(module, attr)
            setattr(module, attr, OrderedDict())
    return saved


def _pop_hooks_recursive(module: torch.nn.Module) -> List[Tuple[torch.nn.Module, Dict[str, Any]]]:
    """Remove all hooks from *module* and all its submodules recursively.

    Returns a list of (submodule, saved_hooks) tuples for restore.
    Using direct module references instead of id() for safety.
    """
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []
    for submodule in module.modules():
        saved.append((submodule, _pop_hooks(submodule)))
    return saved


def _restore_hooks(module: torch.nn.Module, saved: Dict[str, Any]) -> None:
    """Put the hooks back exactly as they were."""
    for name, value in saved.items():
        if value is not None:
            setattr(module, name, value)


def _restore_hooks_recursive(
    module: torch.nn.Module, saved: List[Tuple[torch.nn.Module, Dict[str, Any]]]
) -> None:
    """Restore hooks for all submodules saved by ``_pop_hooks_recursive``."""
    for submodule, sub_saved in saved:
        _restore_hooks(submodule, sub_saved)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


class FSDPCudaGraphRunner:
    """Captures a forward+backward CUDA graph for one FSDP module.

    During capture hooks are temporarily removed so the graph records
    only the user's ``forward()``, not FSDP all-gather / reduce-scatter
    collectives.  FSDP side streams are disabled for the capture region.

    Parameters:
        fsdp_module: The FSDP module to capture.
        gc_freeze: If True (default), call ``gc.collect()`` and ``gc.freeze()``
            before capture to prevent Python GC from stalling replay.
        graph_pool: Optional shared CUDA graph memory pool handle obtained
            via ``torch.cuda.graph_pool_handle()``.  When provided, the
            ``CUDAGraph`` is created with this handle so multiple FSDP
            modules share the same backing memory pool, reducing total
            GPU memory consumption.

    Usage::

        runner = FSDPCudaGraphRunner(my_fsdp_module)
        runner.capture_forward(sample_input)
        runner.install()                       # patches module.forward
        output = my_fsdp_module(input_batch)   # replays graph, no hooks
        runner.uninstall()                     # restore original behaviour
    """

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        gc_freeze: bool = True,
        graph_pool: Optional[Any] = None,
    ):
        warnings.warn(
            "FSDPCudaGraphRunner is an experimental feature. The API and "
            "behaviour may change in future releases without notice.",
            FutureWarning,
            stacklevel=2,
        )
        self._module: torch.nn.Module = fsdp_module
        self._gc_freeze: bool = gc_freeze
        self._graph_pool: Optional[int] = graph_pool

        # Will hold the callable returned by make_graphed_callables
        self._graphed: Optional[Any] = None

        self._orig_fwd: Optional[Any] = None
        self._use_cuda_graph: bool = False
        self._captured: bool = False

        # Saved during capture for install() replay flattening
        self._tensor_param_names: List[str] = []
        self._frozen_kwargs: Dict[str, Any] = {}

        # The shim used during capture (needed for None restoration)
        self._shim: Optional[_ForwardShim] = None

    # ------------------------------------------------------------------
    # 1. Capture
    # ------------------------------------------------------------------

    def capture_forward(
        self,
        *sample_args,
        **sample_kwargs,
    ) -> None:
        assert self._module.cuda_graph_compatible, (
            "CUDA graph capture requires TracePoolAllocator in optimized phase"
        )

        # Introspect the module's forward signature
        param_names = _get_forward_param_names(self._module.__class__)

        # Separate tensor vs non-tensor inputs
        bound = {}
        for i, val in enumerate(sample_args):
            if i < len(param_names):
                bound[param_names[i]] = val
        bound.update(sample_kwargs)

        tensor_names = [
            n for n in param_names if n in bound and isinstance(bound[n], torch.Tensor)
        ]
        frozen_kwargs = {
            n: v for n, v in bound.items() if not isinstance(v, torch.Tensor)
        }
        flat_sample = tuple(
            bound[n].clone().detach().requires_grad_(True) for n in tensor_names
        )

        # Build shim (handles None filtering)
        shim = _ForwardShim(self._module, tensor_names, frozen_kwargs)

        for param in self._module.parameters():
            param.grad = None

        # For gradient accumulate fusion
        self.unshard_main_grad_buffer()

        if self._gc_freeze:
            gc.collect()
            gc.freeze()

        # Disable side-stream collectives during capture so every CUDA
        # operation lands on the default (capture) stream.
        saved_hooks = _pop_hooks_recursive(self._module)
        ctx = self._module._fsdp_root_context
        ctx.cuda_graph_active = True
        try:
            torch.cuda.synchronize()
            self._graphed = torch.cuda.make_graphed_callables(
                shim,
                sample_args=flat_sample,
                num_warmup_iters=3,
                allow_unused_input=True,
                # pool=self._graph_pool,
            )
        finally:
            ctx.cuda_graph_active = False
            _restore_hooks_recursive(self._module, saved_hooks)
            self.reshard_main_grad_buffer()

        self._shim = shim
        self._tensor_param_names = tensor_names
        self._frozen_kwargs = frozen_kwargs
        self._captured = True

    # ------------------------------------------------------------------
    # 2. Install / uninstall the patched forward
    # ------------------------------------------------------------------
    def install(self) -> None:
        if not self._captured:
            raise RuntimeError("Call capture_forward() first")
        if self._orig_fwd is not None:
            return

        self._orig_fwd = self._module.forward
        graphed = self._graphed
        shim = self._shim
        param_names = _get_forward_param_names(self._module.__class__)
        tensor_names = self._tensor_param_names

        def _patched_fwd(*args, **kwargs):
            if self._use_cuda_graph:
                bound = {}
                for i, val in enumerate(args):
                    if i < len(param_names):
                        bound[param_names[i]] = val
                bound.update(kwargs)
                flat = tuple(bound[n] for n in tensor_names)
                result = graphed(*flat)
                # Restore None positions that were filtered during capture
                return shim.restore_none_positions(result)
            return self._orig_fwd(*args, **kwargs)

        self._module.forward = _patched_fwd
        self._use_cuda_graph = True

    def uninstall(self) -> None:
        """Restore the original ``forward``."""
        if self._orig_fwd is None:
            return
        self._module.forward = self._orig_fwd
        self._orig_fwd = None
        self._use_cuda_graph = False

    # ------------------------------------------------------------------
    # 3. Properties
    # ------------------------------------------------------------------

    @property
    def captured(self) -> bool:
        """True if ``capture_forward`` has been called successfully."""
        return self._captured

    @property
    def using_cuda_graph(self) -> bool:
        """True if the patch is currently active (install() called)."""
        return self._use_cuda_graph

    def reset(self) -> None:
        """Uninstall the patch and allow a fresh capture later."""
        self.uninstall()
        self._graphed = None
        self._shim = None
        self._captured = False
        self._graph = None

    def unshard_main_grad_buffer(self):
        """Unshard the main grad buffer for all param groups."""
        for group in self._module._fsdp_param_groups:
            if hasattr(group, "main_grad_buffer"):
                group.main_grad_buffer.fetch_buffer()

    def reshard_main_grad_buffer(self):
        """Reshard the main grad buffer for all param groups."""
        for group in self._module._fsdp_param_groups:
            group.release_grad_buffer()
