# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Module mixin for the minimal Megatron-FSDP path."""

import weakref
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .indexed_order import IndexedOrder
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import MeshAxis, Placements


class FsdpContext:
    """Runtime stream and prefetch state shared by one FSDP subtree."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because it can be detected when ``model_weight``, after syncing
    # from ``main_weight``, has placements different from ``Placements.optimizer``.
    is_last_microbatch: bool
    # True from the root pre-backward hook until autograd completes. Forward
    # hooks use this to identify activation recomputation inside backward.
    backward_phase: bool
    root_module: "FsdpModule"
    # Static orders used to drive all-gather prefetch. We may want to switch to
    # capturing runtime order if static module order proves too fragile. Each
    # FsdpModule tracks its own materialized state via ``FsdpModule._unshard_event``.
    forward_order: IndexedOrder["FsdpModule"]
    backward_order: IndexedOrder["FsdpModule"]

    def __init__(self, device: torch.device, root_module: "FsdpModule") -> None:
        """Create rank-local runtime state for a root FSDP subtree.

        Args:
            device: Device on which this context schedules communication.
            root_module: Outermost module that owns this context.
        """
        self.root_module = root_module
        self.is_last_microbatch = True
        self.backward_phase = False
        self.forward_order = IndexedOrder()
        self.backward_order = IndexedOrder()
        with torch.cuda.device(device):
            self.allgather_stream = torch.cuda.Stream()
            self.reduce_scatter_stream = torch.cuda.Stream()

    def current_stream(self) -> torch.cuda.Stream:
        """Current stream on this context's device."""
        return torch.cuda.current_stream(self.allgather_stream.device)

    def register_post_backward_final_callback(self) -> None:
        """Register this root context's final callback for the current backward.

        Root ``post_backward()`` means only that root-owned parameters have
        accumulated gradients; it may run before descendant reductions, or not
        run at all when the root owns no trainable parameters. Waiting at
        autograd completion orders consumers after every descendant reduction.
        """
        torch.autograd.Variable._execution_engine.queue_callback(self.finalize_backward)

    def finalize_backward(self) -> None:
        """Order the current stream after reductions and leave the backward phase."""
        self.current_stream().wait_stream(self.reduce_scatter_stream)
        self.backward_phase = False


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    # Name relative to the root FSDP module from named_modules().
    # Root uses "" and None means uninitialized.
    _name: str | None
    _parameter_groups: tuple[FsdpParameterGroup, ...]
    _context: FsdpContext | None
    _num_ready_grad_parameters: int
    _num_trainable_parameters: int
    _post_backward_issued: bool
    # Event recorded after this FsdpModule's full parameters are materialized.
    # ``None`` lets pre_forward enqueue an all-gather unless an earlier FsdpModule
    # already prefetched this module.
    _unshard_event: torch.cuda.Event | None

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        use_symm_mem: bool = False,
        fine_grained: bool = False,
        skip_backward_callback: bool = False,
        grad_divisor: int = 1,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        self._context = None
        self._name = None
        self._unshard_event = None
        owned_parameters = _collect_owned_parameters(self)
        axis_indices = tuple(_axis_index(mesh, axis) for axis in placements.dp_axes)
        assert axis_indices == tuple(
            range(mesh.ndim)
        ), "FSDP requires dp_axes to match every mesh axis in mesh order for now."

        if grad_divisor <= 0:
            raise ValueError(f"grad_divisor must be positive, got {grad_divisor}.")

        if any(parameter.is_meta for parameter in owned_parameters.values()):
            # Collect nested FsdpModules to skip — they were already materialized
            # when their own fully_shard() processed them bottom-up.
            ignored_modules: set = set()
            for _, child in self.named_modules():
                if child is not self and isinstance(child, FsdpModule):
                    for sub in child.modules():
                        ignored_modules.add(sub)
            _materialize_meta_params(self, mesh, ignored_modules)
            # Parameters were replaced by m._apply() — re-read.
            owned_parameters = _collect_owned_parameters(self)

        meta_parameter_names = [
            name for name, parameter in owned_parameters.items() if parameter.is_meta
        ]
        if meta_parameter_names:
            raise RuntimeError(
                "FSDP parameter materialization left parameters on the meta device: "
                + ", ".join(repr(name) for name in meta_parameter_names)
            )

        parameter_groups = [
            FsdpParameterGroup(
                owning_module=self,
                parameters=group_parameters,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=mixed_precision_policy,
                use_symm_mem=use_symm_mem,
                grad_divisor=grad_divisor,
            )
            for group_parameters in _group_parameters(owned_parameters)
        ]
        self._parameter_groups = tuple(parameter_groups)
        self._num_ready_grad_parameters = 0
        self._num_trainable_parameters = sum(
            len(group.fsdp_parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._post_backward_issued = False
        self._register_hooks(
            fine_grained=fine_grained, skip_backward_callback=skip_backward_callback
        )
        # Public callables for 1F1B EP overlap schedule integration.
        self.post_forward_release_module = partial(self._post_forward_release)
        self.post_backward_release_module = self._post_backward_release

    def _post_forward_release(self, hook_module=None) -> None:
        """Release forward-pass parameters (reshard only, no gradient reduction).

        Matching the v1 contract: takes an optional hook_module argument
        (ignored — this FsdpModule manages its own parameters)."""
        self.reshard_parameters()

    def _post_backward_release(self, hook_module=None) -> None:
        """Release backward-pass parameters (reshard + reduce gradients).

        Matching the v1 contract: takes an optional hook_module argument
        (ignored — this FsdpModule manages its own parameters)."""
        modules = cast(nn.Module, self).modules()
        for module in reversed(list(modules)):
            if isinstance(module, FsdpModule):
                module._issue_post_backward()

    def _lazy_init_context(self) -> None:
        """Initialize one shared runtime context for this FSDP root subtree.

        MFSDP v2 requires users to apply ``fully_shard`` bottom-up, so child FSDP
        modules are constructed before their eventual root module is constructed.
        This method resolves the root lazily on the first forward through the
        outermost FSDP module and shares that one context with every FSDP
        descendant.

        Alternatives considered:
        - Eagerly initialize contexts during ``fully_shard``. When a parent is
          sharded, we could create a new root context and reassign it to all
          descendant FSDP modules. This creates transient child contexts that are
          never used if the parent is later sharded, and each parent shard must
          walk its descendants again, making nested sharding quadratic.
        - Store an ``is_root`` field on each FSDP module. ``fully_shard`` could
          mark newly sharded modules as roots and clear that flag on descendant
          FSDP modules when a parent is sharded. This avoids creating unused
          contexts but moves root tracking onto every FSDP module, adding
          per-module state that must stay consistent with the final sharded
          module hierarchy.
        """
        if self._context is not None:
            return

        root_module = cast(nn.Module, self)
        first_parameter = next(root_module.parameters(), None)
        if first_parameter is None:
            raise RuntimeError("FSDP root module requires at least one parameter in its subtree.")

        context = FsdpContext(device=first_parameter.device, root_module=self)
        # named_modules() yields FsdpModules in registration order, which is the static
        # forward execution order used to prefetch the next FsdpModule's all-gather.
        for submodule_name, submodule in root_module.named_modules():
            if not isinstance(submodule, FsdpModule):
                continue
            if submodule._context is not None:
                raise RuntimeError(
                    "FSDP context is already initialized for a descendant module. "
                    "Run forward through the root FSDP module first."
                )
            submodule._context = context
            submodule._name = submodule_name
            context.forward_order.append(submodule)

        # Backward starts from the root pre-backward hook before visiting child
        # subtrees in reverse module order.
        _collect_backward_order(root_module, context.backward_order)

    @property
    def context(self) -> FsdpContext:
        """Return the initialized runtime context."""
        assert self._context is not None
        return self._context

    @property
    def name(self) -> str:
        """Return this FsdpModule's name."""
        name = self._name
        if name is None:
            raise RuntimeError("FSDP module name has not been initialized.")
        return name

    def is_root(self) -> bool:
        """Return whether this module is the outermost FsdpModule in its context."""
        if self._context is None:
            return False
        return self.context.root_module is self

    def _register_hooks(
        self, fine_grained: bool = False, skip_backward_callback: bool = False
    ) -> None:
        module = cast(nn.Module, self)
        if fine_grained:
            _register_fine_grained_forward_hooks(self)
            _register_fine_grained_backward_hooks(self)
        else:
            module.register_forward_pre_hook(lambda _module, _args: self.pre_forward())
        module.register_forward_hook(lambda _module, _args, _output: self.post_forward())
        if not fine_grained:
            module.register_full_backward_pre_hook(
                lambda _module, _grad_output: self.pre_backward()
            )
        if self._num_trainable_parameters == 0:
            module.register_full_backward_hook(
                lambda _module, _grad_input, _grad_output: self.post_backward()
            )
            return

        if skip_backward_callback:
            return

        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            for fsdp_parameter in group.fsdp_parameters:
                fsdp_parameter.unsharded.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _make_grad_hook(self) -> Callable[[nn.Parameter], None]:
        def grad_hook(_parameter: nn.Parameter) -> None:
            self._num_ready_grad_parameters += 1
            if self._num_ready_grad_parameters == self._num_trainable_parameters:
                self.post_backward()

        return grad_hook

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute and prefetch the next FsdpModule.

        While this FsdpModule computes, we issue the next FsdpModule's all-gather
        on the comm stream, so ``AG_{i+1}`` is launched before ``F_i`` finishes.
        """
        self._lazy_init_context()
        torch.cuda.nvtx.range_push(self._nvtx_label("forward"))
        self._num_ready_grad_parameters = 0
        context = self.context
        allgather_stream = context.allgather_stream
        current_stream = context.current_stream()

        if self.is_root():
            allgather_stream.wait_stream(current_stream)

        self._unshard_parameter_groups()
        assert self._unshard_event is not None
        # Compute waits only for this FsdpModule's all-gather (the prefetch below is
        # issued afterwards, so it is free to run concurrently with this FsdpModule).
        current_stream.wait_event(self._unshard_event)

        # Activation recomputation runs forward hooks inside backward. Do not
        # prefetch the next module in forward order: its backward may already
        # be complete, so no later backward hook would reshard it.
        if not context.backward_phase:
            next_module = context.forward_order.next_item(self)
            if next_module is not None:
                next_module._unshard_parameter_groups()

    def _unshard_parameter_groups(self) -> None:
        """Unshard this FsdpModule's parameter groups on the all-gather stream.

        If ``_unshard_event`` is already set, this FsdpModule was already
        unsharded or prefetched and this method is a no-op. Otherwise, this
        method records ``_unshard_event`` after materialization so compute
        can wait without depending on later release work.
        """
        if self._unshard_event is not None:
            return

        allgather_stream = self.context.allgather_stream
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.unshard_parameters()
            self._unshard_event = allgather_stream.record_event()

    def unshard_parameters(self) -> None:
        """Public API: all-gather full parameter storage for compute.

        Idempotent — if parameters are already unsharded, this is a no-op.
        Called by the 1F1B EP overlap schedule via fine-grained sub-module
        hooks before each individual sub-module compute.
        """
        self._lazy_init_context()
        self._unshard_parameter_groups()
        if self._unshard_event is not None:
            self.context.current_stream().wait_event(self._unshard_event)

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute.

        Skips resharding during backward-phase recompute so that overlapping
        backward passes can still access all-gathered parameters.
        """
        if self.context.backward_phase:
            torch.cuda.nvtx.range_pop()
            return
        self._reshard_parameter_groups()
        torch.cuda.nvtx.range_pop()

    def _reshard_parameter_groups(self) -> None:
        """Reshard parameter groups and release unsharded storage after compute.

        This method clears ``_unshard_event`` after queuing the release, so
        future users enqueue a fresh all-gather.
        """
        for group in self._parameter_groups:
            group.reshard_parameters()

        allgather_stream = self.context.allgather_stream
        allgather_stream.wait_stream(self.context.current_stream())
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.release_unsharded_storage()
            self._unshard_event = None

    def reshard_parameters(self) -> None:
        """Public API: release all-gathered storage and install DTensor parameters.

        Called by the 1F1B EP overlap schedule's per-layer release hooks
        after compute completes on a sub-module.
        """
        for group in self._parameter_groups:
            group.reshard_parameters()

        allgather_stream = self.context.allgather_stream
        allgather_stream.wait_stream(self.context.current_stream())
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.release_unsharded_storage()
            self._unshard_event = None

    def pre_backward(self, register_final_callback: bool = True) -> None:
        """Prepare full parameters and prefetch the next FsdpModule in backward order.

        Args:
            register_final_callback: Whether to finalize through the autograd engine.
                Manual backward schedules finalize explicitly in ``post_backward()``.
        """
        torch.cuda.nvtx.range_push(self._nvtx_label("backward"))
        context = self.context
        current_stream = context.current_stream()
        if self.is_root():
            context.backward_phase = True
            for module in context.forward_order:
                module._post_backward_issued = False
            if register_final_callback:
                context.register_post_backward_final_callback()
            # Fork the reduce-scatter stream from the current stream once, at the
            # start of backward, so every module's post-backward reduce-scatter is
            # part of any active CUDA-graph capture. A stream only joins the
            # capture via this wait_stream edge; without it the first allocation on
            # the reduce-scatter stream falls back to a raw cudaMalloc, which is
            # illegal during capture. Later modules are covered by the post-copy
            # fork each preceding module issues before its collective.
            context.reduce_scatter_stream.wait_stream(current_stream)

        self._unshard_parameter_groups()
        assert self._unshard_event is not None
        current_stream.wait_event(self._unshard_event)

        next_module = context.backward_order.next_item(self)
        if next_module is not None:
            next_module._unshard_parameter_groups()

    def post_backward(self, finalize_context: bool = False) -> None:
        """Reduce gradients and return parameters to their sharded resting state.

        Args:
            finalize_context: Whether to finalize the root context synchronously.
        """
        if finalize_context:
            assert self.is_root()
            for module in reversed(list(self.context.forward_order)):
                module._issue_post_backward()
            self.context.finalize_backward()
        else:
            self._issue_post_backward()
        torch.cuda.nvtx.range_pop()

    def _issue_post_backward(self) -> None:
        """Reshard and reduce this module's gradients at most once per backward."""
        if self._post_backward_issued:
            return
        self.reshard_parameters()
        self.reduce_grad()
        self._num_ready_grad_parameters = 0
        self._post_backward_issued = True

    def _reduce_gradient_groups(self) -> None:
        """Pack gradients and immediately launch their reduce-scatters."""
        context = self.context
        reduce_scatter_stream = context.reduce_scatter_stream
        current_stream = context.current_stream()

        for group in self._parameter_groups:
            if not group.requires_grad:
                continue

            with torch.cuda.stream(reduce_scatter_stream):
                partial_grad = group.allocate_partial_grad_buffer()

            current_stream.wait_stream(reduce_scatter_stream)
            group.copy_gradients_to_partial_buffer(partial_grad)

            reduce_scatter_stream.wait_stream(current_stream)
            with torch.cuda.stream(reduce_scatter_stream):
                group.reduce_partial_gradients(partial_grad, self.context.is_last_microbatch)

    def reduce_grad(self) -> None:
        """Public API: pack gradients and launch their reduce-scatters.

        Called by the 1F1B EP overlap schedule's per-layer release hooks
        after backward compute completes.  Only operates on parameter groups
        that require gradients.
        """
        self._reduce_gradient_groups()

    def _replace_param_with_raw_if_needed(self) -> None:
        """Initialize the root context before a fine-grained schedule runs.

        Provided for compatibility with the 1F1B EP overlap schedule, which
        calls this method to swap optimizer-facing DTensor parameters back to
        raw nn.Parameters before accessing sub-modules directly.  The
        experimental API stores raw tensors backed by DBuffer at all times,
        so no swap is needed, but initializing here ensures a child FSDP unit
        cannot mistake itself for the root when it executes first.
        """
        self._lazy_init_context()

    @property
    def parameter_groups(self) -> tuple[FsdpParameterGroup, ...]:
        """Parameter groups owned by this FsdpModule."""
        return self._parameter_groups

    def _nvtx_label(self, phase: Literal["forward", "backward"]) -> str:
        name = self.name if self.name else "<root>"
        return f"MFSDP {name} {phase}"


def _collect_backward_order(module: nn.Module, order: IndexedOrder["FsdpModule"]) -> None:
    """Collect FsdpModules in static backward prefetch order."""
    if isinstance(module, FsdpModule):
        order.append(module)

    for child in reversed(list(module.children())):
        _collect_backward_order(child, order)


def _materialize_meta_params(
    module: nn.Module,
    mesh: DeviceMesh,
    ignored_modules: set | None = None,
) -> None:
    """Materialize meta parameters to real tensors and initialize weights.

    Replaces every meta ``nn.Parameter`` with a real tensor on the current
    CUDA device, calls ``m.reset_parameters()`` to re-initialize, and
    broadcasts weights from DP rank 0.

    Args:
        module: Root module whose meta parameters should be materialized.
        mesh: Data-parallel device mesh used for the weight broadcast.
        ignored_modules: Set of module instances to skip (nested FsdpModules
            already materialized by their own ``fully_shard()`` call).
    """
    ignored_modules = ignored_modules or set()
    device = torch.cuda.current_device()
    device = device if isinstance(device, torch.device) else torch.device("cuda", device)

    from torch.distributed.tensor import DTensor

    for name, m in reversed(list(module.named_modules())):
        if m in ignored_modules:
            continue
        if m is not module and isinstance(m, FsdpModule):
            continue
        if not any(p.is_meta for p in m.parameters(recurse=False)):
            m._apply(lambda t: t if t.is_meta else t.to(device), recurse=False)
            continue

        m._apply(lambda t: (torch.empty_like(t, device=device) if t.is_meta else t), recurse=False)
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif hasattr(m, "_reset_parameters"):
            m._reset_parameters()
        else:
            raise ValueError(f"Module {name!r} contains meta parameters but cannot reset them")

        m._apply(lambda t: t if t.is_meta else t.to(device), recurse=False)
        for p in m.parameters(recurse=False):
            if p.is_meta:
                raise RuntimeError(
                    f"Module {name!r} contains meta parameters after materialization"
                )

    if mesh.size() > 1:
        for param in module.parameters():
            if param.is_meta or isinstance(param, DTensor):
                continue
            for mesh_dim in range(mesh.ndim):
                group = mesh.get_group(mesh_dim=mesh_dim)
                if torch.distributed.get_world_size(group) == 1:
                    continue
                src_rank = torch.distributed.get_global_rank(group, 0)
                torch.distributed.broadcast(param.data, src=src_rank, group=group)


# ---------------------------------------------------------------------------
# Fine-grained hook registration for 1F1B EP overlap support
# ---------------------------------------------------------------------------

_FSDP_PARENT_MODULE_REF_ATTR = "_fsdp_parent_module_ref"


def _find_fsdp_target(submodule: nn.Module) -> FsdpModule | None:
    """Return the nearest parent FsdpModule for *submodule*, if any."""
    if isinstance(submodule, FsdpModule):
        return submodule
    parent_ref = getattr(submodule, _FSDP_PARENT_MODULE_REF_ATTR, None)
    return parent_ref() if parent_ref is not None else None


def _register_fine_grained_forward_hooks(fsdp_module: FsdpModule) -> None:
    """Register pre-forward hooks on every sub-module of *fsdp_module*.

    When the 1F1B EP overlap schedule calls individual sub-modules directly
    (e.g., ``layer.attn.forward()``), the hook resolves the parent FsdpModule
    and calls ``unshard_parameters()``.
    """
    for submodule in fsdp_module.modules():
        if submodule is fsdp_module:
            continue
        target = _find_fsdp_target(submodule)
        if target is not None and target is not fsdp_module:
            continue
        object.__setattr__(submodule, _FSDP_PARENT_MODULE_REF_ATTR, weakref.ref(fsdp_module))
        submodule.register_forward_pre_hook(
            _fine_grained_pre_forward_hook, prepend=True, with_kwargs=True
        )


def _fine_grained_pre_forward_hook(submodule: nn.Module, args, kwargs) -> None:
    """Pre-forward hook for fine-grained sub-modules."""
    target = _find_fsdp_target(submodule)
    if target is None:
        return
    target.unshard_parameters()


def _register_fine_grained_backward_hooks(fsdp_module: FsdpModule) -> None:
    """Register pre-backward hooks on every sub-module of *fsdp_module*.

    Uses ``register_multi_grad_hook`` on sub-module output tensors.  When
    autograd reaches a sub-module during backward, the hook calls
    ``unshard_parameters()`` on the parent FsdpModule.
    """
    for submodule in fsdp_module.modules():
        if submodule is fsdp_module:
            continue
        target = _find_fsdp_target(submodule)
        if target is not None and target is not fsdp_module:
            continue
        _create_fine_grained_backward_hook(submodule)


def _create_fine_grained_backward_hook(submodule: nn.Module) -> None:
    """Wrap *submodule* so a pre-backward hook fires via register_multi_grad_hook."""

    def _forward_hook(_module, inputs, output):
        output_list = []
        if isinstance(output, torch.Tensor):
            output_list = [output]
        elif isinstance(output, (tuple, list)):
            output_list = [t for t in output if isinstance(t, torch.Tensor)]

        def _multi_grad_hook(grads):
            target = _find_fsdp_target(submodule)
            if target is None:
                return
            target.unshard_parameters()

        torch.autograd.graph.register_multi_grad_hook(output_list, _multi_grad_hook, mode="any")
        return output

    submodule.register_forward_hook(_forward_hook)


def _axis_index(mesh: DeviceMesh, axis: MeshAxis) -> int:
    if isinstance(axis, int):
        axis_index = axis
        if axis_index < 0:
            axis_index += mesh.ndim
        if axis_index < 0 or axis_index >= mesh.ndim:
            raise ValueError(f"Mesh axis {axis} is out of bounds for mesh ndim {mesh.ndim}.")
        return axis_index

    dim_names = mesh.mesh_dim_names
    if dim_names is None or axis not in dim_names:
        raise ValueError(f"Mesh axis {axis!r} is not present in mesh dim names {dim_names}.")
    return dim_names.index(axis)


def _collect_owned_parameters(root_module: nn.Module) -> dict[str, nn.Parameter]:
    parameters: dict[str, nn.Parameter] = {}

    def visit(submodule: nn.Module, submodule_fqn: str) -> None:
        direct_parameters = submodule.named_parameters(recurse=False, remove_duplicate=False)

        for local_parameter_name, parameter in direct_parameters:
            parameter_fqn = (
                f"{submodule_fqn}.{local_parameter_name}" if submodule_fqn else local_parameter_name
            )
            if get_containing_parameter_group(parameter) is not None:
                raise ValueError(
                    f"Parameter {parameter_fqn!r} is already owned by another FsdpModule."
                )
            parameters[parameter_fqn] = parameter

        for child_name, child_module in submodule.named_children():
            if isinstance(child_module, FsdpModule):
                continue
            child_fqn = f"{submodule_fqn}.{child_name}" if submodule_fqn else child_name
            visit(child_module, child_fqn)

    visit(root_module, "")
    return parameters


def _group_parameters(parameters: dict[str, nn.Parameter]) -> list[dict[str, nn.Parameter]]:
    grouped: dict[tuple[torch.dtype, bool], dict[str, nn.Parameter]] = {}
    for name, parameter in parameters.items():
        key = (parameter.dtype, parameter.requires_grad)
        grouped.setdefault(key, {})[name] = parameter
    return [grouped[key] for key in grouped]
