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

import enum
import weakref
from collections.abc import Callable
from typing import Literal, cast
from weakref import ref

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .indexed_order import IndexedOrder
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import MeshAxis, Placements


def _is_in_backward() -> bool:
    """Return whether the current thread is executing an autograd GraphTask."""
    return torch._C._current_graph_task_id() != -1


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because it can be detected when ``model_weight``, after syncing
    # from ``main_weight``, has placements different from ``Placements.optimizer``.
    is_last_microbatch: bool
    use_symmetric_memory: bool
    unify_communication_stream: bool
    # Static orders used to drive all-gather prefetch. We may want to switch to
    # capturing runtime order if static module order proves too fragile. Each
    # FsdpModule tracks its own materialized state via ``FsdpModule._unshard_event``.
    forward_order: IndexedOrder["FsdpModule"]
    backward_order: IndexedOrder["FsdpModule"]

    def __init__(
        self,
        device: torch.device,
        use_symmetric_memory: bool = False,
        unify_communication_stream: bool = False,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: Device on which this context schedules communication.
            use_symmetric_memory: Whether modules constructed in this context allocate
                communication staging buffers from PyTorch's NCCL symmetric-memory pool.
            unify_communication_stream: Whether all-gathers and reduce-scatters share one
                communication stream to reduce peak transient memory.
        """
        self.is_last_microbatch = True
        self.use_symmetric_memory = use_symmetric_memory
        self.unify_communication_stream = unify_communication_stream
        self.forward_order = IndexedOrder()
        self.backward_order = IndexedOrder()
        # Construction-only; empty after finalization.
        self._registered_modules: list[FsdpModule] = []
        self._is_finalized = False
        self.allgather_stream = torch.cuda.Stream(device)
        if unify_communication_stream:
            # A unified stream lets an all-gather reuse the storage released by a
            # preceding reduce-scatter.
            self.reduce_scatter_stream = self.allgather_stream
        else:
            self.reduce_scatter_stream = torch.cuda.Stream(device)

    def register_module(self, module: "FsdpModule") -> None:
        """Register a module constructed in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots, names, and cross-root prefetch orders."""
        if self._is_finalized:
            raise RuntimeError("FSDP context is already finalized.")

        children: set[FsdpModule] = set()
        for module in self._registered_modules:
            _collect_fsdp_children(cast(nn.Module, module), children)
        # FsdpModules that are not descendants of any other FsdpModule.
        roots = [module for module in self._registered_modules if module not in children]

        for root in roots:
            root._is_root = True
            for name, module in cast(nn.Module, root).named_modules():
                if not isinstance(module, FsdpModule):
                    continue
                module._name = name
                self.forward_order.append(module)

        for root in reversed(roots):
            _collect_backward_order(cast(nn.Module, root), self.backward_order)

        self._registered_modules.clear()
        self._is_finalized = True

    def ensure_finalized(self) -> None:
        """Raise if construction has not completed for this context."""
        if not self._is_finalized:
            raise RuntimeError(
                "FSDP context is not finalized. Exit fully_shard_context before running forward."
            )

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

        def post_backward_final_callback() -> None:
            self.current_stream().wait_stream(self.reduce_scatter_stream)

        torch.autograd.Variable._execution_engine.queue_callback(post_backward_final_callback)


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    class Phase(enum.Enum):
        """Lifecycle phase of this FsdpModule."""

        RESTING = enum.auto()
        FORWARD = enum.auto()
        BACKWARD = enum.auto()

    # Name relative to the root FSDP module from named_modules().
    # Root uses "" and None means uninitialized.
    _name: str | None
    _parameter_groups: tuple[FsdpParameterGroup, ...]
    _context: FsdpContext
    _num_ready_grad_parameters: int
    _is_root: bool
    _num_trainable_parameters: int
    # Event recorded after this FsdpModule's full parameters are materialized.
    # ``None`` lets pre_forward enqueue an all-gather unless an earlier FsdpModule
    # already prefetched this module.
    _unshard_event: torch.cuda.Event | None
    # ``phase`` is FORWARD between pre_forward() and post_forward(), BACKWARD
    # between pre_backward() and post_backward(), and RESTING otherwise. The only
    # exception is non-reentrant activation recomputation: it runs between pre_backward()
    # and post_backward(), preserving BACKWARD through its nested forward hooks.
    _phase: Phase

    def __init__(
        self,
        context: FsdpContext,
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        grad_divisor: int = 1,
        use_symmetric_memory: bool = False,
        fine_grained: bool = False,
        skip_backward_callback: bool = False,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        self._context = context
        self._is_root = False
        self._name = None
        self._unshard_event = None
        self._phase = FsdpModule.Phase.RESTING
        owned_parameters = _collect_owned_parameters(self)
        assert tuple(placements.dp_axes) == tuple(
            range(mesh.ndim)
        ), "FSDP requires dp_axes to match every mesh axis in mesh order for now."
        if grad_divisor <= 0:
            raise ValueError(f"grad_divisor must be positive, got {grad_divisor}.")

        meta_parameter_names = [
            name for name, parameter in owned_parameters.items() if parameter.is_meta
        ]
        if meta_parameter_names:
            raise RuntimeError(
                "MFSDP v2 requires materialized parameters; "
                "found meta parameters: " + ", ".join(repr(name) for name in meta_parameter_names)
            )

        parameter_groups = [
            FsdpParameterGroup(
                owning_module=self,
                parameters=group_parameters,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=mixed_precision_policy,
                allgather_stream=context.allgather_stream,
                reduce_scatter_stream=context.reduce_scatter_stream,
                grad_divisor=grad_divisor,
                use_symmetric_memory=use_symmetric_memory,
            )
            for group_parameters in _group_parameters(owned_parameters)
        ]
        self._parameter_groups = tuple(parameter_groups)
        self._num_ready_grad_parameters = 0
        self._num_trainable_parameters = sum(
            len(group.fsdp_parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._register_hooks(
            fine_grained=fine_grained, skip_backward_callback=skip_backward_callback
        )
        context.register_module(self)

    @property
    def context(self) -> FsdpContext:
        """Return the FSDP context."""
        return self._context

    @property
    def phase(self) -> Phase:
        """Return this module's lifecycle phase."""
        return self._phase

    @phase.setter
    def phase(self, phase: Phase) -> None:
        """Transition this module between its valid lifecycle phases."""
        allowed_transitions = {
            (FsdpModule.Phase.RESTING, FsdpModule.Phase.FORWARD),
            (FsdpModule.Phase.FORWARD, FsdpModule.Phase.RESTING),
            (FsdpModule.Phase.RESTING, FsdpModule.Phase.BACKWARD),
            (FsdpModule.Phase.BACKWARD, FsdpModule.Phase.RESTING),
        }
        if (self._phase, phase) not in allowed_transitions:
            raise RuntimeError(f"Invalid FSDP module phase transition: {self._phase} -> {phase}.")
        self._phase = phase

    @property
    def name(self) -> str:
        """Return this FsdpModule's name."""
        name = self._name
        if name is None:
            raise RuntimeError("FSDP module name has not been initialized.")
        return name

    def is_root(self) -> bool:
        """Return whether this module is an outermost FsdpModule in its context."""
        return self._is_root

    def _register_hooks(
        self, fine_grained: bool = False, skip_backward_callback: bool = False
    ) -> None:
        module = cast(nn.Module, self)
        if fine_grained:
            _register_fine_grained_forward_hooks(self)
            _register_fine_grained_backward_hooks(self)
        else:
            # Use PyTorch's callback module argument instead of capturing self so
            # these hooks do not retain a deleted FSDP module.
            module.register_forward_pre_hook(
                lambda hooked_module, _args: cast(FsdpModule, hooked_module).pre_forward()
            )
            module.register_forward_hook(
                lambda hooked_module, _args, _output: cast(FsdpModule, hooked_module).post_forward()
            )
            module.register_full_backward_pre_hook(
                lambda hooked_module, _grad_output: cast(FsdpModule, hooked_module).pre_backward()
            )
        if self._num_trainable_parameters == 0:
            module.register_full_backward_hook(
                lambda hooked_module, _grad_input, _grad_output: cast(
                    FsdpModule, hooked_module
                ).post_backward()
            )
            return

        if skip_backward_callback:
            return

        # Gradient reduction for trainable parameters is parameter-completion
        # based: once every owned Parameter has accumulated its grad, this
        # FsdpModule can reduce and reshard. Module full-backward hooks can fire
        # before that when module inputs do not require grad.
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            for fsdp_parameter in group.fsdp_parameters:
                fsdp_parameter.unsharded.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _make_grad_hook(self) -> Callable[[nn.Parameter], None]:
        module_ref = ref(self)

        def grad_hook(_parameter: nn.Parameter) -> None:
            module = module_ref()
            if module is None:
                return
            module._num_ready_grad_parameters += 1
            if module._num_ready_grad_parameters == module._num_trainable_parameters:
                module.post_backward()

        return grad_hook

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute and prefetch the next FsdpModule.

        While this FsdpModule computes, we issue the next FsdpModule's all-gather
        on the comm stream, so ``AG_{i+1}`` is launched before ``F_i`` finishes.
        """
        context = self.context
        # This is the first MFSDP hook to run, so finalize the context here once
        # before any module begins communication.
        context.ensure_finalized()
        # A reentrant checkpoint recomputes before the child module's backward-pre
        # hook runs. The active autograd GraphTask identifies that recomputation.
        is_recomputing = self.phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
        if self.phase is not FsdpModule.Phase.BACKWARD:
            self.phase = FsdpModule.Phase.FORWARD
        torch.cuda.nvtx.range_push(self._nvtx_label("forward"))
        self._num_ready_grad_parameters = 0
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
        if not is_recomputing:
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
        self._unshard_parameter_groups()
        if self._unshard_event is not None:
            self.context.current_stream().wait_event(self._unshard_event)

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        # Recomputed parameters are consumed immediately by this module's
        # backward. Keep them materialized to avoid an unnecessary all-gather;
        # post_backward() will reshard them after gradient reduction.
        is_recomputing = self.phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
        if not is_recomputing:
            self._reshard_parameter_groups()
        if self.phase is FsdpModule.Phase.FORWARD:
            self.phase = FsdpModule.Phase.RESTING
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
        # Release on the all-gather stream where unsharded storage was allocated,
        # so no record_stream() call is required for the storage.
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.release_unsharded_storage()
            self._unshard_event = None

    def reshard_parameters(self) -> None:
        """Public API: release all-gathered storage and install DTensor parameters.

        Called by the 1F1B EP overlap schedule's per-layer release hooks
        after compute completes on a sub-module.
        """
        self._reshard_parameter_groups()

    def pre_backward(self) -> None:
        """Prepare full parameters and prefetch the next FsdpModule in backward order."""
        self.phase = FsdpModule.Phase.BACKWARD
        torch.cuda.nvtx.range_push(self._nvtx_label("backward"))
        context = self.context
        current_stream = context.current_stream()
        if self.is_root():
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

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state.

        Any submodule FsdpModule still in the BACKWARD phase (e.g. the 1F1B
        schedule skipped its per-module release) is finalized first.
        """
        # The 1F1B schedule may skip a per-module release; finalize any submodule
        # still in the BACKWARD phase.
        for module in reversed(list(cast(nn.Module, self).modules())):
            if isinstance(module, FsdpModule) and module.phase is FsdpModule.Phase.BACKWARD:
                module.post_backward()
        self._reduce_gradient_groups()
        self._reshard_parameter_groups()
        self._phase = FsdpModule.Phase.RESTING
        torch.cuda.nvtx.range_pop()

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
        so no swap is needed, but finalizing the context here ensures a child
        FSDP unit cannot mistake itself for the root when it executes first.
        """
        self.context.ensure_finalized()

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

    @property
    def parameter_groups(self) -> tuple[FsdpParameterGroup, ...]:
        """Parameter groups owned by this FsdpModule."""
        return self._parameter_groups

    def _nvtx_label(self, phase: Literal["forward", "backward"]) -> str:
        name = self.name if self.name else "<root>"
        return f"MFSDP {name} {phase}"


def _collect_backward_order(module: nn.Module, order: IndexedOrder["FsdpModule"]) -> None:
    """Collect one root's static backward prefetch order."""
    if isinstance(module, FsdpModule):
        order.append(module)

    for child in reversed(list(module.children())):
        _collect_backward_order(child, order)


def _collect_fsdp_children(module: nn.Module, children: set["FsdpModule"]) -> None:
    """Collect the nearest FSDP descendants of ``module``."""
    for child in module.children():
        if isinstance(child, FsdpModule):
            children.add(child)
        else:
            _collect_fsdp_children(child, children)


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
