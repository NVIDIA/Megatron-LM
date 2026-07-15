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

import dataclasses
from collections import deque
from collections.abc import Callable
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .parameter_group import FsdpParameterGroup, contained_in_parameter_group
from .placement import MeshAxis, Placements


@dataclasses.dataclass(frozen=True)
class DelayedRelease:
    """A module whose unsharded storage can be released after its consumer event."""

    consumer_event: torch.cuda.Event | None
    module: "FsdpModule"


class FsdpContext:
    """Runtime state, streams, and release scheduler shared by one FSDP subtree."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    reduce_scatter_group: dist.ProcessGroup
    delayed_releases: deque[DelayedRelease]
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because it can be detected when ``model_weight``, after syncing
    # from ``main_weight``, has placements different from ``Placements.optimizer``.
    is_last_microbatch: bool
    root_module: "FsdpModule"

    def __init__(self, device: torch.device, root_module: "FsdpModule") -> None:
        """Create rank-local runtime state for a root FSDP subtree.

        Args:
            device: Device on which this context schedules communication.
            root_module: Outermost module that owns this context.
        """
        self.root_module = root_module
        self.is_last_microbatch = True
        self.delayed_releases = deque()
        self._post_backward_callback_queued = False
        with torch.cuda.device(device):
            self.allgather_stream = torch.cuda.Stream()
            self.reduce_scatter_stream = torch.cuda.Stream()
        # Dedicated NCCL communicator for reduce-scatter, distinct from the mesh's
        # all-gather group. A separate communicator lets an eager reduce-scatter run
        # concurrently with the next unit's all-gather instead of contending on one
        # comm's ordered stream -- which is what makes it safe to drop the delayed
        # reduction and launch reduce-scatter immediately in post_backward. new_group
        # is collective, so every rank reaches this on the first forward through the
        # root. Prototype: 1D DP mesh only (the general case needs one dedicated group
        # per mesh-axis subgroup, mirroring DeviceMesh construction).
        mesh = root_module._parameter_groups[0].mesh
        if mesh.ndim != 1:
            raise NotImplementedError(
                "Prototype dedicated reduce-scatter communicator supports a 1D DP mesh "
                f"only; got a {mesh.ndim}D mesh."
            )
        self.reduce_scatter_group = dist.new_group(
            ranks=dist.get_process_group_ranks(mesh.get_group(0))
        )

    def enqueue_release(self, module: "FsdpModule") -> None:
        """Queue a module's unsharded storage for delayed release."""
        consumer_event = torch.cuda.current_stream(self.allgather_stream.device).record_event()
        self.delayed_releases.append(DelayedRelease(consumer_event=consumer_event, module=module))

    def drain_delayed_releases(self, target_length: int) -> None:
        """Release queued module storages FIFO until the queue reaches ``target_length``."""
        if target_length < 0:
            raise ValueError(f"target_length must be non-negative, got {target_length}.")

        while len(self.delayed_releases) > target_length:
            delayed_release = self.delayed_releases.popleft()
            with torch.cuda.stream(self.allgather_stream):
                if delayed_release.consumer_event is not None:
                    self.allgather_stream.wait_event(delayed_release.consumer_event)
                delayed_release.module.release_unsharded_storage()

    def queue_post_backward_callback(self) -> None:
        """Queue an end-of-backward callback so the optimizer waits for reduce-scatter."""
        if self._post_backward_callback_queued:
            return

        self._post_backward_callback_queued = True
        try:
            torch.autograd.Variable._execution_engine.queue_callback(self.finalize_reductions)
        except RuntimeError as error:
            self._post_backward_callback_queued = False
            if str(error) != "Final callbacks can only be installed during backward pass.":
                raise
            self.finalize_reductions()

    def finalize_reductions(self) -> None:
        """Order the default stream (the optimizer's) after all eager reduce-scatters.

        Reduce-scatters are launched eagerly in ``post_backward`` on
        ``reduce_scatter_stream``; their input and output buffers are allocated on and
        consumed by that stream, so the CUDA caching allocator keeps the storage alive
        until each collective completes -- no explicit pending-buffer retention needed.
        This single stream barrier makes the default stream, where the optimizer reads
        the reduced gradients, wait for those reductions to finish.
        """
        try:
            default_stream = torch.cuda.current_stream(self.reduce_scatter_stream.device)
            default_stream.wait_stream(self.reduce_scatter_stream)
        finally:
            self._post_backward_callback_queued = False


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    # Name relative to the root FSDP module from named_modules().
    # Root uses "" and None means uninitialized.
    _name: str | None
    _parameter_groups: tuple[FsdpParameterGroup, ...]
    _context: FsdpContext | None
    _ready_grad_parameters: set[nn.Parameter]
    _num_training_parameters: int

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        use_symm_mem: bool = False,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        self._context = None
        self._name = None
        owned_parameters = _collect_owned_parameters(self)
        axis_indices = tuple(_axis_index(mesh, axis) for axis in placements.dp_axes)
        assert axis_indices == tuple(
            range(mesh.ndim)
        ), "FSDP requires dp_axes to match every mesh axis in mesh order for now."
        parameter_groups = [
            FsdpParameterGroup(
                owning_module=self,
                parameters=group_parameters,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=mixed_precision_policy,
                use_symm_mem=use_symm_mem,
            )
            for group_parameters in _group_parameters(owned_parameters)
        ]
        self._parameter_groups = tuple(parameter_groups)
        self._ready_grad_parameters = set()
        self._num_training_parameters = sum(
            len(group.sharded_parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._register_hooks()

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

        context = FsdpContext(device=self._parameter_groups[0].main_weight.device, root_module=self)
        for submodule_name, submodule in cast(nn.Module, self).named_modules():
            if not isinstance(submodule, FsdpModule):
                continue
            if submodule._context is not None:
                raise RuntimeError(
                    "FSDP context is already initialized for a descendant module. "
                    "Run forward through the root FSDP module first."
                )
            submodule._context = context
            submodule._name = submodule_name

    @property
    def context(self) -> FsdpContext:
        """Return the initialized runtime context."""
        assert self._context is not None
        return self._context

    @property
    def name(self) -> str:
        """Return this FSDP unit's name."""
        name = self._name
        if name is None:
            raise RuntimeError("FSDP module name has not been initialized.")
        return name

    def is_root(self) -> bool:
        """Return whether this module is the outermost FSDP unit in its context."""
        return self.context.root_module is self

    def _register_hooks(self) -> None:
        module = cast(nn.Module, self)
        module.register_forward_pre_hook(lambda _module, _args: self.pre_forward())
        module.register_forward_hook(lambda _module, _args, _output: self.post_forward())
        module.register_full_backward_pre_hook(lambda _module, _grad_output: self.pre_backward())
        # Gradient reduction is parameter-completion based: once every owned
        # Parameter has accumulated its grad, this FSDP unit can reduce and
        # reshard. Module full-backward hooks can fire before that when module
        # inputs do not require grad.
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            for parameter in group.unsharded_parameters:
                parameter.register_post_accumulate_grad_hook(self._make_grad_hook(parameter))

    def _make_grad_hook(self, parameter: nn.Parameter) -> Callable[[nn.Parameter], None]:
        def grad_hook(_parameter: nn.Parameter) -> None:
            self._ready_grad_parameters.add(parameter)
            if len(self._ready_grad_parameters) == self._num_training_parameters:
                self.post_backward()

        return grad_hook

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute."""
        self._lazy_init_context()
        torch.cuda.nvtx.range_push(self._nvtx_label("forward"))
        self._ready_grad_parameters.clear()
        if self.is_root():
            allgather_stream = self.context.allgather_stream
            allgather_stream.wait_stream(torch.cuda.current_stream(allgather_stream.device))
        self._unshard_parameter_groups(sync_model_weight=True)

    def _unshard_parameter_groups(self, *, sync_model_weight: bool) -> None:
        """Materialize full parameters for this FSDP unit."""
        self.context.drain_delayed_releases(target_length=1)

        allgather_stream = self.context.allgather_stream
        current_stream = torch.cuda.current_stream(allgather_stream.device)

        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                if sync_model_weight:
                    # TODO: After NVIDIA/Megatron-LM#5411 lands, move this sync to the
                    # optimizer post-step hook instead of running it every microbatch.
                    group.sync_model_weight_from_main_weight()
                group.unshard_parameters()
        current_stream.wait_stream(allgather_stream)

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        self._reshard_parameter_groups()
        self.context.enqueue_release(self)
        if self.is_root():
            self.context.drain_delayed_releases(target_length=0)
        torch.cuda.nvtx.range_pop()

    def _reshard_parameter_groups(self) -> None:
        for group in self._parameter_groups:
            group.reshard_parameters()

    def pre_backward(self) -> None:
        """Prepare full parameters for backward compute."""
        torch.cuda.nvtx.range_push(self._nvtx_label("backward"))
        self._unshard_parameter_groups(sync_model_weight=False)

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state."""
        self._reduce_gradient_groups()
        self._reshard_parameter_groups()
        self.context.enqueue_release(self)
        if self.is_root():
            self.context.drain_delayed_releases(target_length=0)
        self._ready_grad_parameters.clear()
        torch.cuda.nvtx.range_pop()

    def _reduce_gradient_groups(self) -> None:
        # Eagerly launch each group's reduce-scatter on the dedicated reduce-scatter
        # communicator/stream as soon as its gradients are packed. Because that comm is
        # separate from the all-gather group, an eager reduce-scatter does not serialize
        # against the next unit's all-gather, so no deferral to the next pre_backward and
        # no prepared/pending buffer bookkeeping are needed -- only the end-of-backward
        # barrier in finalize_reductions. Delayed *releases* are unchanged.
        context = self.context
        default_stream = torch.cuda.current_stream(context.reduce_scatter_stream.device)
        scheduled_reduction = False
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            with torch.cuda.stream(context.reduce_scatter_stream):
                partial_grad = group.allocate_partial_grad_buffer()

            # Pack on the default stream (where grads are produced), so reads of
            # parameter.grad and the subsequent `.grad = None` stay same-stream.
            default_stream.wait_stream(context.reduce_scatter_stream)
            group.copy_gradients_to_partial_buffer(partial_grad)

            # Reduce-scatter waits for the pack, then runs eagerly on the reduce-scatter
            # stream. partial_grad was allocated on that stream, so dropping the reference
            # here is safe: the caching allocator retains the storage until the collective
            # completes.
            context.reduce_scatter_stream.wait_stream(default_stream)
            with torch.cuda.stream(context.reduce_scatter_stream):
                group.reduce_partial_gradients(
                    partial_grad, reduce_group=context.reduce_scatter_group
                )
            scheduled_reduction = True

        if scheduled_reduction:
            context.queue_post_backward_callback()

    def release_unsharded_storage(self) -> None:
        """Release unsharded storage owned by this FSDP unit."""
        for group in self._parameter_groups:
            group.release_unsharded_storage()

    @property
    def parameter_groups(self) -> tuple[FsdpParameterGroup, ...]:
        """Parameter groups owned by this FSDP unit."""
        return self._parameter_groups

    def _nvtx_label(self, phase: Literal["forward", "backward"]) -> str:
        name = self.name if self.name else "<root>"
        return f"MFSDP {name} {phase}"


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
        direct_parameters = list(submodule.named_parameters(recurse=False))

        for local_parameter_name, parameter in direct_parameters:
            parameter_fqn = (
                f"{submodule_fqn}.{local_parameter_name}" if submodule_fqn else local_parameter_name
            )
            if contained_in_parameter_group(parameter):
                raise ValueError(f"Parameter {parameter_fqn!r} is already owned by an FSDP unit.")
            parameters[parameter_fqn] = parameter

        for child_name, child_module in submodule.named_children():
            if isinstance(child_module, FsdpModule):
                continue
            child_fqn = f"{submodule_fqn}.{child_name}" if submodule_fqn else child_name
            visit(child_module, child_fqn)

    visit(root_module, "")
    if not parameters:
        raise ValueError("fully_shard requires at least one unowned parameter.")
    return parameters


def _group_parameters(parameters: dict[str, nn.Parameter]) -> list[dict[str, nn.Parameter]]:
    grouped: dict[tuple[torch.dtype, bool], dict[str, nn.Parameter]] = {}
    for name, parameter in parameters.items():
        key = (parameter.dtype, parameter.requires_grad)
        grouped.setdefault(key, {})[name] = parameter
    return [grouped[key] for key in grouped]
