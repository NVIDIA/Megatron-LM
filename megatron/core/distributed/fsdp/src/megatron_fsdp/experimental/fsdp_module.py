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
from typing import cast

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .parameter_group import ParameterGroup, contained_in_parameter_group
from .placement import MeshAxis, Placements


@dataclasses.dataclass(frozen=True)
class DelayedRelease:
    """A module whose unsharded storage can be released after its consumer event."""

    consumer_event: torch.cuda.Event | None
    module: "FsdpModule"


class FsdpContext:
    """Runtime stream and release scheduler shared by one FSDP subtree."""

    communication_stream: torch.cuda.Stream
    delayed_releases: deque[DelayedRelease]
    release_delay: int

    def __init__(self, device: torch.device, release_delay: int = 2) -> None:
        """Create rank-local stream state for a root FSDP subtree.

        Args:
            device: Device on which this context schedules communication.
            release_delay: Number of unsharded module storages retained while
                normal pre-unshard draining proceeds.
        """
        if release_delay < 1:
            raise ValueError(f"release_delay must be at least 1, got {release_delay}.")
        if device.type != "cuda":
            raise ValueError(f"FSDP stream scheduling requires a CUDA device, got {device}.")

        self.device = device
        self.release_delay = release_delay
        self.delayed_releases = deque()
        self._fsdp_modules: set["FsdpModule"] = set()
        with torch.cuda.device(self.device):
            self.communication_stream = torch.cuda.Stream()

    def add_module(self, module: "FsdpModule") -> None:
        """Track a module using this context."""
        self._fsdp_modules.add(module)

    def remove_module(self, module: "FsdpModule") -> None:
        """Stop tracking a module previously using this context."""
        self._fsdp_modules.discard(module)

    def is_root_module(self, module: "FsdpModule") -> bool:
        """Return whether ``module`` is the outermost FSDP unit in this context."""
        for candidate in tuple(self._fsdp_modules):
            if candidate is module:
                continue
            if _contains_module(cast(nn.Module, candidate), cast(nn.Module, module)):
                return False
        return True

    def enqueue_release(self, module: "FsdpModule") -> None:
        """Queue a module's unsharded storage for delayed release."""
        consumer_event = torch.cuda.current_stream(self.device).record_event()
        self.delayed_releases.append(DelayedRelease(consumer_event=consumer_event, module=module))

    def drain_delayed_releases(self, target_length: int) -> None:
        """Release queued module storages FIFO until the queue reaches ``target_length``."""
        if target_length < 0:
            raise ValueError(f"target_length must be non-negative, got {target_length}.")

        while len(self.delayed_releases) > target_length:
            delayed_release = self.delayed_releases.popleft()
            with torch.cuda.stream(self.communication_stream):
                if delayed_release.consumer_event is not None:
                    self.communication_stream.wait_event(delayed_release.consumer_event)
                delayed_release.module.release_unsharded_storage()


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    _parameter_groups: tuple[ParameterGroup, ...]
    _fsdp_context: FsdpContext
    _ready_grad_parameters: set[nn.Parameter]
    num_training_parameters: int

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        fsdp_context: FsdpContext,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        self._assign_fsdp_context(fsdp_context)
        owned_parameters = _materialize_and_collect_owned_parameters(self, _mesh_device(mesh))
        axis_indices = tuple(_axis_index(mesh, axis) for axis in placements.dp_axes)
        assert axis_indices == tuple(
            range(mesh.ndim)
        ), "FSDP requires dp_axes to match every mesh axis in mesh order for now."
        parameter_groups = [
            ParameterGroup(
                owning_module=self,
                parameters=group_parameters,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=mixed_precision_policy,
            )
            for group_parameters in _group_parameters(owned_parameters)
        ]
        self._parameter_groups = tuple(parameter_groups)
        self._ready_grad_parameters = set()
        self.num_training_parameters = sum(
            len(group.sharded_parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._register_hooks()

    def _assign_fsdp_context(self, fsdp_context: FsdpContext) -> None:
        old_context = getattr(self, "_fsdp_context", None)
        if old_context is fsdp_context:
            return
        if old_context is not None:
            old_context.remove_module(self)
        self._fsdp_context = fsdp_context
        self._fsdp_context.add_module(self)

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
            if len(self._ready_grad_parameters) == self.num_training_parameters:
                self.post_backward()

        return grad_hook

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute."""
        self._ready_grad_parameters.clear()
        self._unshard_on_communication_stream(
            wait_for_current_stream=self._fsdp_context.is_root_module(self)
        )

    def _unshard_on_communication_stream(self, *, wait_for_current_stream: bool) -> None:
        """Run this module's all-gather on the shared communication stream."""
        context = self._fsdp_context
        context.drain_delayed_releases(target_length=context.release_delay - 1)

        current_stream = torch.cuda.current_stream(context.device)
        if wait_for_current_stream:
            context.communication_stream.wait_stream(current_stream)

        with torch.cuda.stream(context.communication_stream):
            self._unshard_parameter_groups()
            ready_event = context.communication_stream.record_event()
        current_stream.wait_event(ready_event)

    def _unshard_parameter_groups(self) -> None:
        for group in self._parameter_groups:
            group.unshard_parameters()

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        self._reshard_parameter_groups()
        self._fsdp_context.enqueue_release(self)
        if self._fsdp_context.is_root_module(self):
            self._fsdp_context.drain_delayed_releases(target_length=0)

    def _reshard_parameter_groups(self) -> None:
        for group in self._parameter_groups:
            group.reshard_parameters()

    def pre_backward(self) -> None:
        """Prepare full parameters for backward compute."""
        self._unshard_on_communication_stream(wait_for_current_stream=False)

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state."""
        context = self._fsdp_context
        self._reduce_gradient_groups()
        self._reshard_parameter_groups()
        context.enqueue_release(self)
        if context.is_root_module(self):
            context.drain_delayed_releases(target_length=0)
        self._ready_grad_parameters.clear()

    def _reduce_gradient_groups(self) -> None:
        for group in self._parameter_groups:
            if group.requires_grad:
                group.reduce_gradients()

    def release_unsharded_storage(self) -> None:
        """Release unsharded storage owned by this FSDP unit."""
        for group in self._parameter_groups:
            group.release_unsharded_storage()

    def parameter_groups(self) -> tuple[ParameterGroup, ...]:
        """Return parameter groups owned by this FSDP unit."""
        return self._parameter_groups


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


def _mesh_device(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(mesh.device_type)


def _materialize_and_collect_owned_parameters(
    root_module: nn.Module, device: torch.device
) -> dict[str, nn.Parameter]:
    parameters: dict[str, nn.Parameter] = {}

    def visit(submodule: nn.Module, submodule_fqn: str) -> None:
        direct_parameters = list(submodule.named_parameters(recurse=False))

        if any(parameter.is_meta for _, parameter in direct_parameters):
            if any(not parameter.is_meta for _, parameter in direct_parameters):
                raise ValueError(
                    f"Module {submodule_fqn!r} mixes meta and non-meta direct parameters. "
                    "Initialize all direct parameters on meta or none of them."
                )
            submodule.to_empty(device=device, recurse=False)
            with torch.no_grad():
                if hasattr(submodule, "reset_parameters"):
                    submodule.reset_parameters()
                elif hasattr(submodule, "_reset_parameters"):
                    submodule._reset_parameters()
                else:
                    raise ValueError(
                        f"Module {submodule_fqn!r} does not have "
                        "reset_parameters or _reset_parameters."
                    )
            # Module.to_empty may replace Parameters, so collect direct parameters again.
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


def _contains_module(parent: nn.Module, child: nn.Module) -> bool:
    return any(module is child for module in parent.modules())
