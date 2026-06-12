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

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .parameter_group import ParameterGroup, contained_in_parameter_group
from .placement import MeshAxis, Placements


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    _parameter_groups: tuple[ParameterGroup, ...]
    _ready_grad_parameters: set[nn.Parameter]
    _num_training_parameters: int

    def __init__(
        self, mesh: DeviceMesh, placements: Placements, mixed_precision_policy: MixedPrecisionPolicy
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
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
        self._num_training_parameters = sum(
            len(group.sharded_parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._register_hooks()

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
        self._ready_grad_parameters.clear()
        for group in self._parameter_groups:
            group.sync_model_weight_from_main_weight()
            group.unshard_parameters()

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        for group in self._parameter_groups:
            group.reshard_parameters()

    def pre_backward(self) -> None:
        """Prepare full parameters for backward compute."""
        for group in self._parameter_groups:
            group.unshard_parameters()

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state."""
        for group in self._parameter_groups:
            if group.requires_grad:
                group.reduce_gradients()
            group.reshard_parameters()
        self._ready_grad_parameters.clear()

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
