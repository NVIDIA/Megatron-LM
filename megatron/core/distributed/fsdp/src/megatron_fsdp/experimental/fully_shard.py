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

"""Minimal experimental per-module Megatron-FSDP implementation."""

import dataclasses
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .dbuffer import DBuffer, MeshAxis, Partial, Placement, Replicate

_CONTAINING_PARAMETER_GROUP_ATTR = "_mfsdp_parameter_group"


@dataclasses.dataclass(frozen=True)
class Placements:
    """Per-mesh-axis placements for parameter, gradient, and optimizer buffers."""

    dp_axes: list[MeshAxis]
    parameter: list[Placement]
    gradient: list[Placement]
    optimizer: list[Placement]

    def __post_init__(self) -> None:
        """Validate placement list lengths."""
        axis_count = len(self.dp_axes)
        for name, placements in (
            ("parameter", self.parameter),
            ("gradient", self.gradient),
            ("optimizer", self.optimizer),
        ):
            if len(placements) != axis_count:
                raise ValueError(f"Expected {axis_count} {name} placements, got {len(placements)}.")


class ParameterGroup:
    """A dtype and requires-grad homogeneous group of FSDP-owned parameters."""

    module: nn.Module
    parameters: dict[str, nn.Parameter]
    _unsharded_parameters: dict[str, nn.Parameter]
    mesh: DeviceMesh
    dtype: torch.dtype
    requires_grad: bool
    main_weight: DBuffer
    model_weight: DBuffer
    main_grad: DBuffer | None
    _full_weight: DBuffer | None
    _full_weight_allocated: bool

    def __init__(
        self,
        module: nn.Module,
        parameters: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        model_weight_placements: Sequence[Placement],
        main_grad_placements: Sequence[Placement],
        main_weight_placements: Sequence[Placement],
        mixed_precision_policy: MixedPrecisionPolicy,
    ) -> None:
        """Create persistent sharded buffers for a group of parameters.

        Args:
            module: Closest FSDP root module that owns this parameter group.
            parameters: Root-module-relative FQNs and their parameters.
            mesh: Device mesh used by the buffers.
            model_weight_placements: Placements for compute-weight storage.
            main_grad_placements: Placements for persistent main gradients.
            main_weight_placements: Placements for optimizer-owned main weights.
            mixed_precision_policy: Precision policy for main weights and gradients.
        """
        if not parameters:
            raise ValueError("ParameterGroup requires at least one parameter.")

        # Python dicts preserve insertion order, so values() defines the stable
        # tensor order used by each DBuffer built from this group.
        original_parameters = parameters
        self.module = module
        self.parameters = {}
        self._unsharded_parameters = {}
        self.mesh = mesh
        first_parameter = next(iter(original_parameters.values()))
        self.dtype = first_parameter.dtype
        self.requires_grad = first_parameter.requires_grad
        for name, parameter in original_parameters.items():
            if parameter.dtype != self.dtype:
                raise ValueError(
                    f"Expected parameter {name!r} to have dtype {self.dtype}, got {parameter.dtype}."
                )
            if parameter.requires_grad != self.requires_grad:
                raise ValueError(
                    f"Expected parameter {name!r} to have requires_grad={self.requires_grad}, "
                    f"got {parameter.requires_grad}."
                )
        main_params_dtype = mixed_precision_policy.main_params_dtype
        if main_params_dtype is None:
            raise ValueError(
                "experimental FSDP requires main_params_dtype to be specified explicitly."
            )
        main_grads_dtype = mixed_precision_policy.main_grads_dtype
        if main_grads_dtype is None:
            main_grads_dtype = self.dtype
        self.main_weight = DBuffer(
            mesh=self.mesh,
            placements=main_weight_placements,
            tensor_shapes=[parameter.shape for parameter in original_parameters.values()],
            dtype=main_params_dtype,
            device=_mesh_device(self.mesh),
        )
        for index, parameter in enumerate(original_parameters.values()):
            if parameter.is_meta:
                continue
            self._copy_full_tensor_to_buffer(
                self.main_weight,
                index,
                parameter.detach().to(
                    dtype=main_params_dtype, device=self.main_weight.local_buffer.device
                ),
            )
        self.model_weight = self._make_model_weight(tuple(model_weight_placements))
        self.main_grad = (
            DBuffer(
                mesh=self.mesh,
                placements=main_grad_placements,
                tensor_shapes=self.main_weight.layout.tensor_shapes,
                dtype=main_grads_dtype,
                device=self.main_weight.local_buffer.device,
            )
            if self.requires_grad
            else None
        )
        self._full_weight: DBuffer | None = None
        self._full_weight_allocated = False

        for index, (name, original_parameter) in enumerate(original_parameters.items()):
            sharded_parameter = nn.Parameter(
                self.main_weight.get_dtensor(index), requires_grad=original_parameter.requires_grad
            )
            setattr(sharded_parameter, _CONTAINING_PARAMETER_GROUP_ATTR, self)
            self.parameters[name] = sharded_parameter
            self._set_module_parameter(name, sharded_parameter)

    def _make_model_weight(self, placements: Sequence[Placement]) -> DBuffer:
        model_weight = self.main_weight.redistribute(placements)
        if model_weight.local_buffer.dtype != self.dtype:
            converted = DBuffer(
                mesh=model_weight.mesh,
                placements=model_weight.placements,
                tensor_shapes=model_weight.layout.tensor_shapes,
                dtype=self.dtype,
                device=model_weight.local_buffer.device,
            )
            converted.local_buffer.copy_(model_weight.local_buffer.to(dtype=self.dtype))
            return converted
        return model_weight

    def _copy_full_tensor_to_buffer(
        self, buffer: DBuffer, index: int, tensor: torch.Tensor
    ) -> None:
        """Copy this rank's overlapping slice from a full tensor into a DBuffer."""
        tensor = tensor.contiguous().view(-1)
        shape = buffer.layout.tensor_shapes[index]
        tensor_start = buffer.layout.tensor_to_offset[index]
        tensor_end = tensor_start + shape.numel()
        buffer_start = buffer.offset
        buffer_end = buffer.offset + buffer.local_buffer.numel()
        overlap_start = max(tensor_start, buffer_start)
        overlap_end = min(tensor_end, buffer_end)
        if overlap_end <= overlap_start:
            return
        local_numel = overlap_end - overlap_start
        buffer.local_buffer.narrow(0, overlap_start - buffer_start, local_numel).copy_(
            tensor.narrow(0, overlap_start - tensor_start, local_numel)
        )

    def _set_module_parameter(self, name: str, parameter: nn.Parameter) -> None:
        module, parameter_name = _get_parameter_owner(self.module, name)
        module._parameters[parameter_name] = parameter

    def refresh_model_weight(self) -> None:
        """Refresh compute-weight storage from the current main weights."""
        self.model_weight = self._make_model_weight(self.model_weight.placements)

    def unshard_parameters(self) -> None:
        """Install full parameters for local compute."""
        self.refresh_model_weight()
        self._allocate_full_weight()
        assert self._full_weight is not None
        self._ensure_unsharded_parameters()
        for name, parameter in self._unsharded_parameters.items():
            parameter.grad = None
            self._set_module_parameter(name, parameter)

    def reshard_parameters(self) -> None:
        """Install sharded DTensor parameters on the owning modules."""
        for name, parameter in self.parameters.items():
            self._set_module_parameter(name, parameter)
        self._release_full_weight_storage()

    def _ensure_unsharded_parameters(self) -> None:
        """Create stable full-size autograd leaf parameters."""
        if self._unsharded_parameters:
            return
        assert self._full_weight is not None
        for index, (name, sharded_parameter) in enumerate(self.parameters.items()):
            self._unsharded_parameters[name] = nn.Parameter(
                self._full_weight.get_tensor(index), requires_grad=sharded_parameter.requires_grad
            )

    def _allocate_full_weight(self) -> None:
        """Ensure the stable full-weight storage is allocated and current."""
        if self._full_weight is None:
            self._full_weight = self.model_weight.redistribute([Replicate()] * self.mesh.ndim)
            self._full_weight_allocated = True
            return
        if self._full_weight_allocated:
            return

        self._resize_full_weight_storage(self._full_weight.local_buffer.numel())
        refreshed = self.model_weight.redistribute([Replicate()] * self.mesh.ndim)
        with torch.autograd._unsafe_preserve_version_counter(self._full_weight.local_buffer):
            self._full_weight.local_buffer.copy_(refreshed.local_buffer)
        self._full_weight_allocated = True

    def _release_full_weight_storage(self) -> None:
        """Free full-weight storage without replacing the Storage object."""
        if self._full_weight is None or not self._full_weight_allocated:
            return
        # Non-leaf parameter views saved by autograd keep their Storage object.
        # Retaining that object and resizing it back before backward avoids
        # stale-storage failures while still releasing the allocation.
        self._resize_full_weight_storage(0)
        self._full_weight_allocated = False

    def _resize_full_weight_storage(self, numel: int) -> None:
        assert self._full_weight is not None
        with torch.autograd._unsafe_preserve_version_counter(self._full_weight.local_buffer):
            self._full_weight.local_buffer.untyped_storage().resize_(
                numel * self._full_weight.local_buffer.element_size()
            )

    def reduce_gradients(self, average: bool = True) -> None:
        """Reduce full local gradients into the persistent sharded gradient buffer."""
        if not self.requires_grad:
            return
        assert self.main_grad is not None

        accumulate = self.has_sharded_grad()
        full_grads: list[torch.Tensor] = []
        for name, parameter in self._unsharded_parameters.items():
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient for FSDP parameter {name!r}.")
            full_grads.append(
                parameter.grad.detach().to(dtype=self.main_grad.local_buffer.dtype).contiguous()
            )

        partial_grad = DBuffer.distribute_tensors(
            full_grads, mesh=self.mesh, placements=[Partial()] * self.mesh.ndim
        )
        reduced_grad = partial_grad.redistribute(self.main_grad.placements)
        if average:
            reduced_grad.local_buffer.div_(self.mesh.size())

        if accumulate:
            self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
        else:
            self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)

        for parameter in self._unsharded_parameters.values():
            parameter.grad = None
        self.install_sharded_gradients()

    def has_sharded_grad(self) -> bool:
        """Return whether persistent sharded gradients are currently materialized."""
        return any(parameter.grad is not None for parameter in self.parameters.values())

    def install_sharded_gradients(self) -> None:
        """Install sharded DTensor gradients backed by main_grad."""
        if self.main_grad is None:
            return
        for index, parameter in enumerate(self.parameters.values()):
            parameter.grad = self.main_grad.get_dtensor(index)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear persistent and parameter gradients."""
        if self.main_grad is not None:
            self.main_grad.local_buffer.zero_()
        if set_to_none:
            for parameter in self.parameters.values():
                parameter.grad = None
        else:
            self.install_sharded_gradients()
        for parameter in self._unsharded_parameters.values():
            parameter.grad = None

    def unsharded_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return temporary full-size parameters used for autograd."""
        return tuple(self._unsharded_parameters.values())


class FsdpModule:
    """Mixin attached to modules managed by the minimal experimental FSDP path."""

    _parameter_groups: tuple[ParameterGroup, ...]
    _ready_grad_params: set[nn.Parameter]
    _registered_grad_param_ids: set[int]
    _trainable_param_count: int

    def __init__(self, parameter_groups: Sequence[ParameterGroup]) -> None:
        """Initialize mixin runtime state on an already-constructed module."""
        self._parameter_groups = tuple(parameter_groups)
        self._ready_grad_params: set[nn.Parameter] = set()
        self._registered_grad_param_ids: set[int] = set()
        self._trainable_param_count = sum(
            len(group.parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.register_forward_pre_hook(lambda _module, _args: self.pre_forward())
        self.register_forward_hook(lambda _module, _args, _output: self.post_forward())
        self.register_full_backward_pre_hook(lambda _module, _grad_output: self.pre_backward())

    def _register_grad_hooks(self) -> None:
        """Register post-accumulate hooks on full-size autograd leaf parameters."""
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            for parameter in group.unsharded_parameters():
                parameter_id = id(parameter)
                if parameter_id in self._registered_grad_param_ids:
                    continue
                parameter.register_post_accumulate_grad_hook(self._make_grad_hook(parameter))
                self._registered_grad_param_ids.add(parameter_id)

    def _make_grad_hook(self, parameter: nn.Parameter) -> Callable[[nn.Parameter], None]:
        def grad_hook(_parameter: nn.Parameter) -> None:
            self._ready_grad_params.add(parameter)
            if len(self._ready_grad_params) == self._trainable_param_count:
                self.post_backward()

        return grad_hook

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute."""
        self._ready_grad_params.clear()
        for group in self._parameter_groups:
            group.unshard_parameters()
        self._register_grad_hooks()

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
            group.reduce_gradients()
            group.reshard_parameters()
            group.install_sharded_gradients()
        self._ready_grad_params.clear()

    def parameter_groups(self) -> tuple[ParameterGroup, ...]:
        """Return parameter groups owned by this FSDP unit."""
        return self._parameter_groups


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
    init_model_with_meta_device: bool = False,
) -> None:
    """Shard one module as an experimental per-module FSDP unit.

    Args:
        module: Module whose currently unowned parameters become this FSDP unit.
        mesh: Device mesh used for sharding.
        placements: Parameter, gradient, and optimizer placements.
        mixed_precision_policy: Optional precision policy. Defaults to FP32 main weights
            and parameter-dtype main gradients.
        init_model_with_meta_device: If true, initialize owned meta parameters by
            calling their direct module's reset_parameters() or _reset_parameters().
    """
    if isinstance(module, FsdpModule):
        raise ValueError("This module is already managed by experimental FSDP.")

    mixed_precision_policy = mixed_precision_policy or MixedPrecisionPolicy()
    model_weight_placements = _placements_in_mesh_order(
        mesh, placements.dp_axes, placements.parameter
    )
    main_grad_placements = _placements_in_mesh_order(mesh, placements.dp_axes, placements.gradient)
    main_weight_placements = _placements_in_mesh_order(
        mesh, placements.dp_axes, placements.optimizer
    )
    owned_parameters = _collect_owned_parameters(module)
    if (
        any(parameter.is_meta for parameter in owned_parameters.values())
        and not init_model_with_meta_device
    ):
        raise ValueError(
            "experimental FSDP found meta parameters. Pass init_model_with_meta_device=True "
            "to initialize them with reset_parameters()/_reset_parameters()."
        )
    grouped_parameters = _group_parameters(owned_parameters)
    parameter_groups = [
        ParameterGroup(
            module,
            group_parameters,
            mesh=mesh,
            model_weight_placements=model_weight_placements,
            main_grad_placements=main_grad_placements,
            main_weight_placements=main_weight_placements,
            mixed_precision_policy=mixed_precision_policy,
        )
        for group_parameters in grouped_parameters
    ]
    if init_model_with_meta_device:
        _reset_owned_meta_modules(module, owned_parameters)
        for group in parameter_groups:
            group.refresh_model_weight()

    _attach_mixin(module)
    assert isinstance(module, FsdpModule)
    FsdpModule.__init__(module, parameter_groups=parameter_groups)


def _axis_index(mesh: DeviceMesh, axis: MeshAxis) -> int:
    if isinstance(axis, int):
        if axis < 0:
            axis += mesh.ndim
        if axis < 0 or axis >= mesh.ndim:
            raise ValueError(f"Mesh axis {axis} is out of bounds for mesh ndim {mesh.ndim}.")
        return axis

    dim_names = mesh.mesh_dim_names
    if dim_names is None or axis not in dim_names:
        raise ValueError(f"Mesh axis {axis!r} is not present in mesh dim names {dim_names}.")
    return dim_names.index(axis)


def _placements_in_mesh_order(
    mesh: DeviceMesh, dp_axes: Sequence[MeshAxis], placements: Sequence[Placement]
) -> tuple[Placement, ...]:
    if len(dp_axes) != mesh.ndim:
        raise ValueError(
            "experimental fully_shard currently requires placements for every mesh axis: "
            f"mesh ndim is {mesh.ndim}, got {len(dp_axes)} axes."
        )
    result: list[Placement | None] = [None] * mesh.ndim
    for axis, placement in zip(dp_axes, placements, strict=True):
        axis_index = _axis_index(mesh, axis)
        if result[axis_index] is not None:
            raise ValueError(f"Duplicate placement for mesh axis {axis!r}.")
        result[axis_index] = placement
    if any(placement is None for placement in result):
        raise ValueError("Missing placement for at least one mesh axis.")
    return tuple(placement for placement in result if placement is not None)


def _mesh_device(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(mesh.device_type)


def _collect_owned_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    child_prefixes = [
        f"{name}."
        for name, child in module.named_modules()
        if name and isinstance(child, FsdpModule)
    ]
    parameters: dict[str, nn.Parameter] = {}
    for module_name, child in module.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if any(prefix.startswith(child_prefix) for child_prefix in child_prefixes):
            continue
        for parameter_name, parameter in child.named_parameters(recurse=False):
            name = f"{prefix}{parameter_name}"
            if hasattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR):
                raise ValueError(
                    f"Parameter {name!r} is already owned by an experimental FSDP unit."
                )
            parameters[name] = parameter
    if not parameters:
        raise ValueError("experimental fully_shard requires at least one unowned parameter.")
    return parameters


def _group_parameters(parameters: dict[str, nn.Parameter]) -> list[dict[str, nn.Parameter]]:
    grouped: dict[tuple[torch.dtype, bool], dict[str, nn.Parameter]] = {}
    for name, parameter in parameters.items():
        key = (parameter.dtype, parameter.requires_grad)
        grouped.setdefault(key, {})[name] = parameter
    return [grouped[key] for key in grouped]


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = name.rpartition(".")
    owner = module.get_submodule(module_name) if separator else module
    return owner, parameter_name


def _reset_owned_meta_modules(
    module: nn.Module, original_parameters: dict[str, nn.Parameter]
) -> None:
    """Initialize modules that originally owned meta parameters."""
    modules_to_reset: dict[int, tuple[str, nn.Module]] = {}
    for name, parameter in original_parameters.items():
        if not parameter.is_meta:
            continue
        owner_module, _ = _get_parameter_owner(module, name)
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        modules_to_reset.setdefault(id(owner_module), (module_name, owner_module))

    for module_name, module in modules_to_reset.values():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        elif hasattr(module, "_reset_parameters"):
            module._reset_parameters()
        else:
            raise ValueError(
                f"[init_model_with_meta_device=True] Module {module_name!r} does not have "
                "reset_parameters or _reset_parameters."
            )


def _attach_mixin(module: nn.Module) -> None:
    if isinstance(module, FsdpModule):
        return
    module_cls = module.__class__
    fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
    module.__class__ = fsdp_cls
