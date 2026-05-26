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

"""Minimal per-module Megatron-FSDP implementation."""

import dataclasses
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
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

    owning_module: nn.Module
    parameter_names: tuple[str, ...]
    sharded_parameters: tuple[nn.Parameter, ...]
    unsharded_parameters: tuple[nn.Parameter, ...]
    mesh: DeviceMesh
    dtype: torch.dtype
    requires_grad: bool
    main_weight: DBuffer
    model_weight: DBuffer
    main_grad: DBuffer | None
    _unsharded_model_weight: DBuffer

    def __init__(
        self,
        owning_module: nn.Module,
        parameters: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
    ) -> None:
        """Create persistent sharded buffers for a group of parameters.

        Args:
            owning_module: Closest FSDP root module that owns this parameter group.
            parameters: Root-module-relative FQNs and their parameters.
            mesh: Device mesh used for all DBuffer storage in this version.
            placements: Parameter, gradient, and optimizer placements.
            mixed_precision_policy: Precision policy for main weights and gradients.
        """
        # ---------------------------------------------------------------------
        # Validate group inputs and mesh contract
        # ---------------------------------------------------------------------
        if not parameters:
            raise ValueError("ParameterGroup requires at least one parameter.")

        axis_indices = tuple(_axis_index(mesh, axis) for axis in placements.dp_axes)
        assert axis_indices == tuple(
            range(mesh.ndim)
        ), "FSDP requires dp_axes to match every mesh axis in mesh order for now."

        # ---------------------------------------------------------------------
        # Record shared metadata
        # ---------------------------------------------------------------------
        self.owning_module = owning_module
        self.mesh = mesh
        self.parameter_names = tuple(parameters)
        first_parameter = next(iter(parameters.values()))
        self.dtype = first_parameter.dtype
        self.requires_grad = first_parameter.requires_grad
        for name, parameter in parameters.items():
            if parameter.is_meta:
                raise ValueError(
                    f"Expected parameter {name!r} to be materialized before "
                    "ParameterGroup construction."
                )
            if parameter.dtype != self.dtype:
                raise ValueError(
                    f"Expected parameter {name!r} to have dtype {self.dtype}, "
                    f"got {parameter.dtype}."
                )
            if parameter.requires_grad != self.requires_grad:
                raise ValueError(
                    f"Expected parameter {name!r} to have requires_grad={self.requires_grad}, "
                    f"got {parameter.requires_grad}."
                )

        # ---------------------------------------------------------------------
        # Initialize persistent DBuffers
        # ---------------------------------------------------------------------
        # Python dicts preserve insertion order, so parameter_names and
        # parameters.values() define the same stable DBuffer tensor order.
        tensor_shapes = tuple(parameter.shape for parameter in parameters.values())
        main_weight_dtype = mixed_precision_policy.main_params_dtype or torch.float32
        self.main_weight = DBuffer.distribute_tensors(
            (parameter.to(dtype=main_weight_dtype) for parameter in parameters.values()),
            mesh=self.mesh,
            placements=placements.optimizer,
        )

        self._unsharded_model_weight = DBuffer(
            mesh=self.mesh,
            placements=[Replicate()] * self.mesh.ndim,
            tensor_shapes=tensor_shapes,
            dtype=self.dtype,
            device=self.main_weight.local_buffer.device,
        )
        if main_weight_dtype == self.dtype and placements.optimizer == placements.parameter:
            self.model_weight = self.main_weight
        else:
            self.model_weight = DBuffer(
                mesh=self.mesh,
                placements=placements.parameter,
                tensor_shapes=tensor_shapes,
                dtype=self.dtype,
                device=self.main_weight.local_buffer.device,
            )

        self.main_grad = None
        if self.requires_grad:
            grad_dtype = mixed_precision_policy.main_grads_dtype or self.dtype
            self.main_grad = DBuffer(
                mesh=self.mesh,
                placements=placements.gradient,
                tensor_shapes=self.main_weight.layout.tensor_shapes,
                dtype=grad_dtype,
                device=self.main_weight.local_buffer.device,
            )
            assert self.main_grad.layout == self.main_weight.layout, (
                "main_grad is built from main_weight tensor shapes on the same mesh, "
                "and DBuffer layouts are deterministic from those shapes and mesh size."
            )
            if self.main_grad.placements != self.main_weight.placements:
                raise ValueError(
                    "FSDP temporarily requires main_grad and main_weight to have the same "
                    "placements until HSDP/HFSDP support is implemented. "
                    f"Got main_grad placements {self.main_grad.placements} and "
                    f"main_weight placements {self.main_weight.placements}."
                )

        # ---------------------------------------------------------------------
        # Build parameter tuples for module swapping
        # ---------------------------------------------------------------------
        sharded_parameters: list[nn.Parameter] = []
        unsharded_parameters: list[nn.Parameter] = []
        grad_dtype = self.main_grad.local_buffer.dtype if self.requires_grad else None
        for index, parameter in enumerate(parameters.values()):
            parameter.data = self._unsharded_model_weight.get_tensor(index)
            parameter.grad = None
            if grad_dtype:
                parameter.grad_dtype = grad_dtype
            setattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, self)
            unsharded_parameters.append(parameter)

            sharded_parameter = nn.Parameter(
                self.main_weight.get_dtensor(index), requires_grad=parameter.requires_grad
            )
            if grad_dtype:
                sharded_parameter.grad_dtype = grad_dtype
            setattr(sharded_parameter, _CONTAINING_PARAMETER_GROUP_ATTR, self)
            sharded_parameters.append(sharded_parameter)
        self.sharded_parameters = tuple(sharded_parameters)
        self.unsharded_parameters = tuple(unsharded_parameters)

        # ---------------------------------------------------------------------
        # Install resting sharded parameters
        # ---------------------------------------------------------------------
        self._switch_to_sharded_parameters()
        self._unsharded_model_weight.release_storage()

    def _set_module_parameters(self, parameters: tuple[nn.Parameter, ...]) -> None:
        for name, parameter in zip(self.parameter_names, parameters, strict=True):
            module, parameter_name = _get_parameter_owner(self.owning_module, name)
            module._parameters[parameter_name] = parameter

    def _switch_to_sharded_parameters(self) -> None:
        self._set_module_parameters(self.sharded_parameters)

    def _switch_to_unsharded_parameters(self) -> None:
        self._set_module_parameters(self.unsharded_parameters)

    def sync_model_weight_from_main_weight(self) -> None:
        """Refresh compute weights from optimizer weights."""
        if self.main_weight is self.model_weight:
            return

        self.main_weight.cast(self.model_weight.local_buffer.dtype).redistribute(
            self.model_weight.placements, out=self.model_weight
        )

    def unshard_parameters(self) -> None:
        """Install full parameters for local compute."""
        self.sync_model_weight_from_main_weight()
        self._unsharded_model_weight.reallocate_storage()
        # This buffer backs unsharded Parameters whose views may be saved by autograd.
        # Materializing FSDP-managed storage should not look like a user mutation.
        with torch.autograd._unsafe_preserve_version_counter(
            self._unsharded_model_weight.local_buffer
        ):
            self.model_weight.redistribute(
                self._unsharded_model_weight.placements, out=self._unsharded_model_weight
            )
        self._switch_to_unsharded_parameters()

    def reshard_parameters(self) -> None:
        """Install sharded DTensor parameters on the owning modules."""
        self._switch_to_sharded_parameters()
        self._unsharded_model_weight.release_storage()

    def reduce_gradients(self) -> None:
        """Reduce full local gradients into sharded parameter gradients."""
        assert self.main_grad is not None

        def has_grad(parameters: Iterable[nn.Parameter]) -> bool:
            has_any_grad = False
            has_any_missing_grad = False
            for parameter in parameters:
                if parameter.grad is None:
                    has_any_missing_grad = True
                else:
                    has_any_grad = True
            if has_any_grad and has_any_missing_grad:
                raise RuntimeError("FSDP sharded gradients must be either all set or all None.")
            return has_any_grad

        grads: list[torch.Tensor] = []
        for name, parameter in zip(self.parameter_names, self.unsharded_parameters, strict=True):
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient for FSDP parameter {name!r}.")
            assert parameter.grad.dtype == self.main_grad.local_buffer.dtype, (
                "FSDP unsharded parameter grad dtype should be guaranteed by grad_dtype: "
                f"parameter {name!r} has grad dtype {parameter.grad.dtype}, "
                f"expected main_grad dtype {self.main_grad.local_buffer.dtype}."
            )
            grads.append(parameter.grad)

        partial_grad = DBuffer.distribute_tensors(
            grads, mesh=self.mesh, placements=[Partial(dist.ReduceOp.AVG)] * self.mesh.ndim
        )

        # zero_grad(set_to_none=True) clears sharded parameter grads, so the next
        # backward can reduce directly into main_grad. zero_grad(set_to_none=False)
        # leaves sharded grads installed, so this backward accumulates into main_grad.
        if has_grad(self.sharded_parameters):
            reduced_grad = partial_grad.redistribute(self.main_grad.placements)
            self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
        else:
            partial_grad.redistribute(self.main_grad.placements, out=self.main_grad)
            for index, parameter in enumerate(self.sharded_parameters):
                parameter.grad = self.main_grad.get_dtensor(index)

        for parameter in self.unsharded_parameters:
            parameter.grad = None


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    _parameter_groups: tuple[ParameterGroup, ...]
    _ready_grad_parameters: set[nn.Parameter]
    num_training_parameters: int

    def __init__(
        self, mesh: DeviceMesh, placements: Placements, mixed_precision_policy: MixedPrecisionPolicy
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        owned_parameters = _materialize_and_collect_owned_parameters(self, _mesh_device(mesh))
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

    def _register_hooks(self) -> None:
        self.register_forward_pre_hook(lambda _module, _args: self.pre_forward())
        self.register_forward_hook(lambda _module, _args, _output: self.post_forward())
        self.register_full_backward_pre_hook(lambda _module, _grad_output: self.pre_backward())
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
        for group in self._parameter_groups:
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


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
) -> None:
    """Shard one module as a per-module FSDP unit.

    Args:
        module: Module whose currently unowned parameters become this FSDP unit.
        mesh: Device mesh used for sharding.
        placements: Parameter, gradient, and optimizer placements.
        mixed_precision_policy: Optional precision policy. Defaults to FP32 main weights
            and parameter-dtype main gradients.
    """
    if isinstance(module, FsdpModule):
        raise ValueError("This module is already managed by FSDP.")

    mixed_precision_policy = mixed_precision_policy or MixedPrecisionPolicy()
    original_cls = module.__class__
    _attach_mixin(module)
    try:
        assert isinstance(module, FsdpModule)
        FsdpModule.__init__(
            module, mesh=mesh, placements=placements, mixed_precision_policy=mixed_precision_policy
        )
    except Exception:
        module.__class__ = original_cls
        raise


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
            if hasattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR):
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


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = name.rpartition(".")
    owner = module.get_submodule(module_name) if separator else module
    return owner, parameter_name


def _attach_mixin(module: nn.Module) -> None:
    if isinstance(module, FsdpModule):
        return
    module_cls = module.__class__
    fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
    module.__class__ = fsdp_cls
