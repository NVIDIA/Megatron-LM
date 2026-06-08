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

"""Parameter-group runtime state for the minimal Megatron-FSDP path."""

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .dbuffer import DBuffer
from .placement import Partial, Placements, Replicate

_CONTAINING_PARAMETER_GROUP_ATTR = "_mfsdp_parameter_group"


def contained_in_parameter_group(parameter: nn.Parameter) -> bool:
    """Return whether a parameter is already owned by a ParameterGroup."""
    return hasattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR)


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
        if not parameters:
            raise ValueError("ParameterGroup requires at least one parameter.")

        model_weight_placements = tuple(placements.parameter)
        main_grad_placements = tuple(placements.gradient)
        main_weight_placements = tuple(placements.optimizer)

        # Python dicts preserve insertion order, so parameter_names and
        # parameters.values() define the same stable DBuffer tensor order.
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

        tensor_shapes = tuple(parameter.shape for parameter in parameters.values())
        main_weight_dtype = mixed_precision_policy.main_params_dtype or torch.float32
        self.main_weight = DBuffer.distribute_tensors(
            (parameter.to(dtype=main_weight_dtype) for parameter in parameters.values()),
            mesh=self.mesh,
            placements=main_weight_placements,
        )

        self._unsharded_model_weight = DBuffer(
            mesh=self.mesh,
            placements=[Replicate()] * self.mesh.ndim,
            tensor_shapes=tensor_shapes,
            dtype=self.dtype,
            device=self.main_weight.local_buffer.device,
        )
        if main_weight_dtype == self.dtype and main_weight_placements == model_weight_placements:
            self.model_weight = self.main_weight
        else:
            self.model_weight = DBuffer(
                mesh=self.mesh,
                placements=model_weight_placements,
                tensor_shapes=tensor_shapes,
                dtype=self.dtype,
                device=self.main_weight.local_buffer.device,
            )

        self.main_grad = None
        if self.requires_grad:
            grad_dtype = mixed_precision_policy.main_grads_dtype or self.dtype
            self.main_grad = DBuffer(
                mesh=self.mesh,
                placements=main_grad_placements,
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

        sharded_parameters: list[nn.Parameter] = []
        unsharded_parameters: list[nn.Parameter] = []
        main_grad_dtype = self.main_grad.local_buffer.dtype if self.main_grad is not None else None
        for index, parameter in enumerate(parameters.values()):
            parameter.data = self._unsharded_model_weight.get_local_tensor(index)
            parameter.grad = None
            setattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, self)
            unsharded_parameters.append(parameter)

            sharded_parameter = nn.Parameter(
                self.main_weight.get_dtensor(index), requires_grad=parameter.requires_grad
            )
            if main_grad_dtype:
                sharded_parameter.grad_dtype = main_grad_dtype
            setattr(sharded_parameter, _CONTAINING_PARAMETER_GROUP_ATTR, self)
            sharded_parameters.append(sharded_parameter)
        self.sharded_parameters = tuple(sharded_parameters)
        self.unsharded_parameters = tuple(unsharded_parameters)

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

    def release_unsharded_storage(self) -> None:
        """Release this group's full-parameter storage."""
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
            grads.append(parameter.grad)

        # This packs per-parameter grads into the reduce-scatter input buffer. A future
        # fused-wgrad path can avoid this copy by writing directly into those buffer views.
        partial_grad = DBuffer.distribute_tensors(
            grads, mesh=self.mesh, placements=[Partial(dist.ReduceOp.AVG)] * self.mesh.ndim
        )

        # zero_grad(set_to_none=True) clears sharded parameter grads, so the first
        # backward can reduce directly into main_grad. Later microbatches, or
        # zero_grad(set_to_none=False), leave sharded grads installed and accumulate.
        if has_grad(self.sharded_parameters):
            reduced_grad = partial_grad.redistribute(self.main_grad.placements)
            self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
        else:
            if partial_grad.local_buffer.dtype == self.main_grad.local_buffer.dtype:
                partial_grad.redistribute(self.main_grad.placements, out=self.main_grad)
            else:
                reduced_grad = partial_grad.redistribute(self.main_grad.placements)
                self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)

            for index, parameter in enumerate(self.sharded_parameters):
                parameter.grad = self.main_grad.get_dtensor(index)

        for parameter in self.unsharded_parameters:
            parameter.grad = None


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = name.rpartition(".")
    owner = module.get_submodule(module_name) if separator else module
    return owner, parameter_name
