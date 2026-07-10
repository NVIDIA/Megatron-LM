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
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .dbuffer import DBuffer
from .placement import Partial, Placement, Placements, Replicate, changed_mesh_axis

_CONTAINING_PARAMETER_GROUP_ATTR = "_mfsdp_parameter_group"


def contained_in_parameter_group(parameter: nn.Parameter) -> bool:
    """Return whether a parameter is already owned by an FsdpParameterGroup."""
    return hasattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR)


class FsdpParameterGroup:
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
    _symm_mem_pool: torch.cuda.MemPool | None

    def __init__(
        self,
        owning_module: nn.Module,
        parameters: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        use_symm_mem: bool = False,
    ) -> None:
        """Create persistent sharded buffers for a group of parameters.

        Args:
            owning_module: Closest FSDP root module that owns this parameter group.
            parameters: Root-module-relative FQNs and their parameters.
            mesh: Device mesh used for all DBuffer storage in this version.
            placements: Parameter, gradient, and optimizer placements.
            mixed_precision_policy: Precision policy for main weights and gradients.
            use_symm_mem: Allocate communication staging buffers from PyTorch's
                NCCL symmetric-memory pool.
        """
        if not parameters:
            raise ValueError("FsdpParameterGroup requires at least one parameter.")

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

        if use_symm_mem:
            # PyTorch caches this in C++ and returns early when the backend is already NCCL.
            symm_mem.set_backend("NCCL")
            self._symm_mem_pool = symm_mem.get_mem_pool(self.main_weight.device)
        else:
            self._symm_mem_pool = None

        with self._symmetric_memory_context():
            self._unsharded_model_weight = DBuffer(
                mesh=self.mesh,
                placements=[Replicate()] * self.mesh.ndim,
                tensor_shapes=tensor_shapes,
                dtype=self.dtype,
                device=self.main_weight.device,
            )
        if main_weight_dtype == self.dtype and main_weight_placements == model_weight_placements:
            self.model_weight = self.main_weight
        else:
            self.model_weight = DBuffer(
                mesh=self.mesh,
                placements=model_weight_placements,
                tensor_shapes=tensor_shapes,
                dtype=self.dtype,
                device=self.main_weight.device,
            )

        self.main_grad = None
        if self.requires_grad:
            grad_dtype = mixed_precision_policy.main_grads_dtype or self.dtype
            # Keep main_grad persistent for the initial implementation. For micro-batch
            # size 1, this allocation could be delayed until post_backward and then
            # eagerly deallocated right after optimizer.step(), avoiding main_grad
            # storage during forward. That requires a separate lifetime contract with
            # the optimizer, so this version keeps the simpler persistent buffer.
            self.main_grad = DBuffer(
                mesh=self.mesh,
                placements=main_grad_placements,
                tensor_shapes=self.main_weight.layout.tensor_shapes,
                dtype=grad_dtype,
                device=self.main_weight.device,
            )
            assert self.main_grad.layout == self.main_weight.layout, (
                "main_grad is built from main_weight tensor shapes on the same mesh, "
                "and DBuffer layouts are deterministic from those shapes and mesh size."
            )
            if not _grad_placements_reduce_to_weight(
                self.main_grad.placements, self.main_weight.placements
            ):
                raise ValueError(
                    "FSDP requires main_grad placements to equal main_weight placements, or "
                    "to defer a reduction by being Partial on an axis where main_weight is "
                    "Replicate (HSDP DP-outer accumulation). "
                    f"Got main_grad placements {self.main_grad.placements} and "
                    f"main_weight placements {self.main_weight.placements}."
                )
            # When main_grad placements differ from main_weight, main_grad rests at a
            # DP-outer Partial accumulation state: each backward reduce-scatters DP-inner
            # into it, and the deferred DP-outer reduction runs on the last microbatch.
            self._defers_grad_reduction = self.main_grad.placements != self.main_weight.placements
            self._grad_accumulating = False
            self._reduced_grad: DBuffer | None = None
        sharded_parameters: list[nn.Parameter] = []
        unsharded_parameters: list[nn.Parameter] = []
        main_grad_dtype = self.main_grad.dtype if self.main_grad is not None else None
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

    def _symmetric_memory_context(self):
        if self._symm_mem_pool is None:
            return nullcontext()
        return torch.cuda.use_mem_pool(self._symm_mem_pool)

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

        self.main_weight.cast(self.model_weight.dtype).redistribute(
            self.model_weight.placements, out=self.model_weight
        )

    def unshard_parameters(self) -> None:
        """Install full parameters for local compute."""
        with self._symmetric_memory_context():
            self._unsharded_model_weight.reallocate_storage()
        # This buffer backs unsharded Parameters whose views may be saved by autograd.
        # Autograd records a tensor's version counter when saving it for backward, and
        # in-place writes like the out= redistribution below increment that counter even
        # under no_grad. Without preserving it, backward can fail with "modified by an
        # inplace operation" even though FSDP only materialized internal storage.
        gather_axis = changed_mesh_axis(
            self.model_weight.placements, self._unsharded_model_weight.placements
        )
        if gather_axis is None:
            raise RuntimeError("FSDP parameter unshard requires a changed placement axis.")
        with torch.autograd._unsafe_preserve_version_counter(
            self._unsharded_model_weight.local_buffer
        ):
            if self._symm_mem_pool is not None:
                self._unsharded_model_weight.rendezvous(gather_axis)
            self.model_weight.redistribute(
                self._unsharded_model_weight.placements, out=self._unsharded_model_weight
            )
        self._switch_to_unsharded_parameters()

    def reshard_parameters(self) -> None:
        """Install sharded DTensor parameters on the owning modules."""
        self._switch_to_sharded_parameters()

    def release_unsharded_storage(self) -> None:
        """Release this group's full-parameter storage."""
        # This method is shared by the post-forward and post-backward release
        # paths. Post-forward must release storage because autograd may have
        # saved forward views into the unsharded parameters. Post-backward could
        # replace unsharded parameter .data with size-0 empty tensors, instead
        # of releasing storage, because autograd has consumed those saved views.
        # That alternative is not much cleaner, and splitting post-forward and
        # post-backward reshard behavior would make the caller code less clean,
        # so keep the shared storage-release path.
        self._unsharded_model_weight.release_storage()

    def _reduce_partial_grad(
        self, partial_grad: DBuffer, partial_op: dist.ReduceOp.RedOpType, *, out: "DBuffer | None"
    ) -> DBuffer:
        """Reduce an all-Partial gradient buffer to ``main_grad``'s placements.

        ``DBuffer.redistribute`` changes one mesh axis per call, so a 2-D DP mesh
        (HSDP/HFSDP) composes the reduction axis by axis. Innermost axes are
        reduced first: each reduce-scatter shrinks the buffer before the next
        collective, and every intermediate placement keeps ``Flat`` a suffix
        (``[Partial, Flat]`` is valid, ``[Flat, Partial]`` is not). The outermost
        changed axis is reduced last so it can write directly into ``out``.
        """
        target = self.main_grad.placements
        changed_axes = [
            axis
            for axis in range(self.mesh.ndim)
            if partial_grad.placements[axis] != target[axis]
        ]
        if not changed_axes:
            raise RuntimeError("FSDP gradient reduction requires a changed placement axis.")
        if self._symm_mem_pool is not None and len(changed_axes) > 1:
            raise NotImplementedError(
                "Symmetric-memory gradient reduction supports a single mesh axis; "
                f"got changed axes {changed_axes}."
            )

        current = partial_grad
        for axis in reversed(changed_axes):
            placements = list(current.placements)
            placements[axis] = target[axis]
            if self._symm_mem_pool is not None:
                current.rendezvous(axis)
            is_outermost_change = axis == changed_axes[0]
            current = current.redistribute(placements, out=out if is_outermost_change else None)
            if partial_op == dist.ReduceOp.SUM:
                current.local_buffer.div_(self.mesh.size(axis))
        return current

    def _accumulate_deferred_grad(
        self, partial_grad: DBuffer, partial_op: dist.ReduceOp.RedOpType, is_last_microbatch: bool
    ) -> None:
        """Accumulate DP-inner-reduced grads, deferring the DP-outer reduction.

        main_grad rests at its DP-outer-Partial placement. Each backward
        reduce-scatters DP-inner into main_grad (resetting it on the first
        microbatch, accumulating afterwards). Only the last microbatch reduces
        DP-outer into ``main_weight``'s placement and installs the resulting
        sharded parameter gradients.
        """
        assert self.main_grad is not None
        inner_reduced = self._reduce_partial_grad(partial_grad, partial_op, out=None)
        if self._grad_accumulating:
            self.main_grad.local_buffer.add_(inner_reduced.local_buffer)
        else:
            # First microbatch of a step: reset the accumulator and release the
            # previous step's reduced-grad buffer, whose sharded grads have been
            # cleared by optimizer.zero_grad().
            self._reduced_grad = None
            self.main_grad.local_buffer.copy_(inner_reduced.local_buffer)
            self._grad_accumulating = True

        if not is_last_microbatch:
            return

        # Reduce DP-outer into the optimizer's placement; keep the buffer alive for
        # optimizer.step() through both this reference and the sharded grad DTensors.
        self._reduced_grad = self.main_grad.redistribute(self.main_weight.placements)
        for index, sharded_parameter in enumerate(self.sharded_parameters):
            sharded_parameter.grad = self._reduced_grad.get_dtensor(index)
        self._grad_accumulating = False

    def reduce_gradients(self, is_last_microbatch: bool = True) -> None:
        """Reduce full local gradients into sharded parameter gradients.

        For a plain DP mesh (or matching gradient/optimizer placements) the
        reduction runs fully every backward. When main_grad defers a DP-outer
        reduction (HSDP: main_grad Partial where main_weight is Replicate), each
        backward reduce-scatters DP-inner and accumulates into main_grad, and only
        the last microbatch reduces DP-outer and installs the sharded gradients.
        """
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

        # NCCL symmetric-memory reduce-scatter only selects the symmetric kernel for SUM today.
        # Preserve AVG semantics by reducing SUM and scaling the output below.
        partial_op = dist.ReduceOp.AVG if self._symm_mem_pool is None else dist.ReduceOp.SUM
        with self._symmetric_memory_context():
            partial_grad = DBuffer.distribute_tensors(
                grads, mesh=self.mesh, placements=[Partial(partial_op)] * self.mesh.ndim
            )

        if self._defers_grad_reduction:
            self._accumulate_deferred_grad(partial_grad, partial_op, is_last_microbatch)
            for parameter in self.unsharded_parameters:
                parameter.grad = None
            return

        # zero_grad(set_to_none=True) clears sharded parameter grads, so the next
        # backward can reduce directly into main_grad. zero_grad(set_to_none=False)
        # leaves sharded grads installed, so this backward accumulates into main_grad.
        has_sharded_grads = has_grad(self.sharded_parameters)
        can_reduce_into_main_grad = (
            not has_sharded_grads and partial_grad.dtype == self.main_grad.dtype
        )
        if can_reduce_into_main_grad:
            self._reduce_partial_grad(partial_grad, partial_op, out=self.main_grad)
        else:
            reduced_grad = self._reduce_partial_grad(partial_grad, partial_op, out=None)
            if has_sharded_grads:
                self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
            else:
                self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)

        if not has_sharded_grads:
            for index, parameter in enumerate(self.sharded_parameters):
                parameter.grad = self.main_grad.get_dtensor(index)

        for parameter in self.unsharded_parameters:
            parameter.grad = None


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = name.rpartition(".")
    owner = module.get_submodule(module_name) if separator else module
    return owner, parameter_name


def _grad_placements_reduce_to_weight(
    grad_placements: tuple[Placement, ...], weight_placements: tuple[Placement, ...]
) -> bool:
    """Return whether main_grad placements reduce to main_weight placements.

    They may be equal, or main_grad may be ``Partial`` on a single axis where
    main_weight is ``Replicate`` (an HSDP DP-outer reduction deferred to the last
    microbatch). All other axes must match so the buffers share a layout and
    local size. At most one deferred axis is allowed because the deferred
    reduction finalizes with a single-axis ``DBuffer.redistribute``; multi-axis
    deferral is not yet supported.
    """
    if len(grad_placements) != len(weight_placements):
        return False
    deferred_axes = 0
    for grad_placement, weight_placement in zip(grad_placements, weight_placements):
        if grad_placement == weight_placement:
            continue
        if isinstance(grad_placement, Partial) and isinstance(weight_placement, Replicate):
            deferred_axes += 1
            continue
        return False
    return deferred_axes <= 1
