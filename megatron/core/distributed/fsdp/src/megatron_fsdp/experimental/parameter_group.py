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
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from weakref import ReferenceType, ref

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .dbuffer import DBuffer
from .placement import Partial, Placements, Replicate, changed_mesh_axis

if TYPE_CHECKING:
    from .module import FsdpContext, FsdpModule

_CONTAINING_PARAMETER_GROUP_ATTR = "_mfsdp_parameter_group"


def get_containing_parameter_group(parameter: nn.Parameter) -> "FsdpParameterGroup | None":
    """Return the FSDP parameter group that owns ``parameter``, if any."""
    # This parameter-owned backedge must be weak; otherwise it forms a reference
    # cycle with the parameter group and delays releasing its CUDA storage.
    parameter_group_ref = getattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, None)
    if parameter_group_ref is None:
        return None
    return parameter_group_ref()


def sync_model_weights_from_main_weights(parameters: Iterable[nn.Parameter]) -> None:
    """Sync MFSDP compute weights for parameter groups represented by ``parameters``.

    Parameters outside the experimental MFSDP path are ignored. A parameter group
    may own multiple parameters, but its compute-weight buffer is synced once.
    """
    seen_parameter_groups = set()
    for parameter in parameters:
        if (parameter_group := get_containing_parameter_group(parameter)) is None:
            continue
        if parameter_group in seen_parameter_groups:
            continue
        seen_parameter_groups.add(parameter_group)
        parameter_group.sync_model_weight_from_main_weight()


@dataclass(frozen=True, eq=False)
class FsdpParameter:
    """One physical parameter and its FSDP runtime representations."""

    # Tied weights register one physical parameter under multiple FQNs, all relative
    # to the containing group's owning_module.
    fqns: tuple[str, ...]
    sharded: nn.Parameter
    unsharded: nn.Parameter


class FsdpParameterGroup:
    """A dtype and requires-grad homogeneous group of FSDP-owned parameters."""

    # FsdpModule owns its parameter groups, so this backedge must be weak to avoid
    # a reference cycle that delays releasing CUDA storage until cyclic GC.
    _owning_module: ReferenceType[nn.Module]
    fsdp_parameters: tuple[FsdpParameter, ...]
    mesh: DeviceMesh
    dtype: torch.dtype
    requires_grad: bool
    main_weight: DBuffer
    model_weight: DBuffer
    main_grad: DBuffer | None
    _unsharded_model_weight: DBuffer
    _symm_mem_pool: torch.cuda.MemPool | None
    grad_divisor: int

    def __init__(
        self,
        owning_module: nn.Module,
        parameters: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        reduce_scatter_stream: torch.cuda.Stream,
        grad_divisor: int = 1,
        use_symmetric_memory: bool = False,
    ) -> None:
        """Create persistent sharded buffers for a group of parameters.

        Args:
            owning_module: Closest FSDP root module that owns this parameter group.
            parameters: Root-module-relative FQNs and their parameters.
            mesh: Device mesh used for all DBuffer storage in this version.
            placements: Parameter, gradient, and optimizer placements.
            mixed_precision_policy: Precision policy for main weights and gradients.
            reduce_scatter_stream: Stream on which to allocate the main-gradient buffer.
            use_symmetric_memory: Allocate communication staging buffers from PyTorch's
                NCCL symmetric-memory pool.
            grad_divisor: Additional divisor applied on top of the mesh-size
                averaging. See ``fully_shard``.
        """
        if not parameters:
            raise ValueError("FsdpParameterGroup requires at least one parameter.")
        if use_symmetric_memory and not hasattr(symm_mem, "is_symm_mem_tensor"):
            raise RuntimeError("Symmetric-memory MFSDP requires PyTorch 2.12 or later.")

        parameter_to_fqns: dict[nn.Parameter, list[str]] = {}
        for fqn, parameter in parameters.items():
            parameter_to_fqns.setdefault(parameter, []).append(fqn)

        model_weight_placements = tuple(placements.parameter)
        main_grad_placements = tuple(placements.gradient)
        main_weight_placements = tuple(placements.optimizer)
        self._model_weight_placements = model_weight_placements
        # main_grad rests here (DP-outer-Partial for HSDP) between microbatches and
        # is finalized to main_weight's placements after the last microbatch.
        self._main_grad_placements = main_grad_placements

        # Python dicts preserve insertion order, so parameter_to_fqns and
        # fsdp_parameters define the same stable DBuffer tensor order.
        self._owning_module = ref(owning_module)
        self.mesh = mesh
        self.grad_divisor = grad_divisor
        first_parameter = next(iter(parameter_to_fqns))
        self.dtype = first_parameter.dtype
        self.requires_grad = first_parameter.requires_grad
        for parameter, fqns in parameter_to_fqns.items():
            if parameter.dtype != self.dtype:
                raise ValueError(
                    f"Expected parameter {fqns!r} to have dtype {self.dtype}, "
                    f"got {parameter.dtype}."
                )
            if parameter.requires_grad != self.requires_grad:
                raise ValueError(
                    f"Expected parameter {fqns!r} to have requires_grad={self.requires_grad}, "
                    f"got {parameter.requires_grad}."
                )

        tensor_shapes = tuple(parameter.shape for parameter in parameter_to_fqns)
        main_weight_dtype = mixed_precision_policy.main_params_dtype or torch.float32
        self.main_weight = DBuffer.distribute_tensors(
            (parameter.to(dtype=main_weight_dtype) for parameter in parameter_to_fqns),
            mesh=self.mesh,
            placements=main_weight_placements,
        )

        if use_symmetric_memory:
            # PyTorch caches this in C++ and returns early when the backend is already NCCL.
            symm_mem.set_backend("NCCL")
            self._symm_mem_pool = symm_mem.get_mem_pool(self.main_weight.device)
        else:
            self._symm_mem_pool = None

        # Match the optimizer post-step and checkpoint-load state: compute weights
        # begin as a cast of the optimizer weights and are restored to their
        # configured placements by the first pre-forward unshard.
        self.model_weight = self.main_weight.cast(self.dtype)
        with self._symmetric_memory_context():
            self._unsharded_model_weight = DBuffer(
                mesh=self.mesh,
                placements=[Replicate()] * self.mesh.ndim,
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
            with torch.cuda.stream(reduce_scatter_stream):
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
        fsdp_parameters: list[FsdpParameter] = []
        main_grad_dtype = self.main_grad.dtype if self.main_grad is not None else None
        for index, (parameter, fqns) in enumerate(parameter_to_fqns.items()):
            unsharded_tensor = self._unsharded_model_weight.get_local_tensor(index)
            if parameter.is_meta:
                # A meta Parameter cannot set .data to a real tensor because their
                # TensorImpl types are incompatible, so swap in a materialized Parameter.
                # This may be problematic if attributes from the original Parameter need
                # to be copied to the unsharded Parameter.
                materialized_parameter = nn.Parameter(
                    unsharded_tensor, requires_grad=parameter.requires_grad
                )
                torch.utils.swap_tensors(parameter, materialized_parameter)
            else:
                parameter.data = unsharded_tensor
                parameter.grad = None
            # Parameter-owned markers must not retain their FSDP module tree.
            setattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, ref(self))

            sharded_parameter = nn.Parameter(
                self.main_weight.get_dtensor(index), requires_grad=parameter.requires_grad
            )
            if main_grad_dtype:
                sharded_parameter.grad_dtype = main_grad_dtype
            setattr(sharded_parameter, _CONTAINING_PARAMETER_GROUP_ATTR, ref(self))
            fsdp_parameters.append(
                FsdpParameter(fqns=tuple(fqns), sharded=sharded_parameter, unsharded=parameter)
            )
        self.fsdp_parameters = tuple(fsdp_parameters)

        self._unsharded_model_weight.release_storage()
        self._switch_to_sharded_parameters()

    def _get_context(self) -> "FsdpContext":
        """Return the finalized runtime context that owns this parameter group."""
        owning_module = self._owning_module()
        if owning_module is None:
            raise RuntimeError("FSDP parameter group outlived its owning module.")
        context = cast("FsdpModule", owning_module).context
        context.ensure_finalized()
        return context

    def _symmetric_memory_context(self):
        if self._symm_mem_pool is None:
            return nullcontext()
        return torch.cuda.use_mem_pool(self._symm_mem_pool)

    def _set_module_parameter(self, fqns: tuple[str, ...], parameter: nn.Parameter) -> None:
        owning_module = self._owning_module()
        if owning_module is None:
            raise RuntimeError("FSDP parameter group outlived its owning module.")
        for fqn in fqns:
            module, parameter_name = _get_parameter_owner(owning_module, fqn)
            module._parameters[parameter_name] = parameter

    def _switch_to_sharded_parameters(self) -> None:
        for fsdp_parameter in self.fsdp_parameters:
            self._set_module_parameter(fsdp_parameter.fqns, fsdp_parameter.sharded)

    def _switch_to_unsharded_parameters(self) -> None:
        for fsdp_parameter in self.fsdp_parameters:
            self._set_module_parameter(fsdp_parameter.fqns, fsdp_parameter.unsharded)

    def sync_model_weight_from_main_weight(self) -> None:
        """Sync optimizer weights to the model-weight representation."""
        context = self._get_context()
        # TODO: Retrieve the all-gather stream directly from self.model_weight after
        # https://github.com/NVIDIA/Megatron-LM/pull/6441 merges.
        allgather_stream = context.allgather_stream
        current_stream = context.current_stream()
        allgather_stream.wait_stream(current_stream)
        with torch.cuda.stream(allgather_stream):
            self.model_weight = self.main_weight.cast(self.model_weight.dtype)
        # CUDA graph capture requires every forked stream to rejoin the capture
        # stream before capture ends.
        current_stream.wait_stream(allgather_stream)

    def unshard_parameters(self) -> None:
        """Install full parameters for local compute."""
        # In ZeRO-1, the post-step cast leaves model_weight sharded. Only the first
        # microbatch sees placements different from the configured model placements
        # and restores the replicated model weight.
        if self.model_weight.placements != self._model_weight_placements:
            with self._symmetric_memory_context():
                # Allocate the restored destination in symmetric memory when enabled so the
                # redistribution can use the faster symmetric-memory all-gather path.
                self.model_weight = self.model_weight.redistribute(self._model_weight_placements)
        if self.model_weight.placements == self._unsharded_model_weight.placements:
            unsharded_model_weight = self.model_weight
        else:
            with self._symmetric_memory_context():
                self._unsharded_model_weight.reallocate_storage()
            # This buffer backs unsharded Parameters whose views may be saved by autograd.
            # Autograd records a tensor's version counter when saving it for backward, and
            # in-place writes like the out= redistribution below increment that counter even
            # under no_grad. Without preserving it, backward can fail with "modified by an
            # inplace operation" even though FSDP only materialized internal storage.
            with torch.autograd._unsafe_preserve_version_counter(
                self._unsharded_model_weight.local_buffer
            ):
                self.model_weight.redistribute(
                    self._unsharded_model_weight.placements, out=self._unsharded_model_weight
                )
            unsharded_model_weight = self._unsharded_model_weight
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            fsdp_parameter.unsharded.data = unsharded_model_weight.get_local_tensor(index)
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

    def allocate_partial_grad_buffer(self) -> DBuffer:
        """Allocate the unreduced reduce-scatter input buffer."""
        assert self.main_grad is not None

        grads: list[torch.Tensor] = []
        for fsdp_parameter in self.fsdp_parameters:
            if fsdp_parameter.unsharded.grad is None:
                raise RuntimeError(f"Missing gradient for FSDP parameter {fsdp_parameter.fqns!r}.")
            grads.append(fsdp_parameter.unsharded.grad)
        with self._symmetric_memory_context():
            return DBuffer(
                mesh=self.mesh,
                placements=[Partial(dist.ReduceOp.AVG)] * self.mesh.ndim,
                tensor_shapes=tuple(grad.shape for grad in grads),
                dtype=grads[0].dtype,
                device=grads[0].device,
            )

    def copy_gradients_to_partial_buffer(self, partial_grad: DBuffer) -> None:
        """Pack full local gradients into an existing reduce-scatter input buffer."""
        # A future fused-wgrad path can write directly into these buffer views.
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            partial_grad.get_local_tensor(index).copy_(fsdp_parameter.unsharded.grad)
            fsdp_parameter.unsharded.grad = None

    def _has_sharded_grads(self) -> bool:
        has_any_grad = False
        has_any_missing_grad = False
        for fsdp_parameter in self.fsdp_parameters:
            if fsdp_parameter.sharded.grad is None:
                has_any_missing_grad = True
            else:
                has_any_grad = True
        if has_any_grad and has_any_missing_grad:
            raise RuntimeError("FSDP sharded gradients must be either all set or all None.")
        return has_any_grad

    def reduce_partial_gradients(
        self, partial_grad: DBuffer, is_last_microbatch: bool = True
    ) -> None:
        """Reduce a packed partial gradient buffer into sharded parameter gradients.

        For HSDP main_grad rests DP-outer-Partial (Partial where main_weight is
        Replicate) between microbatches, accumulating each backward through the
        standard zero_grad contract; the last microbatch reduces the DP-outer axes,
        finalizing main_grad to main_weight's placements so ``.grad`` is the fully
        reduced gradient before ``optimizer.step()``. With every axis Flat (plain
        DP) main_grad already rests finalized.
        """
        assert self.main_grad is not None

        # zero_grad(set_to_none=True) clears sharded parameter grads, so this
        # backward can reduce directly into main_grad. zero_grad(set_to_none=False)
        # leaves sharded grads installed, so this backward accumulates into main_grad.
        has_sharded_grads = self._has_sharded_grads()

        # A non-accumulation main_grad means the previous step finalized it; this
        # only happens on the first microbatch. Restore it to the DP-outer-Partial
        # accumulation placement. HSDP's finalize keeps the buffer size (Replicate
        # on DP-outer), so relabel it in place; ZeRO-1's finalize reduce-scattered
        # DP-outer to a smaller optimizer-sharded buffer, so allocate a fresh one
        # (zeroed only when we accumulate into it, i.e. sharded grads are still set).
        if self.main_grad.placements != self._main_grad_placements:
            assert self.main_grad.allocation_stream == (
                torch.cuda.current_stream(self.main_grad.device)
            )
            reset_axis = changed_mesh_axis(self.main_grad.placements, self._main_grad_placements)
            assert reset_axis is not None  # The placements differ, so an axis changed.
            if isinstance(self.main_grad.placements[reset_axis], Replicate):
                # HSDP: Replicate -> Partial changes only metadata and reuses the tensor.
                self.main_grad = self.main_grad.redistribute(self._main_grad_placements)
            else:
                # ZeRO-1: main_grad was reduce-scattered to the optimizer shard, too small
                # to hold the accumulation, so re-allocate. This runs inside the
                # reduce_scatter stream context (see FsdpModule._reduce_gradient_groups),
                # so the buffer is allocated on that stream and stays race-safe. Zero it
                # only when we accumulate (set_to_none=False); with set_to_none=True the
                # reduction below overwrites it via out=.
                self.main_grad = DBuffer(
                    mesh=self.mesh,
                    placements=self._main_grad_placements,
                    tensor_shapes=self.main_weight.layout.tensor_shapes,
                    dtype=self.main_grad.dtype,
                    device=self.main_weight.device,
                )
                if has_sharded_grads:
                    self.main_grad.local_buffer.zero_()

        if can_reduce_into_main_grad := (
            not has_sharded_grads and partial_grad.dtype == self.main_grad.dtype
        ):
            partial_grad.redistribute(self.main_grad.placements, out=self.main_grad)
            reduced_grad = self.main_grad
        else:
            reduced_grad = partial_grad.redistribute(self.main_grad.placements)

        # Scale this backward's contribution before accumulating it so repeated
        # backwards do not repeatedly scale the running total.
        if self.grad_divisor != 1:
            reduced_grad.local_buffer.div_(self.grad_divisor)

        if reduced_grad is not self.main_grad:
            if has_sharded_grads:
                self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
            else:
                self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)

        if is_last_microbatch:
            # Finalize the deferred DP-outer reduction (all-reduce for HSDP,
            # reduce-scatter for HFSDP) before binding the sharded parameter grads.
            assert self.main_grad.allocation_stream == (
                torch.cuda.current_stream(self.main_grad.device)
            )
            self.main_grad = self.main_grad.redistribute(self.main_weight.placements)

        # Make each sharded parameter's .grad consistent with the final main_grad.
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            fsdp_parameter.sharded.grad = self.main_grad.get_dtensor(index)


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = name.rpartition(".")
    owner = module.get_submodule(module_name) if separator else module
    return owner, parameter_name
