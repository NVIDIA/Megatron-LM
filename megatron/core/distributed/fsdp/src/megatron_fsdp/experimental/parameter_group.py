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
from weakref import ReferenceType, ref

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.tensor as dist_tensor
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .dbuffer import DBuffer
from .placement import Partial, Placement, Placements, Replicate, changed_mesh_axis
from .quantization import (
    E4M3_BLOCK_SIZE,
    allocate_quantize_temp,
    clear_payloads,
    set_columnwise_payload,
    set_rowwise_payload,
    te_cast_master_weights_to_fp8,
)

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
    model_weight: DBuffer | None
    _main_grad: DBuffer | None
    _unsharded_model_weight: DBuffer | None
    _symm_mem_pool: torch.cuda.MemPool | None
    grad_divisor: int
    _model_weight_placements: tuple[Placement, ...]
    # Cached sharded-gradient DTensors keyed by the main_grad DBuffer identity.
    # Rebuilding DTensors every backward costs O(params) ``_FromTorchTensor``
    # calls on the host; when main_grad storage is reused we rebind the cached
    # DTensor's local storage in place instead. Invalidation: cleared whenever
    # ``_main_grad`` is replaced (redistribute) or ``main_grad`` changes dtype.
    _grad_dtensor_cache: list[dist_tensor.DTensor | None]
    _grad_dtensor_cache_main_grad_id: int | None

    def __init__(
        self,
        owning_module: nn.Module,
        parameters: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        allgather_stream: torch.cuda.Stream,
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
            allgather_stream: Stream used to allocate model weights when a dtype cast is required.
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
        self._init_compute_weight_storage(
            tensor_shapes,
            main_weight_dtype,
            model_weight_placements,
            main_weight_placements,
            allgather_stream,
            use_symmetric_memory,
        )

        self._main_grad = None
        self._main_grad_dtype = None
        self._grad_dtensor_cache = []
        self._grad_dtensor_cache_main_grad_id = None
        if self.requires_grad:
            # main_grad itself is materialized lazily by the property below; only its
            # dtype and placement metadata are recorded here. See that property for why.
            self._main_grad_dtype = mixed_precision_policy.main_grads_dtype or self.dtype
        fsdp_parameters: list[FsdpParameter] = []
        main_grad_dtype = self._main_grad_dtype
        for index, (parameter, fqns) in enumerate(parameter_to_fqns.items()):
            unsharded_tensor = (
                self._unsharded_model_weight.get_local_tensor(index)
                if self._unsharded_model_weight is not None
                else None
            )
            self._materialize_unsharded_parameter(parameter, unsharded_tensor)
            # Parameter-owned markers must not retain their FSDP module tree.
            setattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, ref(self))

            sharded_parameter = nn.Parameter(
                self.main_weight.get_dtensor(index), requires_grad=parameter.requires_grad
            )
            sharded_parameter.__fsdp_param__ = True
            if main_grad_dtype:
                sharded_parameter.grad_dtype = main_grad_dtype
            setattr(sharded_parameter, _CONTAINING_PARAMETER_GROUP_ATTR, ref(self))
            fsdp_parameters.append(
                FsdpParameter(fqns=tuple(fqns), sharded=sharded_parameter, unsharded=parameter)
            )
        self.fsdp_parameters = tuple(fsdp_parameters)

        # Compute weights must be initialized before the first forward; subsequent
        # refreshes happen from the FSDP optimizer's post-step hook.
        self.sync_model_weight_from_main_weight()
        self._switch_to_sharded_parameters()
        if self._unsharded_model_weight is not None:
            self._unsharded_model_weight.release_storage()

    def _init_compute_weight_storage(
        self,
        tensor_shapes: tuple[torch.Size, ...],
        main_weight_dtype: torch.dtype,
        model_weight_placements: tuple[Placement, ...],
        main_weight_placements: tuple[Placement, ...],
        allgather_stream: torch.cuda.Stream,
        use_symmetric_memory: bool,
    ) -> None:
        """Create the sharded and replicated compute-weight storage."""
        del model_weight_placements, use_symmetric_memory
        with self._symmetric_memory_context():
            self._unsharded_model_weight = DBuffer(
                mesh=self.mesh,
                placements=[Replicate()] * self.mesh.ndim,
                tensor_shapes=tensor_shapes,
                dtype=self.dtype,
                device=self.main_weight.device,
            )
        if main_weight_dtype == self.dtype:
            self.model_weight = self.main_weight
        else:
            # Keep the resting compute-weight shard on the optimizer placement.
            # ``unshard_parameters`` restores ``model_weight_placements`` before compute.
            with torch.cuda.stream(allgather_stream):
                self.model_weight = DBuffer(
                    mesh=self.mesh,
                    placements=main_weight_placements,
                    tensor_shapes=tensor_shapes,
                    dtype=self.dtype,
                    device=self.main_weight.device,
                )

    def _materialize_unsharded_parameter(
        self, parameter: nn.Parameter, unsharded_tensor: torch.Tensor | None
    ) -> None:
        """Install the full compute parameter on the module's parameter object."""
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
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            self._set_module_parameter(fsdp_parameter.fqns, self._get_unsharded_parameter(index))

    def _get_unsharded_parameter(self, index: int) -> torch.Tensor:
        """Return the parameter object installed on the module for tensor ``index``."""
        return self.fsdp_parameters[index].unsharded

    def sync_model_weight_from_main_weight(self) -> None:
        """Refresh compute weights from optimizer weights."""
        assert self.model_weight is not None
        if self.main_weight is self.model_weight:
            return

        allgather_stream = self.model_weight.allocation_stream
        assert allgather_stream is not None
        current_stream = torch.cuda.current_stream(self.model_weight.device)
        allgather_stream.wait_stream(current_stream)
        with torch.cuda.stream(allgather_stream):
            self.model_weight = self.main_weight.cast(self.model_weight.dtype)
        # CUDA graph capture requires every forked stream to rejoin the capture
        # stream before capture ends.
        current_stream.wait_stream(allgather_stream)

    def sync_model_weight_from_unsharded_weight(self) -> None:
        """Copy reset unsharded weights back into the sharded buffers, aligned across ranks.

        After ``reset_parameters()`` writes the full (Replicate) unsharded weight,
        each rank holds independently-sampled values. Broadcast the full weight from
        rank 0 of every mesh dimension so all ranks align, then scatter it back into
        the sharded optimizer main weight and the compute model weight.
        """
        unsharded = self._unsharded_model_weight
        assert unsharded is not None
        assert self.model_weight is not None
        for mesh_dim in range(self.mesh.ndim):
            group = self.mesh.get_group(mesh_dim=mesh_dim)
            if torch.distributed.get_world_size(group) == 1:
                continue
            src_rank = torch.distributed.get_global_rank(group, 0)
            torch.distributed.broadcast(unsharded.local_buffer, src=src_rank, group=group)

        if self.main_weight is not self.model_weight:
            unsharded.cast(self.main_weight.dtype).redistribute(
                self.main_weight.placements, out=self.main_weight
            )
        unsharded.cast(self.model_weight.dtype).redistribute(
            self.model_weight.placements, out=self.model_weight
        )

    def unshard_parameters(self, orientation: str = "rowwise") -> None:
        """Install full parameters for local compute.

        Args:
            orientation: Which payload orientation to gather for MXFP8 groups.
                Ignored by regular groups.
        """
        del orientation
        assert self.model_weight is not None
        assert self._unsharded_model_weight is not None
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
        if self._unsharded_model_weight is not None:
            self._unsharded_model_weight.release_storage()

    @property
    def main_grad(self) -> DBuffer | None:
        """Sharded gradient buffer, materialized on first access.

        Allocation is deferred to the first backward so that the buffer is created on
        ``reduce_scatter_stream`` -- the same stream that later frees it when
        ``reduce_partial_gradients`` rebinds it. Allocating during ``fully_shard()``
        would bind the block to the caller's stream, typically the default stream. The
        caching allocator binds a block to its allocation stream, so that buffer would
        become reusable by default-stream allocations the moment it is dropped here,
        while the reduce-scatter is still reading it as its input.
        ``FsdpModule._reshard_parameter_groups`` keeps the same
        allocate-and-release-on-one-stream invariant for unsharded parameter storage.

        Returns ``None`` for parameter groups that do not require gradients.
        """
        if self._main_grad is not None:
            return self._main_grad
        if not self.requires_grad:
            return None

        owning_module = self._owning_module()
        if owning_module is None:
            raise RuntimeError("FSDP parameter group outlived its owning module.")
        assert (
            torch.cuda.current_stream(self.main_weight.device)
            == owning_module.context.reduce_scatter_stream
        ), "main_grad must be allocated on the reduce-scatter stream."
        self._main_grad = DBuffer(
            mesh=self.mesh,
            placements=self._main_grad_placements,
            tensor_shapes=self.main_weight.layout.tensor_shapes,
            dtype=self._main_grad_dtype,
            device=self.main_weight.device,
        )
        assert self._main_grad.layout == self.main_weight.layout, (
            "main_grad is built from main_weight tensor shapes on the same mesh, "
            "and DBuffer layouts are deterministic from those shapes and mesh size."
        )
        return self._main_grad

    def allocate_partial_grad_buffer(self) -> DBuffer:
        """Allocate the unreduced reduce-scatter input buffer.

        NOTE: deliberately not cached across microbatches.  Reusing the buffer
        made the NCCL symmetric-memory pool keep the storage registered across
        microbatches, so the next backward's ``copy_`` write forced a device
        sync (cudaEventSynchronize ~24x in nsys).  A fresh buffer per backward
        keeps the allocate-on-reduce-scatter-stream + release invariant that
        avoids allocator/symm-mem serialization.
        """
        assert self.requires_grad

        # NCCL symmetric-memory reduce-scatter only selects the symmetric kernel for SUM today.
        # Preserve AVG semantics by reducing SUM and scaling the output below.
        partial_op = dist.ReduceOp.AVG if self._symm_mem_pool is None else dist.ReduceOp.SUM
        grads: list[torch.Tensor] = []
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            grad = self._get_unsharded_parameter(index).grad
            if grad is None:
                raise RuntimeError(f"Missing gradient for FSDP parameter {fsdp_parameter.fqns!r}.")
            grads.append(grad)
        with self._symmetric_memory_context():
            return DBuffer(
                mesh=self.mesh,
                placements=[Partial(partial_op)] * self.mesh.ndim,
                tensor_shapes=tuple(grad.shape for grad in grads),
                dtype=grads[0].dtype,
                device=grads[0].device,
            )

    def copy_gradients_to_partial_buffer(self, partial_grad: DBuffer) -> None:
        """Pack full local gradients into an existing reduce-scatter input buffer."""
        # A future fused-wgrad path can write directly into these buffer views.
        destinations = [
            partial_grad.get_local_tensor(index) for index in range(len(self.fsdp_parameters))
        ]
        sources = [
            self._get_unsharded_parameter(index).grad
            for index in range(len(self.fsdp_parameters))
        ]
        if destinations:
            torch._foreach_copy_(destinations, sources)
        for index in range(len(self.fsdp_parameters)):
            self._get_unsharded_parameter(index).grad = None

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

        For HSDP/HFSDP main_grad rests DP-outer-Partial between microbatches,
        accumulating each backward through the standard zero_grad contract; the
        last microbatch reduces the DP-outer axes, finalizing main_grad to
        main_weight's placements (all-reduce to Replicate for HSDP, reduce-scatter
        to Flat for HFSDP) so ``.grad`` is the fully reduced gradient before
        ``optimizer.step()``. With every axis Flat (plain DP) main_grad already
        rests finalized.
        """
        assert self.main_grad is not None

        # zero_grad(set_to_none=True) clears sharded parameter grads, so this
        # backward can reduce directly into main_grad. zero_grad(set_to_none=False)
        # leaves sharded grads installed, so this backward accumulates into main_grad.
        has_sharded_grads = self._has_sharded_grads()

        # A non-accumulation main_grad means the previous step finalized it; this
        # only happens on the first microbatch. Restore it to the DP-outer-Partial
        # accumulation placement. HSDP's finalize keeps the buffer size (Replicate
        # on DP-outer), so relabel it in place; HFSDP's finalize reduce-scattered
        # DP-outer to a smaller optimizer-sharded buffer, so allocate a fresh one
        # (zeroed only when we accumulate into it, i.e. sharded grads are still set).
        if self.main_grad.placements != self._main_grad_placements:
            assert self.main_grad.allocation_stream == (
                torch.cuda.current_stream(self.main_grad.device)
            )
            reset_axis = changed_mesh_axis(self.main_grad.placements, self._main_grad_placements)
            assert reset_axis is not None  # the placements differ, so an axis changed
            if isinstance(self.main_grad.placements[reset_axis], Replicate):
                # HSDP: Replicate -> Partial changes only metadata and reuses the tensor.
                self._main_grad = self.main_grad.redistribute(self._main_grad_placements)
            else:
                # HFSDP: main_grad was reduce-scattered to the optimizer shard, too small
                # to hold the accumulation, so re-allocate. This runs inside the
                # reduce_scatter stream context (see FsdpModule._reduce_gradient_groups),
                # so the buffer is allocated on that stream and stays race-safe. Zero it
                # only when we accumulate (set_to_none=False); with set_to_none=True the
                # reduction below overwrites it via out=.
                self._main_grad = DBuffer(
                    mesh=self.mesh,
                    placements=self._main_grad_placements,
                    tensor_shapes=self.main_weight.layout.tensor_shapes,
                    dtype=self.main_grad.dtype,
                    device=self.main_weight.device,
                )
                if has_sharded_grads:
                    self.main_grad.local_buffer.zero_()

        reduce_axis = changed_mesh_axis(partial_grad.placements, self.main_grad.placements)
        if reduce_axis is None:
            raise RuntimeError("FSDP gradient reduction requires a changed placement axis.")
        partial_reduce_op = partial_grad.placements[reduce_axis].reduce_op
        # Start from the caller's extra divisor (1 unless this group sees more than one
        # contribution per mesh rank, as expert parallelism does), then add back the axis
        # size when the collective reduced with SUM instead of averaging.
        grad_divisor = self.grad_divisor
        if partial_reduce_op == dist.ReduceOp.SUM:
            grad_divisor *= self.mesh.size(reduce_axis)
        if self._symm_mem_pool is not None:
            partial_grad.rendezvous(reduce_axis)

        if can_reduce_into_main_grad := (
            not has_sharded_grads and partial_grad.dtype == self.main_grad.dtype
        ):
            partial_grad.redistribute(self.main_grad.placements, out=self.main_grad)
            reduced_grad = self.main_grad
        else:
            reduced_grad = partial_grad.redistribute(self.main_grad.placements)

        # Scale this backward's contribution before accumulating it so repeated
        # backwards do not repeatedly scale the running total.
        if grad_divisor != 1:
            reduced_grad.local_buffer.div_(grad_divisor)

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
            self._main_grad = self.main_grad.redistribute(self.main_weight.placements)

        # Make each sharded parameter's .grad consistent with the final main_grad.
        # Reuse cached DTensors where possible: rebinding storage in place avoids
        # O(params) ``DTensor.from_local`` (``_FromTorchTensor``) host cost per
        # backward.  The cache is keyed on the main_grad DBuffer identity so a
        # fresh buffer (redistribute replacement, dtype change) rebuilds it.
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            fsdp_parameter.sharded.grad = self._get_sharded_grad_dtensor(index)

    def _get_sharded_grad_dtensor(self, index: int) -> dist_tensor.DTensor:
        """Return the sharded-gradient DTensor for parameter ``index``, cached.

        The first call for a given ``main_grad`` storage builds DTensors via
        ``get_dtensor``; later calls with the same storage rebind the cached
        DTensor's local tensor in place, skipping ``DTensor.from_local``.
        """
        main_grad = self.main_grad
        assert main_grad is not None
        cache_id = self._grad_dtensor_cache_main_grad_id
        if cache_id != id(main_grad):
            self._grad_dtensor_cache = [None] * len(self.fsdp_parameters)
            self._grad_dtensor_cache_main_grad_id = id(main_grad)

        cached = self._grad_dtensor_cache[index]
        if cached is not None:
            new_local = main_grad.get_local_tensor(index)
            old_local = cached._local_tensor
            # Rebind only when the backing storage actually changed (e.g. the
            # reduce-scatter wrote into a fresh buffer).  When storage is
            # reused, the cached view already aliases the new contents, so
            # rebinding would just force a redundant DTensor re-validation
            # (extra cudaEventSynchronize + memcpy on the host).
            if (
                old_local is not None
                and old_local.data_ptr() == new_local.data_ptr()
                and old_local.shape == new_local.shape
                and old_local.dtype == new_local.dtype
            ):
                return cached
            # View-level rebind, not a tensor-object replacement: keeps the
            # DTensor shell alive while pointing at the new storage without
            # triggering a device sync.
            object.__setattr__(cached._local_tensor, "data", new_local)
            return cached

        dtensor = main_grad.get_dtensor(index)
        self._grad_dtensor_cache[index] = dtensor
        return dtensor


class Fp8ParameterGroup(FsdpParameterGroup):
    """FSDP parameter group whose parameters are TE MXFP8Tensor primary weights.

    The sharded compute weights rest as row-wise (forward GEMM) and column-wise
    (backward GEMM) MXFP8 E4M3 payloads. Unshard gathers both payloads and binds
    them to the module's MXFP8Tensor objects; reshard detaches them.
    """

    _rowwise_buffer: DBuffer
    _colwise_buffer: DBuffer
    _unsharded_rowwise: DBuffer
    _unsharded_colwise: DBuffer

    def __init__(
        self,
        owning_module: nn.Module,
        parameters: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        allgather_stream: torch.cuda.Stream,
        reduce_scatter_stream: torch.cuda.Stream,
        grad_divisor: int = 1,
        use_symmetric_memory: bool = False,
    ) -> None:
        # Keep the subclass constructor aligned with FsdpParameterGroup. The
        # shared module factory passes these keywords without knowing whether
        # a group owns BF16 or MXFP8 weights.
        if use_symmetric_memory:
            raise ValueError("MFSDP v2 fp8 model weights do not support symmetric memory yet.")
        if te_cast_master_weights_to_fp8() is None:
            raise RuntimeError(
                "MFSDP v2 fp8 model weights require Transformer Engine with "
                "cast_master_weights_to_fp8 support."
            )
        super().__init__(
            owning_module=owning_module,
            parameters=parameters,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=mixed_precision_policy,
            allgather_stream=allgather_stream,
            reduce_scatter_stream=reduce_scatter_stream,
            grad_divisor=grad_divisor,
            use_symmetric_memory=False,
        )

    def _init_compute_weight_storage(
        self,
        tensor_shapes: tuple[torch.Size, ...],
        main_weight_dtype: torch.dtype,
        model_weight_placements: tuple[Placement, ...],
        main_weight_placements: tuple[Placement, ...],
        allgather_stream: torch.cuda.Stream,
        use_symmetric_memory: bool,
    ) -> None:
        del main_weight_dtype, main_weight_placements, allgather_stream, use_symmetric_memory
        # The bf16 model-weight storage is replaced by the two uint8 payload
        # DBuffers; the unsharded parameters are the module's own MXFP8Tensor
        # objects whose raw payloads are rebound from the gathered buffers.
        self.model_weight = None
        self._unsharded_model_weight = None
        device = self.main_weight.device
        self._rowwise_buffer = DBuffer(
            mesh=self.mesh,
            placements=model_weight_placements,
            tensor_shapes=tensor_shapes,
            dtype=torch.uint8,
            device=device,
        )
        self._colwise_buffer = DBuffer(
            mesh=self.mesh,
            placements=model_weight_placements,
            tensor_shapes=tensor_shapes,
            dtype=torch.uint8,
            device=device,
        )
        self._unsharded_rowwise = DBuffer(
            mesh=self.mesh,
            placements=[Replicate()] * self.mesh.ndim,
            tensor_shapes=tensor_shapes,
            dtype=torch.uint8,
            device=device,
        )
        self._unsharded_colwise = DBuffer(
            mesh=self.mesh,
            placements=[Replicate()] * self.mesh.ndim,
            tensor_shapes=tensor_shapes,
            dtype=torch.uint8,
            device=device,
        )
        for index, shape in enumerate(tensor_shapes):
            if (
                len(shape) != 2
                or shape[0] % E4M3_BLOCK_SIZE != 0
                or shape[1] % E4M3_BLOCK_SIZE != 0
            ):
                raise ValueError(
                    f"MXFP8 parameter tensor {index} with shape {shape} must be 2D with "
                    f"dims divisible by {E4M3_BLOCK_SIZE}."
                )

    def _materialize_unsharded_parameter(
        self, parameter: nn.Parameter, unsharded_tensor: torch.Tensor | None
    ) -> None:
        del unsharded_tensor
        # The module already owns an MXFP8Tensor primary weight. Keep that
        # object and only reset its gradient; payloads are rebound at unshard.
        parameter.grad = None

    def sync_model_weight_from_main_weight(self) -> None:
        """Quantize the sharded main weights into the FP8 payload DBuffers."""
        self._quantize_model_weight_from_main_weight()

    def _quantize_model_weight_from_main_weight(self) -> None:
        """Quantize through TE's ``cast_master_weights_to_fp8``."""
        main = self.main_weight.local_buffer
        cast_master_weights_to_fp8 = te_cast_master_weights_to_fp8()
        assert cast_master_weights_to_fp8 is not None

        model_weights = []
        master_weights = []
        start_offsets = []
        fsdp_shard_model_weights = []
        temps = []
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            tensor = fsdp_parameter.unsharded
            owned_range = self.main_weight._get_owned_range(index)
            height, width = self.main_weight.layout.tensor_shapes[index]
            temp = allocate_quantize_temp(tensor, height, width, self.main_weight.device)
            temps.append((temp, index, owned_range))
            model_weights.append(temp)
            if owned_range is None:
                # TE skips the cast for master_weight=None. The empty fragments
                # represent tensors whose rows are owned by other ranks.
                master_weights.append(None)
                start_offsets.append(0)
                fsdp_shard_model_weights.append(
                    (
                        temp._rowwise_data.reshape(-1)[:0],
                        temp._columnwise_data.reshape(-1)[:0],
                    )
                )
                continue
            numel = owned_range.numel
            start_offset = owned_range.tensor_relative_offset
            master_weights.append(main.narrow(0, owned_range.buffer_relative_offset, numel))
            start_offsets.append(start_offset)
            end_offset = start_offset + numel
            fsdp_shard_model_weights.append(
                (
                    temp._rowwise_data.reshape(-1)[start_offset:end_offset],
                    temp._columnwise_data.reshape(-1)[start_offset:end_offset],
                )
            )

        gather_axis = changed_mesh_axis(
            self._model_weight_placements, tuple(Replicate() for _ in range(self.mesh.ndim))
        )
        if gather_axis is None:
            raise RuntimeError("FSDP fp8 parameter quantize requires a changed placement axis.")
        cast_master_weights_to_fp8(
            model_weights=model_weights,
            master_weights=master_weights,
            start_offsets=start_offsets,
            group=self.mesh.get_group(gather_axis),
            fsdp_shard_model_weights=fsdp_shard_model_weights,
        )

        # Persist only each rank's shard; temporary full-size tensors can die here.
        for temp, index, owned_range in temps:
            if owned_range is None:
                continue
            numel = owned_range.numel
            rows_local = numel // temp.shape[-1]
            start_offset = owned_range.tensor_relative_offset
            end_offset = start_offset + numel
            self._rowwise_buffer.get_local_tensor(index).copy_(
                temp._rowwise_data.reshape(-1)[start_offset:end_offset].view(rows_local, -1)
            )
            self._colwise_buffer.get_local_tensor(index).copy_(
                temp._columnwise_data.reshape(-1)[start_offset:end_offset].view(rows_local, -1)
            )

    def unshard_parameters(self, orientation: str = "rowwise") -> None:
        """Gather and bind both MXFP8 payload orientations.

        TE primary-weight layers require row-wise and column-wise payloads even
        during forward, so ``orientation`` is accepted for lifecycle parity but
        both payloads are gathered.
        """
        del orientation
        for source, target in (
            (self._rowwise_buffer, self._unsharded_rowwise),
            (self._colwise_buffer, self._unsharded_colwise),
        ):
            target.reallocate_storage()
            gather_axis = changed_mesh_axis(source.placements, target.placements)
            if gather_axis is None:
                raise RuntimeError(
                    "FSDP fp8 parameter unshard requires a changed placement axis."
                )
            source.redistribute(target.placements, out=target)
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            tensor = fsdp_parameter.unsharded
            set_rowwise_payload(tensor, self._unsharded_rowwise.get_local_tensor(index))
            set_columnwise_payload(tensor, self._unsharded_colwise.get_local_tensor(index))
        self._switch_to_unsharded_parameters()

    def release_unsharded_storage(self) -> None:
        """Detach FP8 payloads and release gathered buffers."""
        for fsdp_parameter in self.fsdp_parameters:
            clear_payloads(fsdp_parameter.unsharded)
        self._unsharded_rowwise.release_storage()
        self._unsharded_colwise.release_storage()


def _get_parameter_owner(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a root-module-relative parameter FQN to its direct owner."""
    module_name, separator, parameter_name = name.rpartition(".")
    owner = module.get_submodule(module_name) if separator else module
    return owner, parameter_name
