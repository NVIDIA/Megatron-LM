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

"""Distributed tensor buffers for Megatron-FSDP."""

import dataclasses
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import HAVE_TE_MXFP8TENSOR
from .layout import GlobalLayout, Shape, non_leading_numel
from .placement import changed_mesh_axis

if HAVE_TE_MXFP8TENSOR:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
else:
    tex = None
    MXFP8Quantizer = None
    MXFP8Tensor = None

_MXFP8_BLOCK_SIZE = 32


def is_mxfp8_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is a TE MXFP8 tensor with both physical data representations."""
    return (
        HAVE_TE_MXFP8TENSOR
        and isinstance(tensor, MXFP8Tensor)
        and tensor._rowwise_data is not None
        and tensor._columnwise_data is not None
    )


@dataclasses.dataclass(frozen=True)
class _OwnedRange:
    numel: int
    tensor_relative_offset: int
    buffer_relative_offset: int


@dataclasses.dataclass
class _MXFP8LocalTensor:
    """Packed MXFP8 physical planes and the TE views they define."""

    rowwise_data: torch.Tensor
    columnwise_data: torch.Tensor
    rowwise_scale_inv: torch.Tensor
    columnwise_scale_inv: torch.Tensor
    shapes: tuple[torch.Size | None, ...]
    rowwise_scale_shapes: tuple[torch.Size | None, ...]
    columnwise_scale_shapes: tuple[torch.Size | None, ...]
    offsets: tuple[int | None, ...]
    scale_offsets: tuple[int, ...]
    columnwise_scale_offsets: tuple[int, ...]
    tensors: tuple[torch.Tensor | None, ...]

    @classmethod
    def allocate(
        cls,
        shapes: tuple[torch.Size | None, ...],
        data_numel: int,
        data_offsets: tuple[int | None, ...],
        device: torch.device | str,
    ) -> "_MXFP8LocalTensor":
        """Allocate packed MXFP8 planes for local logical tensor shapes."""
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        placeholders = [
            (
                None
                if shape is None
                else quantizer(torch.zeros(shape, dtype=torch.bfloat16, device=device))
            )
            for shape in shapes
        ]
        rowwise_scale_shapes = tuple(
            None if tensor is None else tensor._rowwise_scale_inv.shape for tensor in placeholders
        )
        columnwise_scale_shapes = tuple(
            None if tensor is None else tensor._columnwise_scale_inv.shape
            for tensor in placeholders
        )

        def offsets(sizes: Iterable[int]) -> tuple[int, ...]:
            result = [0]
            for size in sizes:
                result.append(result[-1] + size)
            return tuple(result)

        scale_offsets = offsets(
            0 if shape is None else shape.numel() for shape in rowwise_scale_shapes
        )
        columnwise_scale_offsets = offsets(
            0 if shape is None else shape.numel() for shape in columnwise_scale_shapes
        )
        storage = cls(
            rowwise_data=torch.empty(data_numel, dtype=torch.uint8, device=device),
            columnwise_data=torch.empty(data_numel, dtype=torch.uint8, device=device),
            rowwise_scale_inv=torch.empty(scale_offsets[-1], dtype=torch.uint8, device=device),
            columnwise_scale_inv=torch.empty(
                columnwise_scale_offsets[-1], dtype=torch.uint8, device=device
            ),
            shapes=shapes,
            rowwise_scale_shapes=rowwise_scale_shapes,
            columnwise_scale_shapes=columnwise_scale_shapes,
            offsets=data_offsets,
            scale_offsets=scale_offsets,
            columnwise_scale_offsets=columnwise_scale_offsets,
            tensors=(),
        )
        storage.tensors = tuple(storage._make_tensor(index) for index in range(len(shapes)))
        return storage

    def physical_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the packed physical tensors that own all MXFP8 storage."""
        return (
            self.rowwise_data,
            self.columnwise_data,
            self.rowwise_scale_inv,
            self.columnwise_scale_inv,
        )

    def tensor(self, index: int) -> torch.Tensor | None:
        """Return TE's logical MXFP8 tensor view for one packed member."""
        return self.tensors[index]

    def _make_tensor(self, index: int) -> torch.Tensor | None:
        """Create one persistent TE wrapper over this payload's plane views."""
        shape = self.shapes[index]
        start = self.offsets[index]
        if shape is None or start is None:
            return None
        end = start + shape.numel()
        scale_start, scale_end = self.scale_offsets[index : index + 2]
        columnwise_scale_start, columnwise_scale_end = self.columnwise_scale_offsets[
            index : index + 2
        ]
        return MXFP8Tensor(
            shape=shape,
            dtype=torch.bfloat16,
            rowwise_data=self.rowwise_data.narrow(0, start, end - start).view(shape),
            rowwise_scale_inv=self.rowwise_scale_inv.narrow(
                0, scale_start, scale_end - scale_start
            ).view(self.rowwise_scale_shapes[index]),
            columnwise_data=self.columnwise_data.narrow(0, start, end - start).view(shape),
            columnwise_scale_inv=self.columnwise_scale_inv.narrow(
                0, columnwise_scale_start, columnwise_scale_end - columnwise_scale_start
            ).view(self.columnwise_scale_shapes[index]),
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3),
            with_gemm_swizzled_scales=False,
            device=self.rowwise_data.device,
            requires_grad=False,
        )


def _validate_placements(placements: Iterable[Placement]) -> None:
    """Validate DBuffer placements form a supported contiguous local layout."""
    seen_shard = False
    for placement in placements:
        if not isinstance(placement, (Replicate, Partial, Shard)):
            raise TypeError(f"Unsupported DBuffer placement: {placement!r}.")
        if isinstance(placement, Shard):
            if placement.dim != 0:
                raise NotImplementedError(
                    f"DBuffer supports only dim-0 Shard placements, got {placement!r}."
                )
            seen_shard = True
        elif seen_shard:
            raise ValueError(
                "Shard placements must be a suffix of the placement list so each "
                "local buffer is a contiguous global-buffer range."
            )


def _get_reduce_op(partial_placement: Partial) -> dist.ReduceOp.RedOpType:
    """Convert a DTensor Partial reduction name to a torch.distributed op."""
    reduce_ops = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.AVG}
    return reduce_ops[partial_placement.reduce_op]


class DBuffer:
    """A distributed buffer holding a group of logical tensors.

    DBuffer is analogous to DTensor, but manages a group of logical tensors in
    one local storage tensor, or a TE-owned MXFP8 logical tensor with its
    physical data and scale tensors. It stores enough metadata to return
    per-tensor views, redistribute the buffer across mesh axes, and materialize
    per-tensor DTensors for optimizer state or distributed checkpointing.
    """

    # DBuffer owns only the data-parallel sub-mesh. Higher-level callers, such as
    # FsdpParameterGroup, should extend returned DTensors with tensor-parallel mesh axes
    # because TP sharding metadata lives on nn.Parameter in MCore/TransformerEngine.
    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    layout: GlobalLayout
    offset: int
    local_tensor: torch.Tensor | _MXFP8LocalTensor

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
        dtype: torch.dtype,
        device: torch.device | str,
        *,
        block_size: int = 1,
    ) -> None:
        """Create a DBuffer and allocate its local buffer.

        Args:
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            tensor_shapes: Global shapes for each logical tensor in this buffer.
            dtype: Physical dtype for the local buffer. ``torch.uint8`` creates
                an MXFP8 buffer with TE's default quantizer.
            device: Device for the local buffer.
        """
        placements = tuple(placements)
        if len(placements) != mesh.ndim:
            raise ValueError(
                f"Expected {mesh.ndim} placements for device mesh, got {len(placements)}."
            )
        _validate_placements(placements)
        self.mesh = mesh
        self.placements = placements

        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        self.layout = GlobalLayout.build(
            tensor_shapes, dp_size=self.mesh.size(), block_size=block_size
        )

        self.offset, local_numel = self.layout.get_local_range(self.mesh, self.placements)
        self._local_numel = local_numel
        if dtype != torch.uint8:
            self.local_tensor = torch.empty(local_numel, dtype=dtype, device=device)
            return

        if not HAVE_TE_MXFP8TENSOR:
            raise RuntimeError("MXFP8 DBuffer construction requires Transformer Engine.")
        if any(len(shape) != 2 for shape in tensor_shapes):
            raise NotImplementedError("Experimental MXFP8 MFSDP supports 2D tensors only.")
        if mesh.ndim != 1 or any(
            not isinstance(placement, (Replicate, Shard)) for placement in placements
        ):
            raise NotImplementedError(
                "Experimental MXFP8 MFSDP supports one-dimensional Replicate/Shard meshes."
            )

        local_shapes: list[torch.Size | None] = []
        data_offsets: list[int | None] = []
        for index, shape in enumerate(tensor_shapes):
            owned_range = self._get_owned_range(index)
            if owned_range is None:
                local_shapes.append(None)
                data_offsets.append(None)
                continue
            row_size = shape[1]
            if owned_range.tensor_relative_offset % row_size or owned_range.numel % row_size:
                raise NotImplementedError("MXFP8 MFSDP requires dim-0 tensor shards.")
            local_shape = torch.Size((owned_range.numel // row_size, row_size))
            local_end = owned_range.tensor_relative_offset + owned_range.numel
            if (
                owned_range.tensor_relative_offset % (128 * row_size)
                or (local_end < shape.numel() and local_end % (128 * row_size))
                or local_shape[1] % _MXFP8_BLOCK_SIZE
            ):
                raise NotImplementedError(
                    "MXFP8 MFSDP requires 32-column tensors and 128-row internal shard boundaries."
                )
            local_shapes.append(local_shape)
            data_offsets.append(owned_range.buffer_relative_offset)
        self.local_tensor = _MXFP8LocalTensor.allocate(
            tuple(local_shapes), local_numel, tuple(data_offsets), device
        )

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the buffer's physical storage."""
        if self.is_mxfp8:
            return self.local_tensor.rowwise_data.dtype
        return self.local_tensor.dtype

    @property
    def device(self) -> torch.device:
        """Device of the stored tensor."""
        if self.is_mxfp8:
            return self.local_tensor.rowwise_data.device
        return self.local_tensor.device

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether the local buffer is allocated from symmetric memory."""
        return (
            not self.is_mxfp8
            and hasattr(symm_mem, "is_symm_mem_tensor")
            and symm_mem.is_symm_mem_tensor(self.local_tensor)
        )

    @property
    def is_mxfp8(self) -> bool:
        """Whether this buffer is backed by a TE MXFP8 logical tensor."""
        return isinstance(self.local_tensor, _MXFP8LocalTensor)

    def _mxfp8_physical_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the physical tensors owned by this buffer's TE wrapper."""
        assert self.is_mxfp8
        return self.local_tensor.physical_tensors()

    @staticmethod
    def _compact_mxfp8_scale(tensor: torch.Tensor, *, columnwise: bool) -> torch.Tensor:
        """Return the valid, non-padded region of one TE MXFP8 scale tensor."""
        if columnwise:
            return tensor._columnwise_scale_inv[
                : tensor.shape[0] // _MXFP8_BLOCK_SIZE, : tensor.shape[1]
            ].contiguous()
        return tensor._rowwise_scale_inv[
            : tensor.shape[0], : tensor.shape[1] // _MXFP8_BLOCK_SIZE
        ].contiguous()

    @staticmethod
    def _unpack_mxfp8_scale(destination: torch.Tensor, source: torch.Tensor) -> None:
        """Copy compact MXFP8 scales into TE's padded scale allocation."""
        destination.zero_()
        destination[: source.shape[0], : source.shape[1]].copy_(source)

    def reallocate_storage(self) -> None:
        """Restore the local buffer's backing storage to its logical size."""
        # The allocator may hand back a different address than release_storage() freed.
        # Tensors sharing this Storage read its pointer on each access, so views into the
        # buffer -- including ones autograd saved -- follow it to the new allocation.
        if self.is_mxfp8:
            for tensor in self._mxfp8_physical_tensors():
                self._resize_storage(tensor, tensor.numel())
            return
        self._resize_storage(self.local_tensor, self.local_tensor.numel())

    def release_storage(self) -> None:
        """Release local buffer storage without replacing the Storage object."""
        # Autograd may save views that share this Storage object. Resizing the
        # existing Storage releases the allocation while preserving those aliases
        # for a later reallocate_storage().
        if self.is_mxfp8:
            for tensor in self._mxfp8_physical_tensors():
                self._resize_storage(tensor, 0)
            return
        self._resize_storage(self.local_tensor, 0)

    def rendezvous(self, mesh_axis: int) -> None:
        """Rendezvous this local buffer for symmetric-memory collectives."""
        group = self.mesh.get_group(mesh_axis)
        symm_mem.rendezvous(self.local_tensor, group=group.group_name)

    @staticmethod
    def _resize_storage(tensor: torch.Tensor, numel: int) -> None:
        tensor.untyped_storage().resize_(numel * tensor.element_size())

    def _get_owned_range(self, tensor_index: int) -> _OwnedRange | None:
        """Return this buffer's owned range for logical tensor ``tensor_index``."""
        tensor_start = self.layout.tensor_to_offset[tensor_index]
        tensor_end = tensor_start + self.layout.tensor_shapes[tensor_index].numel()
        buffer_start = self.offset
        buffer_end = self.offset + self._local_numel

        overlap_start = max(tensor_start, buffer_start)
        overlap_end = min(tensor_end, buffer_end)
        if overlap_start >= overlap_end:
            return None

        return _OwnedRange(
            numel=overlap_end - overlap_start,
            tensor_relative_offset=overlap_start - tensor_start,
            buffer_relative_offset=overlap_start - buffer_start,
        )

    @classmethod
    def from_local(
        cls,
        local_buffer: torch.Tensor,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        layout: GlobalLayout,
    ) -> "DBuffer":
        """Create a DBuffer from an existing local buffer.

        Args:
            local_buffer: Contiguous local tensor storage for this rank. DBuffer
                uses it directly in collectives such as all-gather and
                reduce-scatter, which are efficient with contiguous tensors.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            layout: Existing global layout for the logical tensors in this buffer.

        Returns:
            A DBuffer that reuses ``local_buffer`` without allocating storage.
        """
        placements = tuple(placements)
        if len(placements) != mesh.ndim:
            raise ValueError(
                f"Expected {mesh.ndim} placements for device mesh, got {len(placements)}."
            )
        _validate_placements(placements)
        if local_buffer.dim() != 1:
            raise ValueError("local_buffer must be a flat 1D tensor.")
        if not local_buffer.is_contiguous():
            raise ValueError("local_buffer must be contiguous for collective operations.")

        offset, local_numel = layout.get_local_range(mesh, placements)
        if local_buffer.numel() != local_numel:
            raise ValueError(
                f"Expected local_buffer with {local_numel} elements, got "
                f"{local_buffer.numel()}."
            )

        buffer = cls.__new__(cls)
        buffer.mesh = mesh
        buffer.placements = placements
        buffer.layout = layout
        buffer.offset = offset
        buffer.local_tensor = local_buffer
        buffer._local_numel = local_numel
        return buffer

    def view(self, placements: Iterable[Placement]) -> "DBuffer":
        """Return a storage-sharing buffer with supported ``placements``.

        Views preserve placements, relabel a full local buffer, or locally slice
        one full local buffer to Flat. A view that changes a Partial placement is
        only a storage destination: callers must populate it with a reduction
        before reading it.
        """
        if self.is_mxfp8:
            raise NotImplementedError("MXFP8 DBuffer views are not supported; use redistribute().")
        placements = tuple(placements)
        if len(placements) != self.mesh.ndim:
            raise ValueError(
                f"Expected {self.mesh.ndim} placements for device mesh, got {len(placements)}."
            )

        changed_axis = changed_mesh_axis(self.placements, placements)
        if changed_axis is None:
            return self
        source_placement = self.placements[changed_axis]
        destination_placement = placements[changed_axis]
        if isinstance(source_placement, (Replicate, Partial)) and isinstance(
            destination_placement, Shard
        ):
            offset, local_numel = self.layout.get_local_range(self.mesh, placements)
            local_offset = offset - self.offset
            if local_offset < 0 or local_offset + local_numel > self.local_tensor.numel():
                raise RuntimeError("DBuffer view is not contained in its source local buffer.")
            return DBuffer.from_local(
                self.local_tensor.narrow(0, local_offset, local_numel),
                self.mesh,
                placements,
                self.layout,
            )
        if isinstance(source_placement, Partial) and isinstance(destination_placement, Replicate):
            return DBuffer.from_local(self.local_tensor, self.mesh, placements, self.layout)
        raise ValueError(
            "DBuffer.view() supports identical placements, a Partial-to-Replicate relabel, "
            "or a Replicate/Partial-to-Flat slice, "
            f"got {self.placements!r} -> {placements!r}."
        )

    @classmethod
    def distribute_tensors(
        cls,
        tensors: Iterable[torch.Tensor],
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        *,
        block_size: int = 1,
    ) -> "DBuffer":
        """Distribute full local tensors into a DBuffer.

        Args:
            tensors: Full tensors available on this rank. Meta tensors contribute
                shape and dtype metadata but no values.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.

        Returns:
            A DBuffer whose real local storage matches ``placements``. Ranges
            corresponding to meta tensors are left uninitialized.
        """
        tensors = tuple(tensor.detach().contiguous() for tensor in tensors)
        if not tensors:
            raise ValueError("DBuffer.distribute_tensors() requires at least one tensor.")

        dtype = tensors[0].dtype
        for tensor in tensors:
            if tensor.dtype != dtype:
                raise ValueError("All tensors in a DBuffer must have the same dtype.")

        tensor_shapes = tuple(tensor.shape for tensor in tensors)
        buffer = cls(
            mesh=mesh,
            placements=placements,
            tensor_shapes=tensor_shapes,
            dtype=dtype,
            device=mesh.device_type,
            block_size=block_size,
        )
        # Only logical tensor ranges are initialized. Padding and layout gaps are not
        # observable through get_local_tensor() and can remain unspecified.
        for index, tensor in enumerate(tensors):
            owned_range = buffer._get_owned_range(index)
            if owned_range is None or tensor.is_meta:
                continue

            source_slice = tensor.view(-1).narrow(
                0, owned_range.tensor_relative_offset, owned_range.numel
            )
            buffer.local_tensor.narrow(
                0, owned_range.buffer_relative_offset, owned_range.numel
            ).copy_(source_slice)
        return buffer

    def _create_or_validate_out(
        self,
        out: "DBuffer | None",
        *,
        placements: Iterable[Placement] | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DBuffer":
        if placements is None:
            placements = self.placements
        else:
            placements = tuple(placements)
        if dtype is None:
            dtype = self.dtype
        if out is None:
            return DBuffer(
                mesh=self.mesh,
                placements=placements,
                tensor_shapes=self.layout.tensor_shapes,
                dtype=dtype,
                device=self.device,
                block_size=self.layout.block_size,
            )

        if out.mesh != self.mesh:
            raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
        if out.placements != placements:
            raise ValueError(f"Expected out placements {placements!r}, got {out.placements!r}.")
        if out.layout != self.layout:
            raise ValueError(f"Expected out layout {self.layout!r}, got {out.layout!r}.")
        if out.dtype != dtype:
            raise ValueError(f"Expected out dtype {dtype}, got {out.dtype}.")
        if out.device != self.device:
            raise ValueError(f"Expected out device {self.device}, got {out.device}.")
        return out

    def sync_from_main(self, main_weight: "DBuffer") -> None:
        """Refresh this compute buffer from optimizer-layout master weights."""
        if self.is_mxfp8:
            with torch.no_grad():
                for index in range(len(self.layout.tensor_shapes)):
                    destination = self._get_mxfp8_tensor(index)
                    if destination is not None:
                        destination.quantize_(main_weight.get_local_tensor(index))
        else:
            main_weight.cast(self.dtype, out=self)

    def materialize_unsharded_from(self, source: "DBuffer") -> "DBuffer":
        """Populate this full-parameter buffer from ``source`` when needed."""
        # Ordinary parameters can be rebound to source's flat views. An MXFP8
        # parameter permanently owns this DBuffer's TE wrapper, so it must be
        # refreshed even when both buffers have replicated placements.
        if source.placements == self.placements and not self.is_mxfp8:
            return source

        self.reallocate_storage()
        if self.is_mxfp8:
            source.redistribute(self.placements, out=self)
        else:
            # This storage backs parameter views possibly saved by autograd.
            # Preserve its version while FSDP writes the freshly materialized values.
            with torch.autograd._unsafe_preserve_version_counter(self.local_tensor):
                source.redistribute(self.placements, out=self)
        return self

    def initialize_unsharded_parameter(self, parameter: "torch.nn.Parameter", index: int) -> None:
        """Install this buffer's initial local tensor into one module parameter."""
        if parameter.is_meta or self.is_mxfp8:
            local_tensor = self.get_local_tensor(index)
            materialized = torch.nn.Parameter(local_tensor, requires_grad=parameter.requires_grad)
            torch.utils.swap_tensors(parameter, materialized)
        else:
            parameter.data = self.get_local_tensor(index)
            parameter.grad = None

    def install_unsharded_parameter_views(self, parameters: Iterable["torch.nn.Parameter"]) -> None:
        """Point ordinary unsharded parameters at this buffer's current local views."""
        if self.is_mxfp8:
            # TE wrappers installed by initialize_unsharded_parameter() retain views
            # into the packed physical planes and are updated in place.
            return
        for index, parameter in enumerate(parameters):
            parameter.data = self.get_local_tensor(index)

    def cast(self, dtype: torch.dtype, *, out: "DBuffer | None" = None) -> "DBuffer":
        """Return this buffer with the same layout and placements in ``dtype``."""
        if self.is_mxfp8:
            raise NotImplementedError("Cast MXFP8 DBuffers by requantizing from their main weight.")
        if self.dtype == dtype and out is None:
            return self

        destination = self._create_or_validate_out(out, dtype=dtype)
        destination.local_tensor.copy_(self.local_tensor)
        return destination

    def redistribute(
        self, new_placements: Iterable[Placement], *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Redistribute this buffer to ``new_placements``.

        This dispatcher supports the one-axis transitions:
        Flat -> Replicate, Partial -> Replicate, Partial -> Flat,
        Replicate -> Flat, and Replicate -> Partial. Other placement changes are
        intentionally unsupported.
        """
        new_placements = tuple(new_placements)
        if len(new_placements) != self.mesh.ndim:
            raise ValueError(
                f"Expected {self.mesh.ndim} placements for device mesh, got "
                f"{len(new_placements)}."
            )
        _validate_placements(new_placements)

        if self.is_mxfp8:
            return self._redistribute_mxfp8(new_placements, out=out)

        changed_axis = changed_mesh_axis(self.placements, new_placements)
        if changed_axis is None:
            if out is None:
                return self
            out = self._create_or_validate_out(out, placements=new_placements)
            out.local_tensor.copy_(self.local_tensor)
            return out

        axis = changed_axis
        old_placement = self.placements[axis]
        new_placement = new_placements[axis]
        if isinstance(old_placement, Shard) and isinstance(new_placement, Replicate):
            return self.allgather(axis, out=out)
        if isinstance(old_placement, Partial) and isinstance(new_placement, Replicate):
            return self.allreduce(axis, out=out)
        if isinstance(old_placement, Partial) and isinstance(new_placement, Shard):
            return self.reduce_scatter(axis, new_placement, out=out)
        if isinstance(old_placement, Replicate) and isinstance(new_placement, Shard):
            view = self.view(new_placements)
            if out is None:
                return view
            out = self._create_or_validate_out(out, placements=new_placements)
            out.local_tensor.copy_(view.local_tensor)
            return out
        if isinstance(old_placement, Replicate) and isinstance(new_placement, Partial):
            # Replicate and Partial share the same local layout, so relabel the
            # buffer without communication. Value-preserving for AVG only -- the
            # mean of identical per-rank locals is that value; SUM would need a
            # 1/axis_size rescale, which no caller needs.
            if new_placement.reduce_op != "avg":
                raise NotImplementedError(
                    "Replicate -> Partial redistribute supports AVG only, got "
                    f"{new_placement.reduce_op!r}."
                )
            if out is not None:
                raise NotImplementedError(
                    "Replicate -> Partial redistribute does not support an out buffer."
                )
            return DBuffer.from_local(self.local_tensor, self.mesh, new_placements, self.layout)
        raise NotImplementedError(
            "Unsupported DBuffer placement transition on axis "
            f"{axis}: {old_placement!r} -> {new_placement!r}."
        )

    def _redistribute_mxfp8(
        self, new_placements: tuple[Placement, ...], *, out: "DBuffer | None"
    ) -> "DBuffer":
        """Redistribute TE MXFP8 physical storage while preserving its wrapper."""
        if out is None or not out.is_mxfp8:
            raise ValueError(
                "MXFP8 DBuffer redistribution requires a preallocated MXFP8 out buffer."
            )
        if out.mesh != self.mesh or out.placements != new_placements:
            raise ValueError(
                "MXFP8 DBuffer out buffer must match the requested mesh and placements."
            )
        changed_axis = changed_mesh_axis(self.placements, new_placements)
        if changed_axis is None:
            for destination, source in zip(
                out._mxfp8_physical_tensors(), self._mxfp8_physical_tensors()
            ):
                destination.copy_(source)
            return out

        source_placement = self.placements[changed_axis]
        destination_placement = new_placements[changed_axis]
        if isinstance(source_placement, Shard) and isinstance(destination_placement, Replicate):
            return self.allgather(changed_axis, out=out)
        if isinstance(source_placement, Replicate) and isinstance(destination_placement, Shard):
            for destination, source in zip(
                out._mxfp8_physical_tensors()[:2], self._mxfp8_physical_tensors()[:2]
            ):
                destination.copy_(source.narrow(0, out.offset - self.offset, out._local_numel))
            for index in range(len(self.layout.tensor_shapes)):
                destination = out._get_mxfp8_tensor(index)
                if destination is None:
                    continue
                owned_range = out._get_owned_range(index)
                assert owned_range is not None
                source = self.get_local_tensor(index)
                row_size = source.shape[1]
                destination.copy_(
                    source.narrow(
                        0, owned_range.tensor_relative_offset // row_size, destination.shape[0]
                    )
                )
            return out
        raise NotImplementedError(
            f"Unsupported MXFP8 DBuffer placement transition: {source_placement!r} -> "
            f"{destination_placement!r}."
        )

    def allgather(self, mesh_axis: int, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-gather a sharded axis into Replicate placement."""
        if not isinstance(self.placements[mesh_axis], Shard):
            raise ValueError(
                f"allgather() currently requires a Shard placement on axis {mesh_axis!r}."
            )

        if self.is_mxfp8:
            return self._allgather_mxfp8(mesh_axis, out=out)

        placements = list(self.placements)
        placements[mesh_axis] = Replicate()
        _validate_placements(placements)
        out = self._create_or_validate_out(out, placements=placements)
        # Symmetric-memory registration is scoped to the collective's process
        # group, so rendezvous the output on the same mesh axis as the all-gather.
        if out.is_symmetric_memory:
            out.rendezvous(mesh_axis)
        dist.all_gather_into_tensor(
            output_tensor=out.local_tensor,
            input_tensor=self.local_tensor,
            group=self.mesh.get_group(mesh_axis),
        )
        return out

    def _allgather_mxfp8(self, mesh_axis: int, *, out: "DBuffer | None") -> "DBuffer":
        """All-gather an MXFP8 wrapper's data and compact valid scale regions."""
        if out is None or not out.is_mxfp8:
            raise ValueError("MXFP8 DBuffer all-gather requires a preallocated MXFP8 out buffer.")
        if not isinstance(out.placements[mesh_axis], Replicate):
            raise ValueError("MXFP8 all-gather out buffer must replicate the gathered mesh axis.")
        group = self.mesh.get_group(mesh_axis)
        for destination, source in zip(
            out._mxfp8_physical_tensors()[:2], self._mxfp8_physical_tensors()[:2]
        ):
            dist.all_gather_into_tensor(destination, source, group=group)
        for index, shape in enumerate(self.layout.tensor_shapes):
            source_tensor = self._get_mxfp8_tensor(index)
            destination_tensor = out.get_local_tensor(index)
            owned_range = self._get_owned_range(index)
            local_rows = 0 if owned_range is None else owned_range.numel // shape[1]
            row_counts = torch.empty(
                self.mesh.size(mesh_axis), dtype=torch.int64, device=self.device
            )
            dist.all_gather_into_tensor(
                row_counts,
                torch.tensor([local_rows], dtype=torch.int64, device=self.device),
                group=group,
            )
            max_rows = int(row_counts.max())
            for columnwise in (False, True):
                scale_rows = local_rows // _MXFP8_BLOCK_SIZE if columnwise else local_rows
                max_scale_rows = max_rows // _MXFP8_BLOCK_SIZE if columnwise else max_rows
                scale_columns = shape[1] if columnwise else shape[1] // _MXFP8_BLOCK_SIZE
                source = torch.zeros(
                    (max_scale_rows, scale_columns), dtype=torch.uint8, device=self.device
                )
                if source_tensor is not None:
                    source[:scale_rows].copy_(
                        self._compact_mxfp8_scale(source_tensor, columnwise=columnwise)
                    )
                gathered = torch.empty(
                    (self.mesh.size(mesh_axis), max_scale_rows, scale_columns),
                    dtype=torch.uint8,
                    device=self.device,
                )
                dist.all_gather_into_tensor(gathered, source, group=group)
                compact = torch.empty(
                    (shape[0] // _MXFP8_BLOCK_SIZE if columnwise else shape[0], scale_columns),
                    dtype=torch.uint8,
                    device=self.device,
                )
                offset = 0
                for rank, rows in enumerate(row_counts.tolist()):
                    count = rows // _MXFP8_BLOCK_SIZE if columnwise else rows
                    compact[offset : offset + count].copy_(gathered[rank, :count])
                    offset += count
                destination = (
                    destination_tensor._columnwise_scale_inv
                    if columnwise
                    else destination_tensor._rowwise_scale_inv
                )
                self._unpack_mxfp8_scale(destination, compact)
        return out

    def allreduce(self, mesh_axis: int, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-reduce a Partial axis into Replicate placement."""
        axis = mesh_axis
        partial_placement = self.placements[axis]
        if not isinstance(partial_placement, Partial):
            raise ValueError(f"allreduce() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = Replicate()
        out = self._create_or_validate_out(out, placements=placements)
        out.local_tensor.copy_(self.local_tensor)
        dist.all_reduce(
            out.local_tensor, op=_get_reduce_op(partial_placement), group=self.mesh.get_group(axis)
        )
        return out

    def reduce_scatter(
        self, mesh_axis: int, new_placement: Placement, *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Reduce-scatter a Partial axis into ``new_placement``."""
        axis = mesh_axis
        if not isinstance(new_placement, Shard):
            raise NotImplementedError("DBuffer currently supports reduce_scatter() to Shard only.")
        partial_placement = self.placements[axis]
        if not isinstance(partial_placement, Partial):
            raise ValueError(f"reduce_scatter() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        _validate_placements(placements)
        out = self._create_or_validate_out(out, placements=placements)
        reduce_op = _get_reduce_op(partial_placement)
        # Symmetric-memory MFSDP requires this detector, but ordinary DBuffer
        # reductions remain supported on older PyTorch versions that lack it.
        if self.is_symmetric_memory:
            self.rendezvous(axis)
            # NCCL symmetric-memory reduce-scatter selects its symmetric kernel
            # for SUM. Preserve the placement's AVG semantics by scaling the
            # SUM result after the collective.
            if reduce_op == dist.ReduceOp.AVG:
                reduce_op = dist.ReduceOp.SUM
        dist.reduce_scatter_tensor(
            output=out.local_tensor,
            input=self.local_tensor,
            op=reduce_op,
            group=self.mesh.get_group(axis),
        )
        if self.is_symmetric_memory and partial_placement.reduce_op == "avg":
            out.local_tensor.div_(self.mesh.size(axis))
        return out

    def get_local_tensor(self, index: int) -> torch.Tensor:
        """Return this rank's local view for logical tensor ``index``.

        Flat placements shard dim 0, so the returned view preserves all
        non-leading dimensions and only changes the leading dimension.
        """
        shape = self.layout.tensor_shapes[index]
        owned_range = self._get_owned_range(index)

        if self.is_mxfp8:
            tensor = self._get_mxfp8_tensor(index)
            if tensor is not None:
                return tensor
            return torch.empty((0, *shape[1:]), dtype=torch.uint8, device=self.device)

        row_size = non_leading_numel(shape)
        if owned_range is None:
            empty_shape = torch.Size((0, *shape[1:]))
            return torch.empty(empty_shape, dtype=self.dtype, device=self.device)

        if owned_range.tensor_relative_offset % row_size != 0 or owned_range.numel % row_size != 0:
            raise RuntimeError(
                f"Local tensor shard for tensor {index} does not preserve dim-0 boundaries."
            )
        local_shape = torch.Size((owned_range.numel // row_size, *shape[1:]))
        return self.local_tensor.narrow(
            0, owned_range.buffer_relative_offset, owned_range.numel
        ).view(local_shape)

    def _get_mxfp8_tensor(self, index: int) -> torch.Tensor | None:
        """Return this rank's TE wrapper for one logical MXFP8 tensor, if owned."""
        assert self.is_mxfp8
        return self.local_tensor.tensor(index)

    def get_dtensor(self, index: int) -> DTensor:
        """Return logical tensor ``index`` as a DTensor."""
        local_tensor = self.get_local_tensor(index)
        tensor_shape = self.layout.tensor_shapes[index]
        # Keep internal storage details (e.g. Flat and BlockAtomic) out of DTensor placements.
        dtensor_placements = tuple(
            Shard(placement.dim) if isinstance(placement, Shard) else placement
            for placement in self.placements
        )
        return DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=self.mesh,
            placements=dtensor_placements,
            run_check=False,
            shape=tensor_shape,
            stride=local_tensor.stride(),
        )
