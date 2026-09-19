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
import itertools
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

from .layout import GlobalLayout, Shape, non_leading_numel
from .placement import Flat, PlacementReference, TensorAtomic, changed_mesh_axis


@dataclasses.dataclass(frozen=True)
class _OwnedRange:
    numel: int
    tensor_relative_offset: int
    buffer_relative_offset: int


@dataclasses.dataclass(frozen=True)
class _ShardRanges:
    """How one GlobalLayout splits across one mesh under one placement tuple, seen from this rank.

    ``axis_ranges[axis][rank]`` is the ``(offset, numel)`` range that ``rank`` on that mesh
    axis holds. On a Shard axis the sub-range is split across ranks; on a Replicate or
    Partial axis every rank holds the whole sub-range, so the table repeats one entry
    ``mesh.size(axis)`` times. Inner axes split the sub-range this rank owns on outer axes,
    so the table is rank-specific and must not be shared across ranks.
    """

    axis_ranges: tuple[tuple[tuple[int, int], ...], ...]
    # Start of this rank's local range in global element coordinates.
    offset: int
    # Length of this rank's local range in elements.
    numel: int

    @classmethod
    def build(
        cls, layout: GlobalLayout, mesh: DeviceMesh, placements: tuple[Placement, ...]
    ) -> "_ShardRanges":
        """Compute the per-axis shard table and this rank's local range.

        Axes are processed from innermost (last) to outermost (first), so each
        Shard axis subdivides the sub-range this rank holds on all inner axes.

        Args:
            layout: Global layout being sharded.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.

        Returns:
            The shard table together with this rank's ``(offset, numel)``.
        """
        if isinstance(layout.reference, TensorAtomic) and mesh.ndim != 1:
            raise NotImplementedError(
                f"TensorAtomic layouts require a 1-D mesh, got ndim={mesh.ndim}."
            )
        axis_ranges: list[tuple[tuple[int, int], ...]] = [()] * mesh.ndim
        offset, numel = 0, layout.size
        for axis in reversed(range(mesh.ndim)):
            placement = placements[axis]
            axis_size = mesh.size(axis)
            if isinstance(placement, Shard):
                _check_shard_compatible(layout.reference, placement)
                table = _split_range(layout, axis_size, offset, numel)
                offset, numel = table[mesh.get_local_rank(axis)]
            else:
                # Replicate / Partial: every rank on this axis holds the whole sub-range.
                table = ((offset, numel),) * axis_size
            axis_ranges[axis] = table
        return cls(tuple(axis_ranges), offset, numel)

    def is_uniform(self, axis: int = 0) -> bool:
        """Whether every rank's range on ``axis`` has the same numel."""
        return len({numel for _, numel in self.axis_ranges[axis]}) == 1


def _split_range(
    layout: GlobalLayout, dp_size: int, offset: int, numel: int
) -> tuple[tuple[int, int], ...]:
    """Per-rank (offset, numel) when one mesh axis of size ``dp_size`` shards the range.

    The range being split is ``[offset, offset + numel)``.
    """
    reference = layout.reference
    if isinstance(reference, TensorAtomic):
        # TensorAtomic is restricted to 1-D meshes, so the sub-range being split is
        # always the whole buffer and the segments can be rebuilt from the assignment.
        assert offset == 0 and numel == layout.size
        per_rank = [0] * dp_size
        for tensor_id, rank in reference.tensor_to_owner_rank.items():
            per_rank[rank] += layout.tensor_shapes[tensor_id].numel()
        starts = itertools.accumulate([0, *per_rank[:-1]])
        return tuple(zip(starts, per_rank))
    if numel % dp_size != 0:
        raise ValueError(f"Local range size {numel} is not divisible by axis size {dp_size}.")
    shard = numel // dp_size
    return tuple((offset + rank * shard, shard) for rank in range(dp_size))


def _check_shard_compatible(reference: PlacementReference, placement: Shard) -> None:
    """Reject Shard placements whose slicing disagrees with the layout's reference.

    A TensorAtomic layout can only be sharded by the identical TensorAtomic
    placement, and a Flat / BlockAtomic layout can never be sharded by
    TensorAtomic, because the two schemes place tensors at different offsets.
    """
    reference_is_ta = isinstance(reference, TensorAtomic)
    placement_is_ta = isinstance(placement, TensorAtomic)
    if reference_is_ta != placement_is_ta:
        raise ValueError(
            f"Placement {placement!r} is incompatible with layout reference {reference!r}."
        )
    if placement_is_ta and placement.tensor_to_owner_rank != reference.tensor_to_owner_rank:
        raise ValueError("TensorAtomic placement must match the layout's owner assignment.")


def _validate_placements(placements: Iterable[Placement]) -> None:
    """Validate DBuffer placements form a supported contiguous local layout."""
    seen_shard = False
    for placement in placements:
        if not isinstance(placement, (Replicate, Partial, Shard)):
            raise TypeError(f"Unsupported DBuffer placement: {placement!r}.")

        if isinstance(placement, TensorAtomic) and len(placements) > 1:
            raise NotImplementedError(
                "Currently, DBuffer only supports only 1d device mesh, "
                f"but got ndim={len(placements)}."
            )

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
    one local storage tensor. It stores enough metadata to return per-tensor
    views, redistribute the buffer across mesh axes, and materialize per-tensor
    DTensors for optimizer state or distributed checkpointing.
    """

    # DBuffer owns only the data-parallel sub-mesh. Higher-level callers, such as
    # FsdpParameterGroup, should extend returned DTensors with tensor-parallel mesh axes
    # because TP sharding metadata lives on nn.Parameter in MCore/TransformerEngine.
    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    layout: GlobalLayout
    local_buffer: torch.Tensor
    # Per-axis shard table for (layout, mesh, placements) as seen from this rank.
    _shard_ranges: _ShardRanges

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        layout: GlobalLayout,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Create a DBuffer and allocate its local buffer.

        Args:
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            layout: Global shapes, offsets, and allocation size for this buffer.
            dtype: Dtype for the local buffer.
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

        self.layout = layout
        self._shard_ranges = _ShardRanges.build(self.layout, self.mesh, self.placements)
        self.local_buffer = torch.empty(self._shard_ranges.numel, dtype=dtype, device=device)

    @property
    def offset(self) -> int:
        """Global element offset of the first element in this rank's local buffer."""
        return self._shard_ranges.offset

    @classmethod
    def empty(
        cls,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
        dtype: torch.dtype,
        device: torch.device | str,
        *,
        reference: PlacementReference = Flat(),
    ) -> "DBuffer":
        """Build a layout from logical tensor shapes and allocate its local buffer."""
        layout = GlobalLayout.build(
            tuple(torch.Size(shape) for shape in tensor_shapes),
            dp_size=mesh.size(),
            reference=reference,
        )
        return cls(mesh, placements, layout, dtype, device)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the local buffer."""
        return self.local_buffer.dtype

    @property
    def device(self) -> torch.device:
        """Device of the local buffer."""
        return self.local_buffer.device

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether the local buffer is allocated from symmetric memory."""
        return hasattr(symm_mem, "is_symm_mem_tensor") and symm_mem.is_symm_mem_tensor(
            self.local_buffer
        )

    def reallocate_storage(self) -> None:
        """Restore the local buffer's backing storage to its logical size."""
        # The allocator may hand back a different address than release_storage() freed.
        # Tensors sharing this Storage read its pointer on each access, so views into the
        # buffer -- including ones autograd saved -- follow it to the new allocation.
        self._resize_storage(self.local_buffer.numel())

    def release_storage(self) -> None:
        """Release local buffer storage without replacing the Storage object."""
        # Autograd may save views that share this Storage object. Resizing the
        # existing Storage releases the allocation while preserving those aliases
        # for a later reallocate_storage().
        self._resize_storage(0)

    def rendezvous(self, mesh_axis: int) -> None:
        """Rendezvous this local buffer for symmetric-memory collectives."""
        group = self.mesh.get_group(mesh_axis)
        symm_mem.rendezvous(self.local_buffer, group=group.group_name)

    def _resize_storage(self, numel: int) -> None:
        self.local_buffer.untyped_storage().resize_(numel * self.local_buffer.element_size())

    def _get_owned_range(self, tensor_index: int) -> _OwnedRange | None:
        """Return this buffer's owned range for logical tensor ``tensor_index``."""
        tensor_start = self.layout.tensor_to_offset[tensor_index]
        tensor_end = tensor_start + self.layout.tensor_shapes[tensor_index].numel()
        buffer_start = self.offset
        buffer_end = self.offset + self.local_buffer.numel()

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
        shard_ranges: _ShardRanges | None = None,
    ) -> "DBuffer":
        """Create a DBuffer from an existing local buffer.

        Args:
            local_buffer: Contiguous local tensor storage for this rank. DBuffer
                uses it directly in collectives such as all-gather and
                reduce-scatter, which are efficient with contiguous tensors.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            layout: Existing global layout for the logical tensors in this buffer.
            shard_ranges: Precomputed shard table for ``(layout, mesh, placements)``.
                Callers that already hold one (e.g. ``view()`` and ``redistribute()``)
                pass it to avoid recomputation; when ``None`` it is rebuilt.

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

        shard_ranges = (
            shard_ranges
            if shard_ranges is not None
            else _ShardRanges.build(layout, mesh, placements)
        )
        if local_buffer.numel() != shard_ranges.numel:
            raise ValueError(
                f"Expected local_buffer with {shard_ranges.numel} elements, got "
                f"{local_buffer.numel()}."
            )

        buffer = cls.__new__(cls)
        buffer.mesh = mesh
        buffer.placements = placements
        buffer.layout = layout
        buffer.local_buffer = local_buffer
        buffer._shard_ranges = shard_ranges
        return buffer

    def view(self, placements: Iterable[Placement]) -> "DBuffer":
        """Return a storage-sharing buffer with supported ``placements``.

        Views preserve placements, relabel a full local buffer, or locally slice
        one full local buffer to Flat. A view that changes a Partial placement is
        only a storage destination: callers must populate it with a reduction
        before reading it.
        """
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
            shard_ranges = _ShardRanges.build(self.layout, self.mesh, placements)
            local_offset = shard_ranges.offset - self.offset
            if local_offset < 0 or local_offset + shard_ranges.numel > self.local_buffer.numel():
                raise RuntimeError("DBuffer view is not contained in its source local buffer.")
            return DBuffer.from_local(
                self.local_buffer.narrow(0, local_offset, shard_ranges.numel),
                self.mesh,
                placements,
                self.layout,
                shard_ranges=shard_ranges,
            )
        if isinstance(source_placement, Partial) and isinstance(destination_placement, Replicate):
            return DBuffer.from_local(
                self.local_buffer,
                self.mesh,
                placements,
                self.layout,
                shard_ranges=self._shard_ranges,
            )
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
        reference: PlacementReference = Flat(),
    ) -> "DBuffer":
        """Distribute full local tensors into a DBuffer.

        Args:
            tensors: Full tensors available on this rank. Meta tensors contribute
                shape and dtype metadata but no values.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            reference: Shard placement the global layout is planned against.

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
        buffer = cls.empty(
            mesh=mesh,
            placements=placements,
            tensor_shapes=tensor_shapes,
            dtype=dtype,
            device=mesh.device_type,
            reference=reference,
        )
        # Only logical tensor ranges are initialized. Padding and layout gaps are not
        # observable through get_tensor_view() and can remain unspecified.
        for index, tensor in enumerate(tensors):
            owned_range = buffer._get_owned_range(index)
            if owned_range is None or tensor.is_meta:
                continue

            source_slice = tensor.view(-1).narrow(
                0, owned_range.tensor_relative_offset, owned_range.numel
            )
            buffer.local_buffer.narrow(
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
                layout=self.layout,
                dtype=dtype,
                device=self.device,
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

    def cast(self, dtype: torch.dtype, *, out: "DBuffer | None" = None) -> "DBuffer":
        """Return this buffer with the same layout and placements in ``dtype``."""
        if self.dtype == dtype and out is None:
            return self

        destination = self._create_or_validate_out(out, dtype=dtype)
        destination.local_buffer.copy_(self.local_buffer)
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

        changed_axis = changed_mesh_axis(self.placements, new_placements)
        if changed_axis is None:
            if out is None:
                return self
            out = self._create_or_validate_out(out, placements=new_placements)
            out.local_buffer.copy_(self.local_buffer)
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
            out.local_buffer.copy_(view.local_buffer)
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
            return DBuffer.from_local(
                self.local_buffer,
                self.mesh,
                new_placements,
                self.layout,
                shard_ranges=self._shard_ranges,
            )
        raise NotImplementedError(
            "Unsupported DBuffer placement transition on axis "
            f"{axis}: {old_placement!r} -> {new_placement!r}."
        )

    def allgather(self, mesh_axis: int, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-gather a sharded axis into Replicate placement."""
        if not isinstance(self.placements[mesh_axis], Shard):
            raise ValueError(
                f"allgather() currently requires a Shard placement on axis {mesh_axis!r}."
            )

        placements = list(self.placements)
        placements[mesh_axis] = Replicate()
        _validate_placements(placements)
        out = self._create_or_validate_out(out, placements=placements)
        group = self.mesh.get_group(mesh_axis)
        if self._shard_ranges.is_uniform(mesh_axis):
            # Symmetric-memory registration is scoped to the collective's process
            # group, so rendezvous the output on the same mesh axis as the all-gather.
            if out.is_symmetric_memory:
                out.rendezvous(mesh_axis)
            dist.all_gather_into_tensor(
                output_tensor=out.local_buffer, input_tensor=self.local_buffer, group=group
            )
        else:
            # Non-uniform shards (TensorAtomic): all_gather_into_tensor requires equal
            # input sizes, so gather into per-rank views of the output instead. Each
            # view covers exactly the global range that rank holds on this axis.
            table = self._shard_ranges.axis_ranges[mesh_axis]
            chunks = [
                out.local_buffer.narrow(0, offset - out.offset, numel) for offset, numel in table
            ]
            dist.all_gather(chunks, self.local_buffer, group=group)
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
        out.local_buffer.copy_(self.local_buffer)
        dist.all_reduce(
            out.local_buffer, op=_get_reduce_op(partial_placement), group=self.mesh.get_group(axis)
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
        group = self.mesh.get_group(axis)
        if out._shard_ranges.is_uniform(mesh_axis):
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
                output=out.local_buffer, input=self.local_buffer, op=reduce_op, group=group
            )
        else:
            # Non-uniform shards (TensorAtomic): reduce_scatter_tensor requires equal
            # output sizes, so split this rank's Partial input into per-destination-rank
            # views and use the list variant.
            table = out._shard_ranges.axis_ranges[axis]
            chunks = [
                self.local_buffer.narrow(0, offset - self.offset, numel) for offset, numel in table
            ]
            dist.reduce_scatter(out.local_buffer, chunks, op=reduce_op, group=group)

        if self.is_symmetric_memory and partial_placement.reduce_op == "avg":
            out.local_buffer.div_(self.mesh.size(axis))
        return out

    def get_tensor_view(self, index: int) -> torch.Tensor:
        """Return this rank's local view for logical tensor ``index``.

        Flat placements shard dim 0, so the returned view preserves all
        non-leading dimensions and only changes the leading dimension.
        """
        shape = self.layout.tensor_shapes[index]
        owned_range = self._get_owned_range(index)

        row_size = non_leading_numel(shape)
        if owned_range is None:
            empty_shape = torch.Size((0, *shape[1:]))
            return torch.empty(empty_shape, dtype=self.dtype, device=self.device)

        if owned_range.tensor_relative_offset % row_size != 0 or owned_range.numel % row_size != 0:
            raise RuntimeError(
                f"Local tensor shard for tensor {index} does not preserve dim-0 boundaries."
            )
        local_shape = torch.Size((owned_range.numel // row_size, *shape[1:]))
        return self.local_buffer.narrow(
            0, owned_range.buffer_relative_offset, owned_range.numel
        ).view(local_shape)

    def get_dtensor(self, index: int) -> DTensor:
        """Return logical tensor ``index`` as a DTensor."""
        local_tensor = self.get_tensor_view(index)
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
