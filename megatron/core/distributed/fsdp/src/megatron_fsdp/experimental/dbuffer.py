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
import torch.distributed.tensor as dist_tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from .layout import GlobalLayout, Shape, non_leading_numel
from .placement import Flat, Partial, Placement, Replicate


@dataclasses.dataclass(frozen=True)
class _OwnedRange:
    numel: int
    tensor_relative_offset: int
    buffer_relative_offset: int


def _validate_mesh_axis(mesh: DeviceMesh, axis: int) -> None:
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"Mesh axis must be an int, got {type(axis).__name__}.")
    if axis < 0 or axis >= mesh.ndim:
        raise ValueError(f"Mesh axis {axis} is out of bounds for mesh ndim {mesh.ndim}.")


def _validate_placements(placements: Iterable[Placement]) -> None:
    """Validate DBuffer placements form a supported contiguous local layout."""
    seen_flat = False
    for placement in placements:
        if not isinstance(placement, (Replicate, Partial, Flat)):
            raise TypeError(f"Unsupported DBuffer placement: {placement!r}.")
        if isinstance(placement, Flat):
            seen_flat = True
        elif seen_flat:
            raise ValueError(
                "Flat placements must be a suffix of the placement list so each "
                "local buffer is a contiguous global-buffer range."
            )


class DBuffer:
    """A distributed buffer holding a group of logical tensors.

    DBuffer is analogous to DTensor, but manages a group of logical tensors in
    one local storage tensor. It stores enough metadata to return per-tensor
    views, redistribute the buffer across mesh axes, and materialize per-tensor
    DTensors for optimizer state or distributed checkpointing.
    """

    # DBuffer owns only the data-parallel sub-mesh. Higher-level callers, such as
    # ParameterGroup, should extend returned DTensors with tensor-parallel mesh axes
    # because TP sharding metadata lives on nn.Parameter in MCore/TransformerEngine.
    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    layout: GlobalLayout
    offset: int
    local_buffer: torch.Tensor

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Create a DBuffer and allocate its local buffer.

        Args:
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            tensor_shapes: Global shapes for each logical tensor in this buffer.
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

        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        self.layout = GlobalLayout.build(tensor_shapes, dp_size=self.mesh.size())

        self.offset, local_numel = self.layout.get_local_range(self.mesh, self.placements)
        self.local_buffer = torch.empty(local_numel, dtype=dtype, device=device)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the local buffer."""
        return self.local_buffer.dtype

    @property
    def device(self) -> torch.device:
        """Device of the local buffer."""
        return self.local_buffer.device

    def reallocate_storage(self) -> None:
        """Restore the local buffer's backing storage to its logical size."""
        self._resize_storage(self.local_buffer.numel())

    def release_storage(self) -> None:
        """Release local buffer storage without replacing the Storage object."""
        # Autograd may save views that share this Storage object. Resizing the
        # existing Storage releases the allocation while preserving those aliases
        # for a later reallocate_storage().
        self._resize_storage(0)

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

    def copy_tensors_(self, tensors: Iterable[torch.Tensor]) -> None:
        """Copy full local tensor values into this DBuffer's local range."""
        tensors = tuple(
            (
                tensor.to(self.local_buffer.device)
                if tensor.device != self.local_buffer.device and not tensor.is_meta
                else tensor
            )
            .detach()
            .contiguous()
            for tensor in tensors
        )
        if len(tensors) != len(self.layout.tensor_shapes):
            raise ValueError(
                f"Expected {len(self.layout.tensor_shapes)} tensors, got {len(tensors)}."
            )
        for tensor, expected_shape in zip(tensors, self.layout.tensor_shapes, strict=True):
            if tensor.shape != expected_shape:
                raise ValueError(f"Expected tensor shape {expected_shape}, got {tensor.shape}.")
            if tensor.dtype != self.local_buffer.dtype or tensor.device != self.local_buffer.device:
                raise ValueError(
                    "All tensors copied into a DBuffer must match the buffer dtype and device."
                )

        # Only logical tensor ranges are initialized. Padding and layout gaps are not
        # observable through get_local_tensor() and can remain unspecified.
        for index, tensor in enumerate(tensors):
            owned_range = self._get_owned_range(index)
            if owned_range is None:
                continue

            source_slice = tensor.view(-1).narrow(
                0, owned_range.tensor_relative_offset, owned_range.numel
            )
            self.local_buffer.narrow(
                0, owned_range.buffer_relative_offset, owned_range.numel
            ).copy_(source_slice)

    @classmethod
    def from_local(
        cls,
        local_buffer: torch.Tensor,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
    ) -> "DBuffer":
        """Create a DBuffer from an existing local buffer.

        Args:
            local_buffer: Contiguous local tensor storage for this rank. DBuffer
                uses it directly in collectives such as all-gather and
                reduce-scatter, which are efficient with contiguous tensors.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            tensor_shapes: Global shapes for each logical tensor in this buffer.

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

        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        layout = GlobalLayout.build(tensor_shapes, dp_size=mesh.size())
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
        buffer.local_buffer = local_buffer
        return buffer

    @classmethod
    def distribute_tensors(
        cls, tensors: Iterable[torch.Tensor], mesh: DeviceMesh, placements: Iterable[Placement]
    ) -> "DBuffer":
        """Distribute full local tensor values into a DBuffer.

        Args:
            tensors: Full tensor values available on this rank.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.

        Returns:
            A DBuffer whose local storage matches ``placements``.
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
        )
        buffer.copy_tensors_(tensors)
        return buffer

    def _create_or_validate_out(
        self, placements: Iterable[Placement], out: "DBuffer | None"
    ) -> "DBuffer":
        placements = tuple(placements)
        if out is None:
            return DBuffer(
                mesh=self.mesh,
                placements=placements,
                tensor_shapes=self.layout.tensor_shapes,
                dtype=self.dtype,
                device=self.device,
            )

        if out.mesh != self.mesh:
            raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
        if out.placements != placements:
            raise ValueError(f"Expected out placements {placements!r}, got {out.placements!r}.")
        if out.layout != self.layout:
            raise ValueError(f"Expected out layout {self.layout!r}, got {out.layout!r}.")
        if out.dtype != self.dtype:
            raise ValueError(f"Expected out dtype {self.dtype}, got {out.dtype}.")
        if out.device != self.device:
            raise ValueError(f"Expected out device {self.device}, got {out.device}.")
        return out

    def cast(self, dtype: torch.dtype) -> "DBuffer":
        """Return this buffer with the same layout and placements in ``dtype``."""
        if self.dtype == dtype:
            return self

        destination = DBuffer(
            mesh=self.mesh,
            placements=self.placements,
            tensor_shapes=self.layout.tensor_shapes,
            dtype=dtype,
            device=self.device,
        )
        destination.local_buffer.copy_(self.local_buffer)
        return destination

    def redistribute(
        self, new_placements: Iterable[Placement], *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Redistribute this buffer to ``new_placements``.

        This dispatcher supports the one-axis transitions:
        Flat -> Replicate, Partial -> Replicate, Partial -> Flat, and
        Replicate -> Flat. Other placement changes are intentionally unsupported.
        """
        new_placements = tuple(new_placements)
        if len(new_placements) != self.mesh.ndim:
            raise ValueError(
                f"Expected {self.mesh.ndim} placements for device mesh, got "
                f"{len(new_placements)}."
            )
        _validate_placements(new_placements)

        changed_axis: int | None = None
        for axis, (old_placement, new_placement) in enumerate(
            zip(self.placements, new_placements, strict=True)
        ):
            if old_placement == new_placement:
                continue
            if changed_axis is not None:
                raise NotImplementedError(
                    "redistribute() currently supports one placement change, "
                    f"got changed axes {changed_axis} and {axis}."
                )
            changed_axis = axis

        if changed_axis is None:
            if out is None:
                return self
            out = self._create_or_validate_out(new_placements, out)
            out.local_buffer.copy_(self.local_buffer)
            return out

        axis = changed_axis
        old_placement = self.placements[axis]
        new_placement = new_placements[axis]
        if isinstance(old_placement, Flat) and isinstance(new_placement, Replicate):
            return self.allgather(axis, out=out)
        if isinstance(old_placement, Partial) and isinstance(new_placement, Replicate):
            return self.allreduce(axis, out=out)
        if isinstance(old_placement, Partial) and isinstance(new_placement, Flat):
            return self.reduce_scatter(axis, new_placement, out=out)
        if isinstance(old_placement, Replicate) and isinstance(new_placement, Flat):
            return self.scatter(axis, new_placement, out=out)
        raise NotImplementedError(
            "Unsupported DBuffer placement transition on axis "
            f"{axis}: {old_placement!r} -> {new_placement!r}."
        )

    def allgather(self, mesh_axis: int, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-gather a sharded axis into Replicate placement."""
        _validate_mesh_axis(self.mesh, mesh_axis)
        if not isinstance(self.placements[mesh_axis], Flat):
            raise ValueError(
                f"allgather() currently requires Flat placement on axis {mesh_axis!r}."
            )

        placements = list(self.placements)
        placements[mesh_axis] = Replicate()
        _validate_placements(placements)
        out = self._create_or_validate_out(placements, out)
        dist.all_gather_into_tensor(
            output_tensor=out.local_buffer,
            input_tensor=self.local_buffer,
            group=self.mesh.get_group(mesh_axis),
        )
        return out

    def allreduce(self, mesh_axis: int, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-reduce a Partial axis into Replicate placement."""
        _validate_mesh_axis(self.mesh, mesh_axis)
        axis = mesh_axis
        partial_placement = self.placements[axis]
        if not isinstance(partial_placement, Partial):
            raise ValueError(f"allreduce() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = Replicate()
        out = self._create_or_validate_out(placements, out)
        out.local_buffer.copy_(self.local_buffer)
        dist.all_reduce(
            out.local_buffer, op=partial_placement.reduce_op, group=self.mesh.get_group(axis)
        )
        return out

    def reduce_scatter(
        self, mesh_axis: int, new_placement: Placement, *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Reduce-scatter a Partial axis into ``new_placement``."""
        _validate_mesh_axis(self.mesh, mesh_axis)
        axis = mesh_axis
        if not isinstance(new_placement, Flat):
            raise NotImplementedError("DBuffer currently supports reduce_scatter() to Flat only.")
        partial_placement = self.placements[axis]
        if not isinstance(partial_placement, Partial):
            raise ValueError(f"reduce_scatter() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        _validate_placements(placements)
        out = self._create_or_validate_out(placements, out)
        dist.reduce_scatter_tensor(
            output=out.local_buffer,
            input=self.local_buffer,
            op=partial_placement.reduce_op,
            group=self.mesh.get_group(axis),
        )
        return out

    def scatter(
        self, mesh_axis: int, new_placement: Placement, *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Locally chunk a Replicate axis into ``new_placement``."""
        _validate_mesh_axis(self.mesh, mesh_axis)
        axis = mesh_axis
        if not isinstance(new_placement, Flat):
            raise NotImplementedError("DBuffer currently supports scatter() to Flat only.")
        if not isinstance(self.placements[axis], Replicate):
            raise ValueError(f"scatter() requires Replicate placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        _validate_placements(placements)

        if out is None:
            destination_offset, destination_numel = self.layout.get_local_range(
                self.mesh, placements
            )
        else:
            out = self._create_or_validate_out(placements, out)
            destination_offset = out.offset
            destination_numel = out.local_buffer.numel()

        local_buffer_offset = destination_offset - self.offset
        if (
            local_buffer_offset < 0
            or local_buffer_offset + destination_numel > self.local_buffer.numel()
        ):
            raise RuntimeError("scatter() destination is not contained in the source local buffer.")
        local_slice = self.local_buffer.narrow(0, local_buffer_offset, destination_numel)
        if out is None:
            return DBuffer.from_local(local_slice, self.mesh, placements, self.layout.tensor_shapes)

        out.local_buffer.copy_(local_slice)
        return out

    def get_local_tensor(self, index: int) -> torch.Tensor:
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
        torch_placements = []
        for placement in self.placements:
            if isinstance(placement, Replicate):
                torch_placements.append(dist_tensor.Replicate())
            elif isinstance(placement, Flat):
                torch_placements.append(dist_tensor.Shard(0))
            elif isinstance(placement, Partial):
                raise ValueError("Partial DBuffer placements cannot be represented as DTensor.")
            else:
                raise TypeError(f"Unsupported placement for DTensor conversion: {placement!r}.")

        local_tensor = self.get_local_tensor(index)
        tensor_shape = self.layout.tensor_shapes[index]
        # DBuffer uses contiguous flat storage, and Flat only shards dim 0, so
        # the local view's stride matches the logical global tensor stride.
        return DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=self.mesh,
            placements=tuple(torch_placements),
            run_check=False,
            shape=tensor_shape,
            stride=local_tensor.stride(),
        )
