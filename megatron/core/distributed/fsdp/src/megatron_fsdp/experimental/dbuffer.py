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
import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.distributed.tensor as dist_tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from .placement import Flat, Partial, Placement, Replicate, validate_placements

MeshAxis = int | str
Shape = torch.Size | Iterable[int]


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global tensor layout in element coordinates."""

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int

    def get_local_range(self, mesh: DeviceMesh, placements: Iterable[Placement]) -> tuple[int, int]:
        """Return this rank's local element offset and length for ``placements``."""
        offset = 0
        numel = self.size
        for axis, placement in reversed(tuple(enumerate(placements))):
            if not isinstance(placement, Flat):
                continue
            axis_size = mesh.size(axis)
            if numel % axis_size != 0:
                raise ValueError(
                    f"Local range size {numel} is not divisible by Flat axis size {axis_size}."
                )
            shard_size = numel // axis_size
            offset += mesh.get_local_rank(axis) * shard_size
            numel = shard_size
        return offset, numel


def _non_leading_numel(shape: torch.Size) -> int:
    return shape[1:].numel() if len(shape) > 1 else 1


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


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


def _compute_layout(shapes: Iterable[Shape], dp_size: int) -> GlobalLayout:
    """Compute global tensor element offsets and padded size.

    This is a DBuffer-specific reimplementation of
    ``param_and_grad_buffer.build_data_parallel_buffer_index``. It keeps only
    the global offset construction and final DP-LCM padding; DBuffer derives
    rank-local slices later through DTensor placements.

    The computed layout is compatible with Flat, TensorAtomic, and BlockAtomic,
    even though the latter two are not implemented.

    Args:
        shapes: Logical tensor shapes in tensor-id order.
        dp_size: Data-parallel shard count for this global layout.

    Returns:
        Global layout with element offsets and size padded to a multiple of
        ``LCM(shape[1:].numel()) * dp_size``.
    """
    if dp_size <= 0:
        raise ValueError(f"DP size must be positive, got {dp_size}.")

    shapes = tuple(torch.Size(shape) for shape in shapes)
    chunk_size = 1
    for shape in shapes:
        non_leading_numel = _non_leading_numel(shape)
        if non_leading_numel <= 0:
            raise ValueError(f"Cannot compute a layout for zero-sized non-leading dims: {shape}.")
        chunk_size = math.lcm(chunk_size, non_leading_numel)

    # The LCM part is the packing grid. Since every row size divides this grid,
    # DP shard boundaries that are multiples of the grid avoid splitting dim-0 rows.
    tensor_to_offset: list[int | None] = [None] * len(shapes)
    fragment_items = []
    regular_items = []
    for tensor_id, shape in enumerate(shapes):
        if shape.numel() < chunk_size:
            fragment_items.append((tensor_id, shape))
        else:
            regular_items.append((tensor_id, shape))

    # Regular tensors anchor the layout. Fragments are held back to fill padding
    # gaps left by regular tensors whose sizes are not exact multiples of the grid.
    fragment_items.sort(key=lambda id_shape: id_shape[1].numel(), reverse=True)

    next_offset = 0
    while regular_items:
        tensor_id, shape = regular_items.pop(0)
        tensor_numel = shape.numel()
        tensor_to_offset[tensor_id] = next_offset

        if tensor_numel % chunk_size == 0:
            next_offset += tensor_numel
            continue

        gap_offset = next_offset + tensor_numel
        next_offset += _pad_to_multiple(tensor_numel, chunk_size)
        fragment_gap_end = next_offset
        remainder = tensor_numel % chunk_size

        # Try to pair this non-divisible regular tensor with a conjugate regular
        # tensor whose remainder fits in the same LCM part. The conjugate starts
        # in the gap and then continues with full LCM parts after this one.
        conjugate_item = None
        for candidate_item in regular_items[:]:
            _, candidate_shape = candidate_item
            candidate_numel = candidate_shape.numel()
            candidate_remainder = candidate_numel % chunk_size
            if candidate_remainder == 0:
                continue
            if remainder + candidate_remainder <= chunk_size:
                conjugate_item = candidate_item
                regular_items.remove(candidate_item)
                break

        if conjugate_item is not None:
            conjugate_id, conjugate_shape = conjugate_item
            conjugate_numel = conjugate_shape.numel()
            conjugate_remainder = conjugate_numel % chunk_size
            conjugate_offset = next_offset - conjugate_remainder
            tensor_to_offset[conjugate_id] = conjugate_offset
            fragment_gap_end = conjugate_offset
            next_offset += (conjugate_numel // chunk_size) * chunk_size

        # Fill any remaining gap with fragments, keeping each fragment aligned to
        # its own row size so dim-0 rows remain contiguous within DP shards.
        for fragment in fragment_items[:]:
            frag_id, frag_shape = fragment
            frag_numel = frag_shape.numel()
            aligned_gap_offset = _pad_to_multiple(gap_offset, _non_leading_numel(frag_shape))
            if aligned_gap_offset + frag_numel > fragment_gap_end:
                continue
            tensor_to_offset[frag_id] = aligned_gap_offset
            gap_offset = aligned_gap_offset + frag_numel
            fragment_items.remove(fragment)

    # Fragments that did not fit into regular-tensor gaps are appended at the tail.
    for frag_id, frag_shape in fragment_items:
        next_offset = _pad_to_multiple(next_offset, _non_leading_numel(frag_shape))
        tensor_to_offset[frag_id] = next_offset
        next_offset += frag_shape.numel()

    if any(offset is None for offset in tensor_to_offset):
        raise AssertionError(f"Incomplete DBuffer layout for shapes {shapes}.")

    resolved_tensor_to_offset = tuple(offset for offset in tensor_to_offset if offset is not None)
    for shape, offset in zip(shapes, resolved_tensor_to_offset, strict=True):
        row_size = _non_leading_numel(shape)
        if offset % row_size != 0:
            raise AssertionError(f"Tensor offset {offset} is not aligned to row size {row_size}.")
    size = _pad_to_multiple(next_offset, chunk_size * dp_size)
    return GlobalLayout(tensor_shapes=shapes, tensor_to_offset=resolved_tensor_to_offset, size=size)


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
        validate_placements(placements)

        self.mesh = mesh
        self.placements = placements

        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        self.layout = _compute_layout(tensor_shapes, dp_size=self.mesh.size())

        self.offset, local_numel = self.layout.get_local_range(self.mesh, self.placements)
        self.local_buffer = torch.empty(local_numel, dtype=dtype, device=device)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the local buffer."""
        return self.local_buffer.dtype

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
            local_buffer: Local tensor storage for this rank.
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
        validate_placements(placements)
        if local_buffer.dim() != 1:
            raise ValueError("local_buffer must be a flat 1D tensor.")

        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        layout = _compute_layout(tensor_shapes, dp_size=mesh.size())
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
        tensors = tuple(
            (
                tensor.to(mesh.device_type)
                if tensor.device.type != mesh.device_type and not tensor.is_meta
                else tensor
            )
            .detach()
            .contiguous()
            for tensor in tensors
        )
        if not tensors:
            raise ValueError("DBuffer.distribute_tensors() requires at least one tensor.")

        dtype = tensors[0].dtype
        device = tensors[0].device
        for tensor in tensors:
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError("All tensors in a DBuffer must have the same dtype and device.")

        tensor_shapes = tuple(tensor.shape for tensor in tensors)
        buffer = cls(
            mesh=mesh,
            placements=placements,
            tensor_shapes=tensor_shapes,
            dtype=dtype,
            device=device,
        )
        local_start = buffer.offset
        local_end = local_start + buffer.local_buffer.numel()
        # Only logical tensor ranges are initialized. Padding and layout gaps are not
        # observable through get_tensor() and can remain unspecified.
        for tensor, tensor_start in zip(tensors, buffer.layout.tensor_to_offset, strict=True):
            tensor_end = tensor_start + tensor.numel()
            overlap_start = max(local_start, tensor_start)
            overlap_end = min(local_end, tensor_end)
            if overlap_start >= overlap_end:
                continue

            overlap_numel = overlap_end - overlap_start
            source_offset = overlap_start - tensor_start
            destination_offset = overlap_start - local_start
            buffer.local_buffer.narrow(0, destination_offset, overlap_numel).copy_(
                tensor.view(-1).narrow(0, source_offset, overlap_numel)
            )
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
                device=self.local_buffer.device,
            )

        if out.mesh != self.mesh:
            raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
        if out.placements != placements:
            raise ValueError(f"Expected out placements {placements!r}, got {out.placements!r}.")
        if out.layout != self.layout:
            raise ValueError(f"Expected out layout {self.layout!r}, got {out.layout!r}.")
        if out.dtype != self.dtype:
            raise ValueError(f"Expected out dtype {self.dtype}, got {out.dtype}.")
        if out.local_buffer.device != self.local_buffer.device:
            raise ValueError(
                f"Expected out device {self.local_buffer.device}, got {out.local_buffer.device}."
            )
        return out

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
        validate_placements(new_placements)

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

    def allgather(self, mesh_axis: MeshAxis, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-gather a Flat axis into Replicate placement."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(self.placements[axis], Flat):
            raise ValueError(f"allgather() requires Flat placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = Replicate()
        validate_placements(placements)
        out = self._create_or_validate_out(placements, out)
        dist.all_gather_into_tensor(
            output_tensor=out.local_buffer,
            input_tensor=self.local_buffer,
            group=self.mesh.get_group(axis),
        )
        return out

    def allreduce(self, mesh_axis: MeshAxis, *, out: "DBuffer | None" = None) -> "DBuffer":
        """All-reduce a Partial axis into Replicate placement."""
        axis = _axis_index(self.mesh, mesh_axis)
        partial_placement = self.placements[axis]
        if not isinstance(partial_placement, Partial):
            raise ValueError(f"allreduce() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = Replicate()
        out = self._create_or_validate_out(placements, out)
        out.local_buffer.copy_(self.local_buffer)
        dist.all_reduce(
            out.local_buffer,
            op=partial_placement.reduce_op,
            group=self.mesh.get_group(axis),
        )
        return out

    def reduce_scatter(
        self, mesh_axis: MeshAxis, new_placement: Placement, *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Reduce-scatter a Partial axis into ``new_placement``."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(new_placement, Flat):
            raise NotImplementedError("DBuffer currently supports reduce_scatter() to Flat only.")
        partial_placement = self.placements[axis]
        if not isinstance(partial_placement, Partial):
            raise ValueError(f"reduce_scatter() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        validate_placements(placements)
        out = self._create_or_validate_out(placements, out)
        dist.reduce_scatter_tensor(
            output=out.local_buffer,
            input=self.local_buffer,
            op=partial_placement.reduce_op,
            group=self.mesh.get_group(axis),
        )
        return out

    def scatter(
        self, mesh_axis: MeshAxis, new_placement: Placement, *, out: "DBuffer | None" = None
    ) -> "DBuffer":
        """Locally chunk a Replicate axis into ``new_placement``."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(new_placement, Flat):
            raise NotImplementedError("DBuffer currently supports scatter() to Flat only.")
        if not isinstance(self.placements[axis], Replicate):
            raise ValueError(f"scatter() requires Replicate placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        validate_placements(placements)

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

    def get_tensor(self, index: int) -> torch.Tensor:
        """Return this rank's local view for logical tensor ``index``.

        Flat placements shard dim 0, so the returned view preserves all
        non-leading dimensions and only changes the leading dimension.
        """
        shape = self.layout.tensor_shapes[index]
        offset = self.layout.tensor_to_offset[index]
        numel = shape.numel()

        tensor_start = offset
        tensor_end = offset + numel
        overlap_start = max(tensor_start, self.offset)
        overlap_end = min(tensor_end, self.offset + self.local_buffer.numel())

        row_size = _non_leading_numel(shape)
        if overlap_end <= overlap_start:
            empty_shape = torch.Size((0, *shape[1:]))
            return torch.empty(empty_shape, dtype=self.dtype, device=self.local_buffer.device)

        local_numel = overlap_end - overlap_start
        local_buffer_offset = overlap_start - self.offset
        if (overlap_start - tensor_start) % row_size != 0 or local_numel % row_size != 0:
            raise RuntimeError(
                f"Local tensor shard for tensor {index} does not preserve dim-0 boundaries."
            )
        local_shape = torch.Size((local_numel // row_size, *shape[1:]))
        return self.local_buffer.narrow(0, local_buffer_offset, local_numel).view(local_shape)

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

        local_tensor = self.get_tensor(index)
        tensor_shape = self.layout.tensor_shapes[index]
        return DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=self.mesh,
            placements=tuple(torch_placements),
            run_check=False,
            shape=tensor_shape,
            stride=local_tensor.stride(),
        )
