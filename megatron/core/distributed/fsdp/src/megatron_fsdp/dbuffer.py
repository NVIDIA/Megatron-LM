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

"""Distributed flat buffers for Megatron-FSDP."""

import dataclasses
import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.distributed.tensor as dist_tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

MeshAxis = int | str


class Placement:
    """Base class for DBuffer placements."""


@dataclasses.dataclass(frozen=True)
class Replicate(Placement):
    """Replicated local buffer placement."""


@dataclasses.dataclass(frozen=True)
class Partial(Placement):
    """Unreduced replicated local buffer placement."""


@dataclasses.dataclass(frozen=True)
class Flat(Placement):
    """Flat per-unit dim-0 sharded local buffer placement."""


@dataclasses.dataclass(frozen=True)
class GlobalLayout:
    """Global flat-buffer layout in element coordinates."""

    tensor_shapes: tuple[torch.Size, ...]
    tensor_to_offset: tuple[int, ...]
    size: int


def _numel(shape: torch.Size) -> int:
    return shape.numel()


def _non_leading_numel(shape: torch.Size) -> int:
    return _numel(shape[1:]) if len(shape) > 1 else 1


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
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


def _validate_placement(placement: Placement) -> None:
    if not isinstance(placement, (Replicate, Partial, Flat)):
        raise TypeError(f"Unsupported DBuffer placement: {placement!r}.")


def _validate_placements(placements: Sequence[Placement]) -> None:
    seen_flat = False
    for placement in placements:
        _validate_placement(placement)
        if isinstance(placement, Flat):
            seen_flat = True
        elif seen_flat:
            raise ValueError(
                "Flat placements must be a suffix of the placement list so each "
                "local buffer is a contiguous global-buffer range."
            )


def compute_layout(shapes: Sequence[torch.Size], dp_size: int) -> GlobalLayout:
    """Compute flat-buffer element offsets and globally padded size.

    Args:
        shapes: Logical tensor shapes in tensor-id order.
        dp_size: Data-parallel shard count for this flat buffer layout.

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

    tensor_to_offset: list[int | None] = [None] * len(shapes)
    fragment_items = []
    regular_items = []
    for tensor_id, shape in enumerate(shapes):
        if _numel(shape) < chunk_size:
            fragment_items.append((tensor_id, shape))
        else:
            regular_items.append((tensor_id, shape))

    fragment_items.sort(key=lambda id_shape: -_numel(id_shape[1]))

    data_index = 0
    while regular_items:
        tensor_id, shape = regular_items.pop(0)
        tensor_numel = _numel(shape)
        tensor_to_offset[tensor_id] = data_index

        if tensor_numel % chunk_size == 0:
            data_index += tensor_numel
            continue

        gap_offset = data_index + tensor_numel
        data_index += _pad_to_multiple(tensor_numel, chunk_size)
        fragment_gap_end = data_index
        remain = tensor_numel % chunk_size

        found_rhs = None
        for rhs in regular_items[:]:
            _, rhs_shape = rhs
            rhs_numel = _numel(rhs_shape)
            rhs_remain = rhs_numel % chunk_size
            if rhs_remain == 0:
                continue
            if remain + rhs_remain <= chunk_size:
                found_rhs = rhs
                regular_items.remove(rhs)
                break

        if found_rhs is not None:
            rhs_id, rhs_shape = found_rhs
            rhs_numel = _numel(rhs_shape)
            rhs_remain = rhs_numel % chunk_size
            rhs_offset = data_index - rhs_remain
            tensor_to_offset[rhs_id] = rhs_offset
            fragment_gap_end = rhs_offset
            data_index += (rhs_numel // chunk_size) * chunk_size

        for fragment in fragment_items[:]:
            frag_id, frag_shape = fragment
            frag_numel = _numel(frag_shape)
            aligned_gap_offset = _pad_to_multiple(gap_offset, _non_leading_numel(frag_shape))
            if aligned_gap_offset + frag_numel > fragment_gap_end:
                continue
            tensor_to_offset[frag_id] = aligned_gap_offset
            gap_offset = aligned_gap_offset + frag_numel
            fragment_items.remove(fragment)

    for frag_id, frag_shape in fragment_items:
        data_index = _pad_to_multiple(data_index, _non_leading_numel(frag_shape))
        tensor_to_offset[frag_id] = data_index
        data_index += _numel(frag_shape)

    if any(offset is None for offset in tensor_to_offset):
        raise AssertionError(f"Incomplete DBuffer layout for shapes {shapes}.")

    resolved_tensor_to_offset = tuple(offset for offset in tensor_to_offset if offset is not None)
    for shape, offset in zip(shapes, resolved_tensor_to_offset, strict=True):
        row_size = _non_leading_numel(shape)
        if offset % row_size != 0:
            raise AssertionError(f"Tensor offset {offset} is not aligned to row size {row_size}.")
    size = _pad_to_multiple(data_index, chunk_size * dp_size)
    return GlobalLayout(tensor_shapes=shapes, tensor_to_offset=resolved_tensor_to_offset, size=size)


def _local_shard_range_for_placements(
    mesh: DeviceMesh, layout: GlobalLayout, placements: Sequence[Placement]
) -> tuple[int, int]:
    offset = 0
    numel = layout.size
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
    return offset, offset + numel


class DBuffer:
    """A flat distributed buffer holding a group of logical tensors.

    DBuffer stores one flat local tensor and enough metadata to return per-tensor
    views, redistribute the buffer across mesh axes, and materialize per-tensor
    DTensors for optimizer state or distributed checkpointing.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    layout: GlobalLayout
    offset: int
    local_buffer: torch.Tensor

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Sequence[Placement],
        tensor_shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Create a DBuffer and allocate its local flat buffer.

        Args:
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            tensor_shapes: Global shapes for each logical tensor in this buffer.
            dtype: Dtype for the local flat buffer.
            device: Device for the local flat buffer.
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
        self.layout = compute_layout(tensor_shapes, dp_size=int(self.mesh.mesh.numel()))

        self.offset, end = _local_shard_range_for_placements(
            self.mesh, self.layout, self.placements
        )
        local_numel = end - self.offset
        self.local_buffer = torch.empty(local_numel, dtype=dtype, device=device)

    @classmethod
    def distribute_tensors(
        cls, tensors: Sequence[torch.Tensor], mesh: DeviceMesh, placements: Sequence[Placement]
    ) -> "DBuffer":
        """Distribute full local tensor values into a DBuffer.

        Args:
            tensors: Full tensor values available on this rank.
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.

        Returns:
            A DBuffer whose local storage matches ``placements``.
        """
        if not tensors:
            raise ValueError("DBuffer.distribute_tensors() requires at least one tensor.")

        tensors = tuple(
            (
                tensor.to(mesh.device_type)
                if tensor.device.type != mesh.device_type and not tensor.is_meta
                else tensor
            )
            for tensor in tensors
        )
        dtype = tensors[0].dtype
        device = tensors[0].device
        for tensor in tensors:
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError("All tensors in a DBuffer must have the same dtype and device.")
            if not tensor.is_contiguous():
                raise ValueError("DBuffer.distribute_tensors() expects contiguous tensors.")

        tensor_shapes = tuple(tensor.shape for tensor in tensors)
        buffer = cls(
            mesh=mesh,
            placements=placements,
            tensor_shapes=tensor_shapes,
            dtype=dtype,
            device=device,
        )
        full_buffer = torch.zeros(buffer.layout.size, dtype=dtype, device=device)
        for tensor, offset in zip(tensors, buffer.layout.tensor_to_offset, strict=True):
            full_buffer.narrow(0, offset, tensor.numel()).copy_(tensor.view(-1))

        local_buffer = full_buffer.narrow(0, buffer.offset, buffer.local_buffer.numel())
        buffer.local_buffer.copy_(local_buffer)
        return buffer

    def redistribute(self, new_placements: Sequence[Placement]) -> "DBuffer":
        """Redistribute this buffer to ``new_placements``.

        This dispatcher composes the supported one-axis transitions:
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

        buffer = self
        for axis, new_placement in enumerate(new_placements):
            old_placement = buffer.placements[axis]
            if old_placement == new_placement:
                continue
            if isinstance(old_placement, Flat) and isinstance(new_placement, Replicate):
                buffer = buffer.allgather(axis)
            elif isinstance(old_placement, Partial) and isinstance(new_placement, Replicate):
                buffer = buffer.allreduce(axis)
            elif isinstance(old_placement, Partial) and isinstance(new_placement, Flat):
                buffer = buffer.reduce_scatter(axis, new_placement)
            elif isinstance(old_placement, Replicate) and isinstance(new_placement, Flat):
                buffer = buffer.scatter(axis, new_placement)
            else:
                raise NotImplementedError(
                    "Unsupported DBuffer placement transition on axis "
                    f"{axis}: {old_placement!r} -> {new_placement!r}."
                )

        if buffer.placements != new_placements:
            raise AssertionError(
                f"Redistribute produced placements {buffer.placements}, expected {new_placements}."
            )
        return buffer

    def allgather(self, mesh_axis: MeshAxis) -> "DBuffer":
        """All-gather a Flat axis into Replicate placement."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(self.placements[axis], Flat):
            raise ValueError(f"allgather() requires Flat placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = Replicate()
        _validate_placements(placements)
        buffer = DBuffer(
            mesh=self.mesh,
            placements=placements,
            tensor_shapes=self.layout.tensor_shapes,
            dtype=self.local_buffer.dtype,
            device=self.local_buffer.device,
        )
        dist.all_gather_into_tensor(
            output_tensor=buffer.local_buffer,
            input_tensor=self.local_buffer,
            group=self.mesh.get_group(axis),
        )
        return buffer

    def allreduce(self, mesh_axis: MeshAxis) -> "DBuffer":
        """All-reduce a Partial axis into Replicate placement."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(self.placements[axis], Partial):
            raise ValueError(f"allreduce() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = Replicate()
        buffer = DBuffer(
            mesh=self.mesh,
            placements=placements,
            tensor_shapes=self.layout.tensor_shapes,
            dtype=self.local_buffer.dtype,
            device=self.local_buffer.device,
        )
        buffer.local_buffer.copy_(self.local_buffer)
        dist.all_reduce(buffer.local_buffer, op=dist.ReduceOp.SUM, group=self.mesh.get_group(axis))
        return buffer

    def reduce_scatter(self, mesh_axis: MeshAxis, new_placement: Placement) -> "DBuffer":
        """Reduce-scatter a Partial axis into ``new_placement``."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(new_placement, Flat):
            raise NotImplementedError("DBuffer currently supports reduce_scatter() to Flat only.")
        if not isinstance(self.placements[axis], Partial):
            raise ValueError(f"reduce_scatter() requires Partial placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        _validate_placements(placements)
        buffer = DBuffer(
            mesh=self.mesh,
            placements=placements,
            tensor_shapes=self.layout.tensor_shapes,
            dtype=self.local_buffer.dtype,
            device=self.local_buffer.device,
        )
        dist.reduce_scatter_tensor(
            output=buffer.local_buffer,
            input=self.local_buffer,
            op=dist.ReduceOp.SUM,
            group=self.mesh.get_group(axis),
        )
        return buffer

    def scatter(self, mesh_axis: MeshAxis, new_placement: Placement) -> "DBuffer":
        """Locally chunk a Replicate axis into ``new_placement``."""
        axis = _axis_index(self.mesh, mesh_axis)
        if not isinstance(new_placement, Flat):
            raise NotImplementedError("DBuffer currently supports scatter() to Flat only.")
        if not isinstance(self.placements[axis], Replicate):
            raise ValueError(f"scatter() requires Replicate placement on axis {mesh_axis!r}.")

        placements = list(self.placements)
        placements[axis] = new_placement
        _validate_placements(placements)
        buffer = DBuffer(
            mesh=self.mesh,
            placements=placements,
            tensor_shapes=self.layout.tensor_shapes,
            dtype=self.local_buffer.dtype,
            device=self.local_buffer.device,
        )
        local_buffer_offset = buffer.offset - self.offset
        if (
            local_buffer_offset < 0
            or local_buffer_offset + buffer.local_buffer.numel() > self.local_buffer.numel()
        ):
            raise RuntimeError("scatter() destination is not contained in the source local buffer.")
        buffer.local_buffer.copy_(
            self.local_buffer.narrow(0, local_buffer_offset, buffer.local_buffer.numel())
        )
        return buffer

    def get_tensor(self, index: int) -> torch.Tensor:
        """Return this rank's local view for logical tensor ``index``.

        Flat placements shard dim 0, so the returned view preserves all
        non-leading dimensions and only changes the leading dimension.
        """
        shape = self.layout.tensor_shapes[index]
        offset = self.layout.tensor_to_offset[index]
        numel = _numel(shape)

        tensor_start = offset
        tensor_end = offset + numel
        overlap_start = max(tensor_start, self.offset)
        overlap_end = min(tensor_end, self.offset + self.local_buffer.numel())

        row_size = _non_leading_numel(shape)
        if overlap_end <= overlap_start:
            empty_shape = torch.Size((0, *shape[1:]))
            return torch.empty(
                empty_shape, dtype=self.local_buffer.dtype, device=self.local_buffer.device
            )

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
