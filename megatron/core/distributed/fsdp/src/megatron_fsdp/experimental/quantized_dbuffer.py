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

"""Transformer Engine MXFP8 distributed buffers composed from physical planes."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
except ImportError as exc:
    raise ImportError(
        "UnavailableError: QuantizedDBuffer requires Transformer Engine MXFP8 support"
    ) from exc

from .dbuffer import DBuffer
from .layout import GlobalLayout, Shape
from .placement import BlockAtomic, RowAtomic

_MXFP8_DTYPE = tex.DType.kFloat8E4M3
# TODO: Support quantizing only rowwise or columnwise data, allocating and
# communicating only the required planes to save memory and communication.
_MXFP8_QUANTIZER = MXFP8Quantizer(_MXFP8_DTYPE)
_MXFP8_BLOCK_SIZE = 32


def effective_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return MFSDP's storage dtype for a parameter."""
    return torch.uint8 if isinstance(tensor, MXFP8Tensor) else tensor.dtype


def _rowwise_scale_layout(data_layout: GlobalLayout) -> GlobalLayout:
    """Derive rowwise scales from the layout shared by rowwise_data and columnwise_data."""
    return GlobalLayout(
        tensor_shapes=tuple(
            torch.Size((shape[0], shape[1] // _MXFP8_BLOCK_SIZE))
            for shape in data_layout.tensor_shapes
        ),
        tensor_to_offset=tuple(
            offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset
        ),
        size=data_layout.size // _MXFP8_BLOCK_SIZE,
        rank_segment_offsets=tuple(
            offset // _MXFP8_BLOCK_SIZE for offset in data_layout.rank_segment_offsets
        ),
        reference=data_layout.reference,
    )


def _columnwise_scale_layout(data_layout: GlobalLayout) -> GlobalLayout:
    """Derive columnwise scales from the layout shared by rowwise_data and columnwise_data."""
    return GlobalLayout(
        tensor_shapes=tuple(
            torch.Size((shape[0] // _MXFP8_BLOCK_SIZE, shape[1]))
            for shape in data_layout.tensor_shapes
        ),
        tensor_to_offset=tuple(
            offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset
        ),
        size=data_layout.size // _MXFP8_BLOCK_SIZE,
        rank_segment_offsets=tuple(
            offset // _MXFP8_BLOCK_SIZE for offset in data_layout.rank_segment_offsets
        ),
    )


def _block_atomic_to_flat(placements: Iterable[Placement]) -> tuple[Placement, ...]:
    """Replace BlockAtomic placements with RowAtomic for coordinates measured in blocks.

    For example, one MXFP8 columnwise scale row represents a 32-row weight
    block, so RowAtomic preserves the shard boundaries of BlockAtomic(32).
    """
    placements = tuple(placements)
    return tuple(
        RowAtomic() if isinstance(placement, BlockAtomic) else placement for placement in placements
    )


def _pad_rowwise_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pad rowwise scales to TE's physical allocation shape."""
    shape = _MXFP8_QUANTIZER.get_scale_shape(
        (scale.shape[0], scale.shape[1] * _MXFP8_BLOCK_SIZE), columnwise=False
    )
    if scale.shape == shape:
        return scale
    return F.pad(scale, (0, shape[1] - scale.shape[1], 0, shape[0] - scale.shape[0]))


def _pad_columnwise_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pad columnwise scales to TE's physical allocation shape."""
    shape = _MXFP8_QUANTIZER.get_scale_shape(
        (scale.shape[0] * _MXFP8_BLOCK_SIZE, scale.shape[1]), columnwise=True
    )
    if scale.shape == shape:
        return scale
    return F.pad(scale, (0, shape[1] - scale.shape[1], 0, shape[0] - scale.shape[0]))


class QuantizedDBuffer:
    """The MFSDP storage and lifecycle for one TE MXFP8 tensor.

    The data layout owns the parameter-to-rank partition. Scale planes use its
    compact MXFP8 coordinates, so every plane on a rank describes the same
    local data rows. TE-only padding is added when materializing a wrapper for
    compute, not to distributed storage.
    """

    rowwise_data: DBuffer
    columnwise_data: DBuffer
    rowwise_scale: DBuffer
    columnwise_scale: DBuffer

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        layout: GlobalLayout,
        device: torch.device | str,
    ) -> None:
        """Allocate MXFP8 data and scale planes from a shared weight layout."""
        if layout.block_size != _MXFP8_BLOCK_SIZE:
            raise ValueError(f"QuantizedDBuffer requires block size {_MXFP8_BLOCK_SIZE}.")
        tensor_shapes = layout.tensor_shapes
        if not tensor_shapes or any(len(shape) != 2 for shape in tensor_shapes):
            raise ValueError("QuantizedDBuffer requires one or more 2D MXFP8 tensor shapes.")
        if any(
            shape[0] % _MXFP8_BLOCK_SIZE or shape[1] % _MXFP8_BLOCK_SIZE for shape in tensor_shapes
        ):
            raise ValueError(
                f"QuantizedDBuffer requires dimensions divisible by {_MXFP8_BLOCK_SIZE}."
            )
        placements = tuple(placements)
        self.rowwise_data = DBuffer(mesh, placements, layout, torch.uint8, device)
        self.columnwise_data = DBuffer(mesh, placements, layout, torch.uint8, device)
        self.rowwise_scale = DBuffer(
            mesh, placements, _rowwise_scale_layout(layout), torch.uint8, device
        )
        self.columnwise_scale = DBuffer(
            mesh,
            _block_atomic_to_flat(placements),
            _columnwise_scale_layout(layout),
            torch.uint8,
            device,
        )

    @classmethod
    def empty(
        cls,
        mesh: DeviceMesh,
        placements: Iterable[Placement],
        tensor_shapes: Iterable[Shape],
        device: torch.device | str,
    ) -> "QuantizedDBuffer":
        """Build an MXFP8 layout from logical tensor shapes and allocate its planes."""
        layout = GlobalLayout.build(
            tensor_shapes, dp_size=mesh.size(), reference=BlockAtomic(_MXFP8_BLOCK_SIZE)
        )
        return cls(mesh, placements, layout, device)

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh shared by all physical planes."""
        return self.rowwise_data.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """Logical weight placements; scale planes map these to scale coordinates."""
        return self.rowwise_data.placements

    @classmethod
    def _from_planes(
        cls,
        rowwise_data: DBuffer,
        columnwise_data: DBuffer,
        rowwise_scale: DBuffer,
        columnwise_scale: DBuffer,
    ) -> "QuantizedDBuffer":
        """Create a composed view from already-allocated physical planes."""
        result = cls.__new__(cls)
        result.rowwise_data = rowwise_data
        result.columnwise_data = columnwise_data
        result.rowwise_scale = rowwise_scale
        result.columnwise_scale = columnwise_scale
        return result

    def get_tensor_view(self, index: int) -> MXFP8Tensor:
        """Return a compact, unswizzled MXFP8 view that aliases all local planes."""
        rowwise_data = self.rowwise_data.get_tensor_view(index)
        rowwise_scale = self.rowwise_scale.get_tensor_view(index)
        columnwise_scale = self.columnwise_scale.get_tensor_view(index)
        return MXFP8Tensor(
            shape=rowwise_data.shape,
            dtype=torch.bfloat16,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=self.columnwise_data.get_tensor_view(index),
            columnwise_scale_inv=columnwise_scale,
            fp8_dtype=_MXFP8_DTYPE,
            quantizer=_MXFP8_QUANTIZER,
            with_gemm_swizzled_scales=False,
            device=rowwise_data.device,
        )

    def get_tensor(self, index: int) -> MXFP8Tensor:
        """Return an unswizzled compute tensor with scales padded for TE's GEMM path.

        Data planes remain views. Scale planes alias storage only when no padding
        is needed; otherwise they are copied into padded allocations.
        """
        tensor = self.get_tensor_view(index)
        # GEMM requires padded scales. Pad here until TE fuses padding into its
        # scale-swizzle kernel, avoiding these separate allocations and copies:
        # https://github.com/NVIDIA/TransformerEngine/issues/3518
        tensor._rowwise_scale_inv = _pad_rowwise_scale(tensor._rowwise_scale_inv)
        tensor._columnwise_scale_inv = _pad_columnwise_scale(tensor._columnwise_scale_inv)
        return tensor

    def quantize_(self, main_weight: DBuffer) -> None:
        """Quantize a local master shard with matching mesh, placements, layout, and device."""
        for attribute in ("mesh", "placements", "layout", "device"):
            expected = getattr(self.rowwise_data, attribute)
            actual = getattr(main_weight, attribute)
            if actual != expected:
                raise ValueError(f"Expected main_weight {attribute} {expected!r}, got {actual!r}.")
        for index in range(len(self.rowwise_data.layout.tensor_shapes)):
            self.get_tensor_view(index).quantize_(main_weight.get_tensor_view(index))

    @property
    def planes(self) -> tuple[DBuffer, DBuffer, DBuffer, DBuffer]:
        """Physical planes in TE's rowwise-data-first order."""
        return self.rowwise_data, self.columnwise_data, self.rowwise_scale, self.columnwise_scale

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether every plane is backed by symmetric memory."""
        return all(plane.is_symmetric_memory for plane in self.planes)

    def reallocate_storage(self) -> None:
        """Restore every plane's backing storage."""
        for plane in self.planes:
            plane.reallocate_storage()

    def release_storage(self) -> None:
        """Release every plane's backing storage while retaining aliases."""
        for plane in self.planes:
            plane.release_storage()

    def view(self, placements: Iterable[Placement]) -> "QuantizedDBuffer":
        """Return a storage-sharing view of every physical plane."""
        placements = tuple(placements)
        if self.rowwise_data.placements == placements:
            return self
        return QuantizedDBuffer._from_planes(
            self.rowwise_data.view(placements),
            self.columnwise_data.view(placements),
            self.rowwise_scale.view(placements),
            self.columnwise_scale.view(_block_atomic_to_flat(placements)),
        )

    def redistribute(
        self, new_placements: Iterable[Placement], *, out: "QuantizedDBuffer | None" = None
    ) -> "QuantizedDBuffer":
        """Redistribute every plane, returning ``out`` when supplied or a new wrapper."""
        new_placements = tuple(new_placements)
        if out is None:
            return self._from_planes(
                self.rowwise_data.redistribute(new_placements),
                self.columnwise_data.redistribute(new_placements),
                self.rowwise_scale.redistribute(new_placements),
                self.columnwise_scale.redistribute(_block_atomic_to_flat(new_placements)),
            )
        if out.mesh != self.mesh:
            raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
        if out.rowwise_data.placements != new_placements:
            raise ValueError(
                "Expected out rowwise-data placements "
                f"{new_placements!r}, got {out.rowwise_data.placements!r}."
            )
        for plane, out_plane in zip(self.planes, out.planes):
            plane.redistribute(out_plane.placements, out=out_plane)
        return out

    def allgather(
        self, mesh_axis: int, *, out: "QuantizedDBuffer | None" = None
    ) -> "QuantizedDBuffer":
        """All-gather every plane, returning ``out`` when supplied or a new wrapper."""
        if out is None:
            return self._from_planes(*(plane.allgather(mesh_axis) for plane in self.planes))
        for plane, out_plane in zip(self.planes, out.planes):
            plane.allgather(mesh_axis, out=out_plane)
        return out
