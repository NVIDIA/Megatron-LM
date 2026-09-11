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
import transformer_engine_torch as tex
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

from .dbuffer import DBuffer
from .layout import GlobalLayout
from .placement import BlockAtomic, Flat

_MXFP8_DTYPE = tex.DType.kFloat8E4M3
_MXFP8_QUANTIZER = MXFP8Quantizer(_MXFP8_DTYPE)
_MXFP8_BLOCK_SIZE = 32


def effective_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return MFSDP's storage dtype for a parameter."""
    return torch.uint8 if isinstance(tensor, MXFP8Tensor) else tensor.dtype


def _rowwise_scale_layout(data_layout: GlobalLayout) -> GlobalLayout:
    """Derive rowwise scale coordinates from the data layout."""
    return GlobalLayout(
        tensor_shapes=tuple(
            torch.Size((shape[0], shape[1] // _MXFP8_BLOCK_SIZE))
            for shape in data_layout.tensor_shapes
        ),
        tensor_to_offset=tuple(
            offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset
        ),
        size=data_layout.size // _MXFP8_BLOCK_SIZE,
        block_size=data_layout.block_size,
    )


def _columnwise_scale_layout(data_layout: GlobalLayout) -> GlobalLayout:
    """Derive columnwise scale coordinates from the data layout."""
    return GlobalLayout(
        tensor_shapes=tuple(
            torch.Size((shape[0] // _MXFP8_BLOCK_SIZE, shape[1]))
            for shape in data_layout.tensor_shapes
        ),
        tensor_to_offset=tuple(
            offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset
        ),
        size=data_layout.size // _MXFP8_BLOCK_SIZE,
    )


def _columnwise_scale_placements(placements: Iterable[Placement]) -> tuple[Placement, ...]:
    """Map weight placements to columnwise-scale coordinates.

    A columnwise scale row describes one 32-row weight block, so it uses
    Flat instead of BlockAtomic(32) to preserve the same shard boundaries.
    """
    placements = tuple(placements)
    return tuple(
        Flat() if isinstance(placement, BlockAtomic) else placement for placement in placements
    )


def _pad_rowwise_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pad rowwise scales to TE's physical allocation shape."""
    shape = _MXFP8_QUANTIZER.get_scale_shape(
        (scale.shape[0], scale.shape[1] * _MXFP8_BLOCK_SIZE), columnwise=False
    )
    if scale.shape == shape:
        return scale
    padded = torch.zeros(shape, dtype=scale.dtype, device=scale.device)
    padded[: scale.shape[0], : scale.shape[1]].copy_(scale)
    return padded


def _pad_columnwise_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pad columnwise scales to TE's physical allocation shape."""
    shape = _MXFP8_QUANTIZER.get_scale_shape(
        (scale.shape[0] * _MXFP8_BLOCK_SIZE, scale.shape[1]), columnwise=True
    )
    if scale.shape == shape:
        return scale
    padded = torch.zeros(shape, dtype=scale.dtype, device=scale.device)
    padded[: scale.shape[0], : scale.shape[1]].copy_(scale)
    return padded


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
        tensor_shapes: Iterable[torch.Size],
        device: torch.device | str,
    ) -> None:
        tensor_shapes = tuple(torch.Size(shape) for shape in tensor_shapes)
        if not tensor_shapes or any(len(shape) != 2 for shape in tensor_shapes):
            raise ValueError("QuantizedDBuffer requires one or more 2D MXFP8 tensor shapes.")
        if any(
            shape[0] % _MXFP8_BLOCK_SIZE or shape[1] % _MXFP8_BLOCK_SIZE for shape in tensor_shapes
        ):
            raise ValueError(
                f"QuantizedDBuffer requires dimensions divisible by {_MXFP8_BLOCK_SIZE}."
            )
        placements = tuple(placements)
        self.rowwise_data = DBuffer.empty(
            mesh, placements, tensor_shapes, torch.uint8, device, block_size=_MXFP8_BLOCK_SIZE
        )
        self.columnwise_data = DBuffer(
            mesh, placements, self.rowwise_data.layout, torch.uint8, device
        )
        self.rowwise_scale = DBuffer(
            mesh, placements, _rowwise_scale_layout(self.rowwise_data.layout), torch.uint8, device
        )
        self.columnwise_scale = DBuffer(
            mesh,
            _columnwise_scale_placements(placements),
            _columnwise_scale_layout(self.rowwise_data.layout),
            torch.uint8,
            device,
        )

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

    def get_local_tensor(self, index: int) -> torch.Tensor:
        """Construct a TE MXFP8 wrapper from this rank's physical-plane views."""
        rowwise_data = self.rowwise_data.get_local_tensor(index)
        rowwise_scale = self.rowwise_scale.get_local_tensor(index)
        columnwise_scale = self.columnwise_scale.get_local_tensor(index)
        return MXFP8Tensor(
            shape=rowwise_data.shape,
            dtype=torch.bfloat16,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=_pad_rowwise_scale(rowwise_scale),
            columnwise_data=self.columnwise_data.get_local_tensor(index),
            columnwise_scale_inv=_pad_columnwise_scale(columnwise_scale),
            fp8_dtype=_MXFP8_DTYPE,
            quantizer=_MXFP8_QUANTIZER,
            with_gemm_swizzled_scales=False,
            device=rowwise_data.device,
        )

    def quantize_(self, main_weight: DBuffer) -> None:
        """Quantize the local FP32 master shard into MXFP8 grouped planes."""
        for index in range(len(self.rowwise_data.layout.tensor_shapes)):
            tensor = self.get_local_tensor(index)
            rowwise_scale = self.rowwise_scale.get_local_tensor(index)
            columnwise_scale = self.columnwise_scale.get_local_tensor(index)
            tensor.quantize_(main_weight.get_local_tensor(index))
            assert tensor._rowwise_scale_inv is not None
            assert tensor._columnwise_scale_inv is not None
            # Unpadded wrapper scales already alias their destination planes.
            if tensor._rowwise_scale_inv.shape != rowwise_scale.shape:
                rowwise_scale.copy_(
                    tensor._rowwise_scale_inv[: rowwise_scale.shape[0], : rowwise_scale.shape[1]]
                )
            if tensor._columnwise_scale_inv.shape != columnwise_scale.shape:
                columnwise_scale.copy_(
                    tensor._columnwise_scale_inv[
                        : columnwise_scale.shape[0], : columnwise_scale.shape[1]
                    ]
                )

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
            self.columnwise_scale.view(_columnwise_scale_placements(placements)),
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
                self.columnwise_scale.redistribute(_columnwise_scale_placements(new_placements)),
            )
        if out.mesh != self.mesh:
            raise ValueError(f"Expected out mesh {self.mesh!r}, got {out.mesh!r}.")
        if out.rowwise_data.placements != new_placements:
            raise ValueError(
                "Expected out rowwise-data placements "
                f"{new_placements!r}, got {out.rowwise_data.placements!r}."
            )
        self.rowwise_data.redistribute(new_placements, out=out.rowwise_data)
        self.columnwise_data.redistribute(new_placements, out=out.columnwise_data)
        self.rowwise_scale.redistribute(new_placements, out=out.rowwise_scale)
        self.columnwise_scale.redistribute(
            _columnwise_scale_placements(new_placements), out=out.columnwise_scale
        )
        return out

    def allgather(
        self, mesh_axis: int, *, out: "QuantizedDBuffer | None" = None
    ) -> "QuantizedDBuffer":
        """All-gather every plane, returning ``out`` when supplied or a new wrapper."""
        result_planes = tuple(
            plane.allgather(mesh_axis, out=None if out is None else out.planes[index])
            for index, plane in enumerate(self.planes)
        )
        return out if out is not None else self._from_planes(*result_planes)
