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

import functools
from collections.abc import Iterable

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
except ImportError as exc:
    # Keep "UnavailableError": .gitlab/scripts/check_imports.py imports this module
    # directly and uses that string to classify missing optional dependencies as graceful.
    raise ImportError(
        "UnavailableError: QuantizedDBuffer requires Transformer Engine MXFP8 support"
    ) from exc

from . import layout_builder
from .dbuffer import DBuffer
from .layout import GlobalLayout, Shape
from .placement import BlockAtomic, RowAtomic, ShardLayout

_MXFP8_DTYPE = tex.DType.kFloat8E4M3
_MXFP8_BLOCK_SIZE = 32


@functools.cache
def _get_quantizer(*, rowwise: bool, columnwise: bool) -> MXFP8Quantizer:
    """Return a shared MXFP8 quantizer with the given usages."""
    quantizer = MXFP8Quantizer(_MXFP8_DTYPE, rowwise=rowwise, columnwise=columnwise)
    quantizer.optimize_for_gemm = False
    return quantizer


def effective_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return MFSDP's storage dtype for a parameter."""
    return torch.uint8 if isinstance(tensor, MXFP8Tensor) else tensor.dtype


def _rowwise_scale_layout(data_layout: GlobalLayout) -> GlobalLayout:
    """Derive rowwise scales from the layout shared by rowwise_data and columnwise_data."""
    return GlobalLayout(
        tensor_shapes=(
            (shape[0], shape[1] // _MXFP8_BLOCK_SIZE) for shape in data_layout.tensor_shapes
        ),
        tensor_to_offset=(offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset),
        size=data_layout.size // _MXFP8_BLOCK_SIZE,
        rank_to_offset=(offset // _MXFP8_BLOCK_SIZE for offset in data_layout.rank_to_offset),
    )


def _columnwise_scale_layout(data_layout: GlobalLayout) -> GlobalLayout:
    """Derive columnwise scales from the layout shared by rowwise_data and columnwise_data."""
    return GlobalLayout(
        tensor_shapes=(
            (shape[0] // _MXFP8_BLOCK_SIZE, shape[1]) for shape in data_layout.tensor_shapes
        ),
        tensor_to_offset=(offset // _MXFP8_BLOCK_SIZE for offset in data_layout.tensor_to_offset),
        size=data_layout.size // _MXFP8_BLOCK_SIZE,
        rank_to_offset=(offset // _MXFP8_BLOCK_SIZE for offset in data_layout.rank_to_offset),
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


class QuantizedDBuffer:
    """The MFSDP storage and lifecycle for one TE MXFP8 tensor.

    The data layout owns the parameter-to-rank partition. Scale planes use its
    compact MXFP8 coordinates, so every plane on a rank describes the same
    local data rows. Rows must be divisible by 32 and columns by 128, so
    unswizzled scales need no padding. Requires TE fixes #3710 (quantization)
    and #3768 (GEMM preparation).
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
        tensor_shapes = layout.tensor_shapes
        if not tensor_shapes or any(len(shape) != 2 for shape in tensor_shapes):
            raise ValueError("QuantizedDBuffer requires one or more 2D MXFP8 tensor shapes.")
        if any(shape[0] % _MXFP8_BLOCK_SIZE or shape[1] % 128 for shape in tensor_shapes):
            raise ValueError(
                "QuantizedDBuffer requires rows divisible by 32 and columns divisible by 128."
            )
        layout.validate_for_row_atomic(block_size=_MXFP8_BLOCK_SIZE)
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
        *,
        shard_layout: ShardLayout = ShardLayout.PADDED,
    ) -> "QuantizedDBuffer":
        """Build an MXFP8 layout from logical tensor shapes and allocate its planes.

        Args:
            mesh: Device mesh whose dimensions correspond to ``placements``.
            placements: Per-mesh-axis DBuffer placements.
            tensor_shapes: Logical 2D shapes that the buffer manages.
            device: Device for the local planes.
            shard_layout: Padded (equal) or unpadded (32-row-block-aligned, unequal) rank
                segments; see ``DBuffer.empty``.
        """
        layout = layout_builder.build_row_atomic_layout(
            tensor_shapes,
            dp_size=mesh.size(),
            block_size=_MXFP8_BLOCK_SIZE,
            shard_layout=shard_layout,
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

    @property
    def is_rowwise_allocated(self) -> bool:
        """Whether rowwise data and scales are allocated."""
        return self.rowwise_data.is_allocated and self.rowwise_scale.is_allocated

    @property
    def is_columnwise_allocated(self) -> bool:
        """Whether columnwise data and scales are allocated."""
        return self.columnwise_data.is_allocated and self.columnwise_scale.is_allocated

    def get_tensor_view(self, index: int) -> MXFP8Tensor:
        """Return a compact, unswizzled MXFP8 view that aliases every allocated plane pair.

        Released planes are omitted, so the view never reads storage that
        ``release_storage()`` freed.
        """
        rowwise, columnwise = self.is_rowwise_allocated, self.is_columnwise_allocated
        if not (rowwise or columnwise):
            raise RuntimeError("QuantizedDBuffer has no allocated rowwise or columnwise planes.")
        rowwise_data = self.rowwise_data.get_tensor_view(index) if rowwise else None
        columnwise_data = self.columnwise_data.get_tensor_view(index) if columnwise else None
        data = rowwise_data if rowwise_data is not None else columnwise_data
        return MXFP8Tensor(
            shape=data.shape,
            dtype=torch.bfloat16,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=(self.rowwise_scale.get_tensor_view(index) if rowwise else None),
            columnwise_data=columnwise_data,
            columnwise_scale_inv=(
                self.columnwise_scale.get_tensor_view(index) if columnwise else None
            ),
            fp8_dtype=_MXFP8_DTYPE,
            quantizer=_get_quantizer(rowwise=rowwise, columnwise=columnwise),
            with_gemm_swizzled_scales=False,
            device=data.device,
        )

    def quantize_(self, main_weight: DBuffer) -> None:
        """Quantize a local master shard with matching mesh, placements, layout, and device."""
        for attribute in ("mesh", "placements", "layout", "device"):
            expected = getattr(self.rowwise_data, attribute)
            actual = getattr(main_weight, attribute)
            if actual != expected:
                raise ValueError(f"Expected main_weight {attribute} {expected!r}, got {actual!r}.")
        assert (
            self.is_rowwise_allocated and self.is_columnwise_allocated
        ), "quantize_() requires both rowwise and columnwise planes to be allocated."

        for index in range(len(self.rowwise_data.layout.tensor_shapes)):
            self.get_tensor_view(index).quantize_(main_weight.get_tensor_view(index))

    @property
    def planes(self) -> tuple[DBuffer, DBuffer, DBuffer, DBuffer]:
        """Physical planes in TE's rowwise-data-first order."""
        return self.rowwise_data, self.columnwise_data, self.rowwise_scale, self.columnwise_scale

    def planes_for(self, *, rowwise: bool, columnwise: bool) -> tuple[DBuffer, ...]:
        """Physical data and scale planes of the selected representations."""
        planes: tuple[DBuffer, ...] = ()
        if rowwise:
            planes += (self.rowwise_data, self.rowwise_scale)
        if columnwise:
            planes += (self.columnwise_data, self.columnwise_scale)
        return planes

    @property
    def is_symmetric_memory(self) -> bool:
        """Whether every plane is backed by symmetric memory."""
        return all(plane.is_symmetric_memory for plane in self.planes)

    def reallocate_storage(self) -> None:
        """Restore every plane's backing storage."""
        for plane in self.planes:
            plane.reallocate_storage()

    def release_storage(self, *, rowwise_only: bool = False) -> None:
        """Release all plane storage, or only rowwise storage, while retaining aliases."""
        # Preserve Storage aliases that autograd may have saved for backward.
        for plane in self.planes_for(rowwise=True, columnwise=not rowwise_only):
            plane.release_storage()

    def can_redistribute_locally(self, placements: Iterable[Placement]) -> bool:
        """Whether every plane can preserve its values through a local view.

        All planes use the same mesh partition, so the rowwise-data check suffices.
        """
        return self.rowwise_data.can_redistribute_locally(placements)

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

    def is_alias_of(self, other: "QuantizedDBuffer") -> bool:
        """Whether every corresponding plane shares backing storage."""
        return all(
            plane.is_alias_of(other_plane) for plane, other_plane in zip(self.planes, other.planes)
        )

    def redistribute(
        self,
        new_placements: Iterable[Placement],
        *,
        out: "QuantizedDBuffer | None" = None,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> "QuantizedDBuffer":
        """Redistribute the selected planes, returning ``out`` or a new wrapper."""
        new_placements = tuple(new_placements)
        if out is None:
            if not (rowwise and columnwise):
                raise NotImplementedError("redistribute() without `out` requires every plane.")
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
        for plane, out_plane in zip(
            self.planes_for(rowwise=rowwise, columnwise=columnwise),
            out.planes_for(rowwise=rowwise, columnwise=columnwise),
        ):
            plane.redistribute(out_plane.placements, out=out_plane)
        return out

    def allgather(
        self, mesh_axis: int | Iterable[int], *, out: "QuantizedDBuffer | None" = None
    ) -> "QuantizedDBuffer":
        """Gather one or more axes in every plane, reusing ``out`` when supplied."""
        # Materialize generators once so every plane gathers the same axes.
        if not isinstance(mesh_axis, int):
            mesh_axis = tuple(mesh_axis)
        if out is None:
            return self._from_planes(*(plane.allgather(mesh_axis) for plane in self.planes))
        for plane, out_plane in zip(self.planes, out.planes):
            plane.allgather(mesh_axis, out=out_plane)
        return out
