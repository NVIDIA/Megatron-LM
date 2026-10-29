# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block-major key caches that LiteTopK plugins gather from.

LiteTopK plugins read the keys of a sequence from a paged cache in the SGLang layout: blocks of
64 rows, each block holding the 64 quantized value records followed by the 64 four-byte scales,
addressed through a block table. :class:`KeyCachePool` owns reusable caches, block tables and
the gather destinations a plugin writes, one set per device, stream and operand format.
:func:`pack_key_cache` only copies bytes: the keys are quantized once, by the same function
that quantizes the reference selector's operands, so both selectors see identical keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import Tensor

__all__ = [
    "KeyCachePool",
    "KeyCacheViews",
    "block_major_cache_views",
    "copy_block_major_rows",
    "pack_key_cache",
]

KeyCacheFormat = Literal["fp8"]

_CACHE_BLOCK_SIZE = 64
_SCALE_BYTES = 4
_CAPACITY_ROWS = 4096


class KeyCacheViews(NamedTuple):
    """The views of one pooled key cache sized for a sequence.

    Attributes:
        cache: uint8 ``[blocks, 64, value_bytes + 4]``; each block holds 64 value records, then
            their 64 scales.
        keys: The gather destination of the values, ``[rows, value_bytes]``: float8_e4m3fn for
            the ``fp8`` format.
        scales: The gather destination of the scales, uint8 ``[rows, 4]`` (a float32 scale for
            ``fp8``).
        block_table: int32 ``[1, blocks]``, the identity block table ``0, 1, ..., blocks - 1``.
    """

    cache: Tensor
    keys: Tensor
    scales: Tensor
    block_table: Tensor


@dataclass
class _Workspace:
    capacity: int
    value_bytes: int
    cache: Tensor
    keys: Tensor
    scales: Tensor
    blocks: Tensor

    def nbytes(self) -> int:
        tensors = (self.cache, self.keys, self.scales, self.blocks)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _value_layout(fmt: str, head_dim: int) -> tuple[int, torch.dtype]:
    if fmt == "fp8":
        return head_dim, torch.float8_e4m3fn
    raise ValueError(f"key cache format must be 'fp8', got {fmt!r}")


def _device_key(device: torch.device) -> tuple[torch.device, str]:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device, str(device)


def _stream_id(device: torch.device) -> int:
    """Return a stable execution-stream identity without touching CUDA for CPU devices."""
    return int(torch.cuda.current_stream(device).cuda_stream) if device.type == "cuda" else 0


def block_major_cache_views(cache: Tensor, value_bytes: int) -> tuple[Tensor, Tensor]:
    """View every cache block as its value region followed by its scale region.

    Args:
        cache: Contiguous uint8 ``[blocks, 64, value_bytes + 4]``.
        value_bytes: The bytes of one value record.

    Returns:
        ``(values, scales)``: uint8 views ``[blocks, 64, value_bytes]`` and ``[blocks, 64, 4]``.

    Raises:
        ValueError: If ``cache`` does not have that shape or is not contiguous.
    """
    expected_record_bytes = int(value_bytes) + _SCALE_BYTES
    if (
        cache.ndim != 3
        or cache.dtype != torch.uint8
        or cache.size(1) != _CACHE_BLOCK_SIZE
        or cache.size(2) != expected_record_bytes
        or not cache.is_contiguous()
    ):
        raise ValueError(
            "LiteTopK key cache must be contiguous uint8 "
            f"[blocks,{_CACHE_BLOCK_SIZE},{expected_record_bytes}], got {cache.dtype} "
            f"{tuple(cache.shape)}"
        )
    flat_blocks = cache.view(cache.size(0), _CACHE_BLOCK_SIZE * expected_record_bytes)
    values = flat_blocks[:, : _CACHE_BLOCK_SIZE * value_bytes].view(
        cache.size(0), _CACHE_BLOCK_SIZE, value_bytes
    )
    scales = flat_blocks[:, _CACHE_BLOCK_SIZE * value_bytes :].view(
        cache.size(0), _CACHE_BLOCK_SIZE, _SCALE_BYTES
    )
    return values, scales


def copy_block_major_rows(destination: Tensor, source: Tensor, start_row: int = 0) -> None:
    """Copy contiguous rows into a block-major destination view, starting at an aligned row.

    Rows of the last, partial block past the copied rows are zeroed.

    Args:
        destination: ``[blocks, 64, width]``, a region view from
            :func:`block_major_cache_views`.
        source: ``[rows, width]`` rows of the same dtype.
        start_row: The first destination row; a multiple of 64.

    Raises:
        ValueError: If ``start_row`` is not aligned or the row widths differ.
    """
    start_row = int(start_row)
    if start_row < 0 or start_row % _CACHE_BLOCK_SIZE != 0:
        raise ValueError(f"block-major copy start must align to {_CACHE_BLOCK_SIZE} rows")
    if source.ndim != 2 or destination.ndim != 3 or source.size(1) != destination.size(2):
        raise ValueError("block-major source/destination row widths must match")
    first_block = start_row // _CACHE_BLOCK_SIZE
    full_blocks, tail_rows = divmod(int(source.size(0)), _CACHE_BLOCK_SIZE)
    if full_blocks:
        destination[first_block : first_block + full_blocks].copy_(
            source[: full_blocks * _CACHE_BLOCK_SIZE].view(
                full_blocks, _CACHE_BLOCK_SIZE, source.size(1)
            )
        )
    if tail_rows:
        tail_block = first_block + full_blocks
        destination[tail_block].zero_()
        destination[tail_block, :tail_rows].copy_(source[-tail_rows:])


def pack_key_cache(cache: Tensor, values: Tensor, scales: Tensor, *, start_row: int = 0) -> None:
    """Copy quantized key rows and their scales into a block-major cache.

    Args:
        cache: ``KeyCacheViews.cache`` of a pooled cache.
        values: Contiguous ``[rows, value_bytes]`` with one-byte elements: the ``float8_e4m3fn``
            rows of ``quantize_indexer_fp8_rows``.
        scales: Contiguous ``[rows]`` with four-byte elements: the float32 row scales of the
            same call.
        start_row: The first cache row to write; a multiple of 64, which lets a caller pack a
            long sequence in chunks.

    Raises:
        ValueError: If the shapes, element sizes or devices do not match the cache, or the rows
            do not fit.
    """
    value_bytes = int(cache.size(-1)) - _SCALE_BYTES if cache.ndim == 3 else -1
    rows = int(values.size(0)) if values.ndim == 2 else -1
    if (
        rows < 0
        or values.size(1) != value_bytes
        or values.element_size() != 1
        or not values.is_contiguous()
        or scales.shape != (rows,)
        or scales.element_size() != _SCALE_BYTES
        or not scales.is_contiguous()
    ):
        raise ValueError(
            f"expected contiguous values [rows,{value_bytes}] with one-byte elements and "
            f"scales [rows] with four-byte elements, got {values.dtype} {tuple(values.shape)} "
            f"and {scales.dtype} {tuple(scales.shape)}"
        )
    if values.device != cache.device or scales.device != cache.device:
        raise ValueError(
            f"key rows on {values.device} and {scales.device} cannot be packed into a cache on "
            f"{cache.device}"
        )
    if int(start_row) + rows > cache.size(0) * _CACHE_BLOCK_SIZE:
        raise ValueError(
            f"rows [{start_row}, {int(start_row) + rows}) do not fit a cache of "
            f"{cache.size(0) * _CACHE_BLOCK_SIZE} rows"
        )
    cache_values, cache_scales = block_major_cache_views(cache, value_bytes)
    copy_block_major_rows(cache_values, values.view(torch.uint8), start_row)
    copy_block_major_rows(
        cache_scales, scales.view(torch.uint8).view(rows, _SCALE_BYTES), start_row
    )


class KeyCachePool:
    """Reusable block-major key caches, one per device, stream and operand format.

    Selectors on different streams get separate caches, so an overlapped forward cannot
    overwrite keys that another stream's selector still reads. A cache grows in steps of 4096
    rows and is reused by later, shorter sequences until :meth:`release`.
    """

    def __init__(self) -> None:
        self._workspaces: dict[tuple[str, int, str], _Workspace] = {}

    def acquire(
        self, fmt: KeyCacheFormat, rows: int, device: torch.device, *, head_dim: int = 128
    ) -> KeyCacheViews:
        """Return cache views for a sequence of ``rows`` keys on the current stream.

        The views alias the pooled storage: their contents are undefined until packed, and the
        next :meth:`acquire` on the same device, stream and format returns the same storage.

        Args:
            fmt: The operand format, ``fp8``.
            rows: The number of keys, at least 1.
            device: The device of the cache.
            head_dim: The indexer head dimension.

        Returns:
            The cache, gather destinations and block table for ``rows`` keys.

        Raises:
            ValueError: If ``fmt``, ``rows`` or ``head_dim`` is invalid.
        """
        value_bytes, value_dtype = _value_layout(fmt, head_dim)
        rows = int(rows)
        if rows <= 0:
            raise ValueError(f"a key cache needs at least one row, got {rows}")
        device, device_key = _device_key(device)
        key = (device_key, _stream_id(device), fmt)
        workspace = self._workspaces.get(key)
        if workspace is None or workspace.capacity < rows or workspace.value_bytes != value_bytes:
            capacity = -(-rows // _CAPACITY_ROWS) * _CAPACITY_ROWS
            block_capacity = capacity // _CACHE_BLOCK_SIZE
            workspace = _Workspace(
                capacity=capacity,
                value_bytes=value_bytes,
                cache=torch.empty(
                    (block_capacity, _CACHE_BLOCK_SIZE, value_bytes + _SCALE_BYTES),
                    dtype=torch.uint8,
                    device=device,
                ),
                keys=torch.empty((capacity, value_bytes), dtype=value_dtype, device=device),
                scales=torch.empty((capacity, _SCALE_BYTES), dtype=torch.uint8, device=device),
                blocks=torch.arange(block_capacity, dtype=torch.int32, device=device),
            )
            self._workspaces[key] = workspace
        active_blocks = -(-rows // _CACHE_BLOCK_SIZE)
        return KeyCacheViews(
            cache=workspace.cache[:active_blocks],
            keys=workspace.keys[:rows],
            scales=workspace.scales[:rows],
            block_table=workspace.blocks[:active_blocks].view(1, active_blocks),
        )

    def allocated_bytes(self, device: torch.device | None = None) -> int:
        """Return the bytes held by the pool, on ``device`` or on every device."""
        device_key = None if device is None else _device_key(device)[1]
        return sum(
            workspace.nbytes()
            for (workspace_device, _, _), workspace in self._workspaces.items()
            if device_key is None or workspace_device == device_key
        )

    def release(self, device: torch.device | None = None) -> None:
        """Drop the pooled caches of ``device``, or of every device when None.

        The memory returns to the PyTorch caching allocator once no view references it.
        """
        device_key = None if device is None else _device_key(device)[1]
        for key in list(self._workspaces):
            if device_key is None or key[0] == device_key:
                del self._workspaces[key]
