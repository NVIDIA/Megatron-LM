# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Portions of this file are ported from nvrx_checkpointing
# (https://github.com/ankurv-nvidia/nvrx_checkpointing, quantized_snapshot.py at commit
# 149cfe8b571010c33f39da15c02bfb415606828b), licensed under the Apache License, Version 2.0.

"""Checkpoint views of Transformer Engine quantized tensors that keep their packed storage.

A checkpoint engine that snapshots quantized parameters and dequantizes them on the host can save
them without first materializing high-precision copies on the GPU. While `keep_quantized_tensors`
is active, the sharded state dicts of models keep quantized parameters in their packed form: row
slices of MXFP8 and NVFP4 tensors are views of their packed storage, and GTP native FP8 parameters
in a format that checkpoints store are not dequantized.
"""

from contextlib import contextmanager

import torch

_MXFP8_BLOCK_SIZE = 32
_NVFP4_BLOCK_SIZE = 16
_SCALE_ROW_ALIGNMENT = 128

_keep_quantized = False


@contextmanager
def keep_quantized_tensors(enabled: bool = True):
    """Keeps quantized tensors packed in the sharded state dicts built within the context."""
    global _keep_quantized
    previous, _keep_quantized = _keep_quantized, enabled
    try:
        yield
    finally:
        _keep_quantized = previous


def _has_attributes(value, attributes) -> bool:
    return all(hasattr(value, attribute) for attribute in attributes)


def _is_mxfp8(tensor) -> bool:
    return _has_attributes(
        tensor,
        (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_columnwise_data",
            "_columnwise_scale_inv",
            "_fp8_dtype",
            "_with_gemm_swizzled_scales",
            "get_metadata",
        ),
    )


def _is_nvfp4(tensor) -> bool:
    return _has_attributes(
        tensor,
        (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_amax_rowwise",
            "_fp4_dtype",
            "_with_gemm_swizzled_scales",
            "get_metadata",
        ),
    )


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _rows_in_bounds(tensor, start: int, length: int, block_size: int) -> bool:
    return (
        tensor.dim() == 2
        and 0 <= start < start + length <= tensor.shape[0]
        and tensor.shape[1] % block_size == 0
        and tensor._rowwise_data is not None
        and tensor._rowwise_scale_inv is not None
        and start + _round_up(length, _SCALE_ROW_ALIGNMENT) <= tensor._rowwise_scale_inv.shape[0]
    )


def _mxfp8_rows_are_packed(tensor, start: int, length: int) -> bool:
    return (
        _rows_in_bounds(tensor, start, length, _MXFP8_BLOCK_SIZE)
        and not tensor._with_gemm_swizzled_scales
        and start % _SCALE_ROW_ALIGNMENT == 0
    )


def _nvfp4_rows_are_packed(tensor, start: int, length: int) -> bool:
    return (
        _rows_in_bounds(tensor, start, length, _NVFP4_BLOCK_SIZE)
        and (not tensor._with_gemm_swizzled_scales or start % _SCALE_ROW_ALIGNMENT == 0)
        and (
            not tensor.get_metadata().get("row_scaled_nvfp4", False)
            or tensor._amax_rowwise is not None
        )
    )


def _slice_mxfp8_rows(tensor, start: int, length: int):
    """Rows [start, start + length) of an MXFP8 tensor, as a view of its packed storage."""
    metadata = tensor.get_metadata()
    metadata["rowwise_data"] = tensor._rowwise_data.narrow(0, start, length)
    metadata["rowwise_scale_inv"] = tensor._rowwise_scale_inv.narrow(
        0, start, _round_up(length, _SCALE_ROW_ALIGNMENT)
    )
    metadata["columnwise_data"] = None
    metadata["columnwise_scale_inv"] = None
    metadata["quantizer"] = None
    return type(tensor)(**metadata, shape=(length, tensor.shape[1]), dtype=tensor.dtype)


def _slice_nvfp4_rows(tensor, start: int, length: int):
    """Rows [start, start + length) of an NVFP4 tensor, as a view of its packed storage."""
    metadata = tensor.get_metadata()
    metadata["rowwise_data"] = tensor._rowwise_data.narrow(0, start, length)
    metadata["rowwise_scale_inv"] = tensor._rowwise_scale_inv.narrow(
        0, start, _round_up(length, _SCALE_ROW_ALIGNMENT)
    )
    if metadata.get("row_scaled_nvfp4", False):
        metadata["amax_rowwise"] = tensor._amax_rowwise.narrow(0, start, length)
    metadata["columnwise_data"] = None
    metadata["columnwise_scale_inv"] = None
    metadata["amax_columnwise"] = None
    metadata["quantizer"] = None
    return type(tensor)(**metadata, shape=(length, tensor.shape[1]), dtype=tensor.dtype)


def narrow(tensor, dim: int, start: int, length: int):
    """`tensor.narrow(dim, start, length)`, keeping quantized row slices packed when asked to."""
    if _keep_quantized and dim % tensor.dim() == 0:
        if _is_mxfp8(tensor) and _mxfp8_rows_are_packed(tensor, start, length):
            return _slice_mxfp8_rows(tensor, start, length)
        if _is_nvfp4(tensor) and _nvfp4_rows_are_packed(tensor, start, length):
            return _slice_nvfp4_rows(tensor, start, length)
    return tensor.narrow(dim, start, length)


def chunk(tensor, chunks: int, dim: int = 0):
    """`torch.chunk(tensor, chunks, dim)`, keeping MXFP8 row chunks packed when asked to."""
    if _keep_quantized and dim % tensor.dim() == 0 and _is_mxfp8(tensor):
        rows = tensor.shape[0]
        chunk_rows = -(-rows // chunks)
        slices = [(start, min(chunk_rows, rows - start)) for start in range(0, rows, chunk_rows)]
        if all(_mxfp8_rows_are_packed(tensor, start, length) for start, length in slices):
            return tuple(_slice_mxfp8_rows(tensor, start, length) for start, length in slices)
    return torch.chunk(tensor, chunks, dim)


def _is_dtype(value, number: str, name: str) -> bool:
    return str(value) == number or name in str(value)


def _has_components(tensor, **dtypes) -> bool:
    return all(
        getattr(tensor, name, None) is not None and getattr(tensor, name).dtype == dtype
        for name, dtype in dtypes.items()
    )


def _is_kept_packed(tensor) -> bool:
    if tensor.dtype != torch.bfloat16:
        return False
    if hasattr(tensor, "_fp4_dtype"):
        return (
            _is_dtype(tensor._fp4_dtype, "10", "kFloat4E2M1")
            and getattr(tensor, "_nvfp4_e4m3_max", 448) in (448, 256)
            and hasattr(tensor, "_with_gemm_swizzled_scales")
            and _has_components(
                tensor,
                _rowwise_data=torch.uint8,
                _rowwise_scale_inv=torch.uint8,
                _amax_rowwise=torch.float32,
            )
        )
    if not _is_dtype(getattr(tensor, "_fp8_dtype", None), "7", "kFloat8E4M3"):
        return False
    if _has_attributes(tensor, ("_rowwise_data", "_rowwise_scale_inv")):
        return not getattr(tensor, "_with_gemm_swizzled_scales", False) and _has_components(
            tensor, _rowwise_data=torch.uint8, _rowwise_scale_inv=torch.uint8
        )
    return (
        _has_components(tensor, _data=torch.uint8, _scale_inv=torch.float32)
        and tensor._scale_inv.numel() == 1
    )


def keeps_packed(tensor) -> bool:
    """Whether the sharded state dicts being built keep `tensor` in its packed form."""
    return _keep_quantized and _is_kept_packed(tensor)


def packed_or_dequantized(tensor):
    """`tensor` if checkpoints keep it packed, else its dequantized values, as written on save.

    Checkpoints keep bf16 tensors packed as E4M3 FP8 with one scale, as E4M3 MXFP8 with compact
    scales, and as NVFP4, the formats that the checkpoint engine reads. Other quantized tensors,
    like FP8 tensors with block scales, are dequantized.
    """
    if "dequantize" in type(tensor).__dict__ and not _is_kept_packed(tensor):
        return tensor.dequantize()
    return tensor
