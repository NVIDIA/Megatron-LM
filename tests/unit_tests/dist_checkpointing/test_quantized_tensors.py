# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, quantized_tensors

fp8 = pytest.importorskip("transformer_engine.pytorch.fp8")
float8_blockwise_tensor = pytest.importorskip(
    "transformer_engine.pytorch.tensor.float8_blockwise_tensor"
)
float8_tensor = pytest.importorskip("transformer_engine.pytorch.tensor.float8_tensor")
mxfp8_tensor = pytest.importorskip("transformer_engine.pytorch.tensor.mxfp8_tensor")
nvfp4_tensor = pytest.importorskip("transformer_engine.pytorch.tensor.nvfp4_tensor")

_SUPPORT_CHECKS = {
    "fp8": fp8.check_fp8_support,
    "fp8_blockwise": fp8.check_fp8_block_scaling_support,
    "mxfp8": fp8.check_mxfp8_support,
    "nvfp4": fp8.check_nvfp4_support,
}


def _quantize(encoding, rows=512, cols=64):
    supported, reason = _SUPPORT_CHECKS[encoding]()
    if not supported:
        pytest.skip(reason)
    values = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    e4m3 = mxfp8_tensor.tex.DType.kFloat8E4M3
    if encoding == "fp8":
        quantizer = float8_tensor.Float8CurrentScalingQuantizer(e4m3, device="cuda")
    elif encoding == "fp8_blockwise":
        quantizer = float8_blockwise_tensor.Float8BlockQuantizer(
            e4m3, rowwise=True, columnwise=False
        )
    elif encoding == "mxfp8":
        quantizer = mxfp8_tensor.MXFP8Quantizer(e4m3, columnwise=False)
    else:
        quantizer = nvfp4_tensor.NVFP4Quantizer()
    return quantizer(values)


def _shares_storage(view, tensor):
    return (
        view._rowwise_data.untyped_storage().data_ptr()
        == tensor._rowwise_data.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    ("encoding", "start", "length"), [("mxfp8", 128, 256), ("nvfp4", 128, 256), ("nvfp4", 64, 128)]
)
def test_row_slices_stay_packed_views(encoding, start, length):
    tensor = _quantize(encoding)
    with quantized_tensors.keep_quantized_tensors():
        rows = quantized_tensors.narrow(tensor, 0, start, length)
    assert type(rows) is type(tensor) and _shares_storage(rows, tensor)
    assert torch.equal(rows.dequantize(), tensor.dequantize()[start : start + length])


def test_row_chunks_of_mxfp8_stay_packed_views():
    tensor = _quantize("mxfp8")
    with quantized_tensors.keep_quantized_tensors():
        chunks = quantized_tensors.chunk(tensor, 2, dim=0)
    assert all(type(chunk) is type(tensor) and _shares_storage(chunk, tensor) for chunk in chunks)
    assert torch.equal(torch.cat([chunk.dequantize() for chunk in chunks]), tensor.dequantize())


def test_sharded_tensors_narrow_quantized_data_to_packed_views():
    tensor = _quantize("mxfp8")
    sharded = ShardedTensor.from_rank_offsets("weight", tensor, (0, 1, 2))
    with quantized_tensors.keep_quantized_tensors():
        (narrowed,) = sharded.narrow(0, 256, 256)
    assert type(narrowed.data) is type(tensor) and _shares_storage(narrowed.data, tensor)
    assert narrowed.global_shape == (512, 64) and narrowed.global_offset == (256, 0)


def test_tensors_are_sliced_by_torch_outside_the_context():
    values = torch.arange(8.0).reshape(4, 2)
    assert torch.equal(quantized_tensors.narrow(values, 0, 1, 2), values[1:3])
    assert all(
        torch.equal(a, b)
        for a, b in zip(quantized_tensors.chunk(values, 2), torch.chunk(values, 2))
    )


def test_row_slices_that_cannot_stay_packed_are_sliced_by_torch():
    tensor = _quantize("mxfp8")
    with quantized_tensors.keep_quantized_tensors():
        rows = quantized_tensors.narrow(tensor, 0, 64, 128)
    values = rows.dequantize() if hasattr(rows, "_rowwise_data") else rows
    assert torch.equal(values, tensor.dequantize()[64:192])


@pytest.mark.parametrize("encoding", ["fp8", "mxfp8", "nvfp4"])
def test_tensors_in_formats_that_checkpoints_store_are_kept_packed(encoding):
    tensor = _quantize(encoding)
    assert quantized_tensors.packed_or_dequantized(tensor) is tensor
    assert not quantized_tensors.keeps_packed(tensor)
    with quantized_tensors.keep_quantized_tensors():
        assert quantized_tensors.keeps_packed(tensor)


def test_tensors_in_other_formats_are_dequantized():
    tensor = _quantize("fp8_blockwise")
    values = quantized_tensors.packed_or_dequantized(tensor)
    assert type(values) is torch.Tensor and torch.equal(values, tensor.dequantize())
    with quantized_tensors.keep_quantized_tensors():
        assert not quantized_tensors.keeps_packed(tensor)
