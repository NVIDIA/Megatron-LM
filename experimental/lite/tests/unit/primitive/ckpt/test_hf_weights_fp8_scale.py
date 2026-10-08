# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch
from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader, _read_hf_tensors


def _reader(tensors):
    reader = object.__new__(SafeTensorReader)
    reader.device = torch.device("cpu")
    reader._cached_request = None
    reader._cached_tensor = None
    reader.has_tensor = lambda name: name in tensors
    reader._get_raw_tensor = lambda name, device: tensors[name].to(device)
    return reader


def test_safe_tensor_reader_dequantizes_fp8_weight_with_block_scale() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required")

    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).to(torch.float8_e4m3fn)
    tensors = {
        "w.weight": weight,
        "w.scale": torch.full((1, 1), 2.0, dtype=torch.float32),
    }
    actual = _reader(tensors).get_tensor(
        "w.weight",
        target_shape=torch.Size((2, 2)),
        target_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, weight.float() * 2.0)


@pytest.mark.parametrize(
    "source_dtype",
    [
        pytest.param(getattr(torch, "float8_e4m3fn", None), id="fp8"),
        pytest.param(torch.int8, id="packed-fp4"),
    ],
)
def test_safe_tensor_reader_rejects_one_byte_quantized_weight_without_scale(
    source_dtype,
) -> None:
    if source_dtype is None:
        pytest.skip("torch float8_e4m3fn is required")

    tensors = {
        "missing.weight": torch.ones((2, 2), dtype=source_dtype),
    }

    with pytest.raises(
        RuntimeError,
        match=r"missing\.weight.*missing\.weight_scale_inv.*missing\.scale",
    ):
        _reader(tensors).get_tensor(
            "missing.weight",
            target_shape=torch.Size((2, 2)),
            target_dtype=torch.bfloat16,
        )


def test_multi_source_mapping_dequantizes_each_fp8_block_scaled_source() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required")

    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    second = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to(torch.float8_e4m3fn)
    tensors = {
        "first.weight": first,
        "first.scale": torch.full((1, 1), 2.0),
        "second.weight": second,
        "second.scale": torch.full((1, 1), 0.5),
    }

    class Spec:
        @staticmethod
        def hf_to_native(native_name, sources):
            assert native_name == "fused.weight"
            return torch.cat(sources, dim=0)

    sources = _read_hf_tensors(
        _reader(tensors),
        Spec(),
        "fused.weight",
        ["first.weight", "second.weight"],
        torch.empty((4, 2), dtype=torch.bfloat16),
    )
    actual = Spec.hf_to_native("fused.weight", sources)

    torch.testing.assert_close(
        actual,
        torch.cat([first.float() * 2.0, second.float() * 0.5], dim=0),
    )


def test_multi_source_block_scale_rejects_shape_that_cannot_be_inferred() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required")

    tensors = {
        "first.weight": torch.ones((1, 2), dtype=torch.float8_e4m3fn),
        "first.scale": torch.ones((1, 1)),
        "second.weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        "second.scale": torch.ones((1, 1)),
    }

    with pytest.raises(
        RuntimeError,
        match=r"first\.weight.*target_shape is required.*first\.scale",
    ):
        _read_hf_tensors(
            _reader(tensors),
            object(),
            "fused.weight",
            ["first.weight", "second.weight"],
            torch.empty((3, 2), dtype=torch.bfloat16),
        )
