# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP8 (1x32 E8M0) and compressed-tensors NVFP4 decoding in SafeTensorReader."""

from __future__ import annotations

import torch

from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _reader(tensors):
    reader = object.__new__(SafeTensorReader)
    reader.device = torch.device("cpu")
    reader._cached_request = None
    reader._cached_tensor = None
    reader.has_tensor = lambda name: name in tensors
    reader._get_raw_tensor = lambda name, device: tensors[name].to(device)
    return reader


def test_mxfp8_uses_one_e8m0_exponent_per_32_input_columns() -> None:
    gen = torch.Generator().manual_seed(0)
    weight = (torch.randn(3, 70, generator=gen) * 50).to(torch.float8_e4m3fn)
    exponents = torch.tensor([[127, 120, 130], [100, 127, 140], [127, 127, 1]], dtype=torch.uint8)
    actual = _reader({"w.weight": weight, "w.weight_scale_inv": exponents}).get_tensor(
        "w.weight", target_dtype=torch.bfloat16
    )

    expected = torch.empty(3, 70)
    for row in range(3):
        for col in range(70):
            expected[row, col] = float(weight[row, col].float()) * 2.0 ** (
                int(exponents[row, col // 32]) - 127
            )
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_deepseek_style_128_block_scale_inv_is_unchanged() -> None:
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), 0.25, dtype=torch.float32)
    actual = _reader({"w.weight": weight, "w.weight_scale_inv": scale}).get_tensor("w.weight")
    assert torch.equal(actual, weight.float() * 0.25)


def test_nvfp4_decodes_low_nibble_first_and_divides_by_global_scale() -> None:
    gen = torch.Generator().manual_seed(1)
    rows, cols = 2, 32
    packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=gen)
    packed[0, 0] = 0x08  # code 8 (-0.0) in the low nibble, code 0 in the high nibble
    block_scale = (torch.rand(rows, cols // 16, generator=gen) * 4 + 0.5).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(448.0 * 6.0 / 3.0, dtype=torch.float32)
    tensors = {
        "e.weight_packed": packed,
        "e.weight_scale": block_scale,
        "e.weight_global_scale": global_scale,
    }
    actual = _reader(tensors).get_tensor("e.weight")

    expected = torch.empty(rows, cols)
    for row in range(rows):
        for col in range(cols):
            byte = int(packed[row, col // 2])
            code = byte & 0xF if col % 2 == 0 else byte >> 4
            value = torch.tensor(E2M1[code]) * block_scale[row, col // 16].float()
            expected[row, col] = value / global_scale
    assert torch.equal(actual, expected)
    assert torch.signbit(actual[0, 0]) and not torch.signbit(actual[0, 1])
